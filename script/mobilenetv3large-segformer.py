"""
Hybrid MobileNetV3-Large + SegFormer-B4 for BigPlants-100 classification.

Concept:
- SegFormer generates semantic foreground masks (plant-related regions)
- Images are masked to reduce background noise
- MobileNetV3-Large classifies 100 plant species

Key Features:
- Dataset curation (cap 100 images/class, prioritize leaf/flower/fruit/hand)
- 70/10/20 split + CSV export per requirements
- pHash data leakage detection + handle leakage strategy
- AMP, DataParallel, WeightedRandomSampler
- Metrics: Loss, Accuracy, Macro Precision/Recall/F1
- Saves best_model.pt, training_history.pt, confusion_matrix.csv, classification report
- Visualizations for paper: original / mask / focused image / labels
"""

import os
import json
import time
import random
import argparse
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
	import imagehash
	PHASH_AVAILABLE = True
except ImportError:
	PHASH_AVAILABLE = False
	print("[WARNING] imagehash not installed. pHash duplicate detection disabled.")
	print("          Install with: pip install imagehash")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	classification_report,
	confusion_matrix,
	precision_recall_fscore_support,
)
from tqdm import tqdm

from torch.amp import autocast, GradScaler


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = True


def is_image_file(p: Path) -> bool:
	return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}


def list_images_direct(root: Path) -> List[Path]:
	return [p for p in root.iterdir() if is_image_file(p)]


def denormalize_tensor(img_t: torch.Tensor, mean, std) -> np.ndarray:
	"""Convert normalized tensor (C,H,W) -> uint8 numpy (H,W,C)."""
	img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
	img = img * np.array(std) + np.array(mean)
	img = np.clip(img, 0, 1)
	return (img * 255).astype(np.uint8)


def ensure_serializable(obj):
	if isinstance(obj, (np.integer, )):
		return int(obj)
	if isinstance(obj, (np.floating, )):
		return float(obj)
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	if isinstance(obj, dict):
		return {str(k): ensure_serializable(v) for k, v in obj.items()}
	if isinstance(obj, list):
		return [ensure_serializable(x) for x in obj]
	return obj


# -----------------------------
# Dataset collection (selected / unselected)
# -----------------------------

def collect_all_images_from_dataset(data_root: Path, parts_keep=("hand", "leaf", "flower", "fruit")) -> Dict[str, List[Path]]:
	"""Collect ALL available images for each class/species."""
	species_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
	all_class_imgs = {}

	for species_dir in species_dirs:
		species = species_dir.name
		all_imgs = []

		for part in parts_keep:
			part_dir = species_dir / part
			if part_dir.exists() and part_dir.is_dir():
				all_imgs.extend([p for p in part_dir.rglob("*") if is_image_file(p)])

		all_imgs.extend(list_images_direct(species_dir))

		seen = set()
		uniq = []
		for p in all_imgs:
			if p not in seen:
				seen.add(p)
				uniq.append(p)
		all_class_imgs[species] = uniq

	return all_class_imgs


def build_selection_for_species(species_dir: Path, parts_keep=("hand", "leaf", "flower", "fruit"), per_class_cap=100, seed=42):
	"""Select up to `per_class_cap` images for one species, prioritizing subfolders in parts_keep."""
	rng = random.Random(seed)
	sub_images: List[Path] = []

	for part in parts_keep:
		part_dir = species_dir / part
		if part_dir.exists() and part_dir.is_dir():
			sub_images.extend([p for p in part_dir.rglob("*") if is_image_file(p)])

	sub_images = list(dict.fromkeys(sub_images))
	rng.shuffle(sub_images)

	if len(sub_images) >= per_class_cap:
		return sub_images[:per_class_cap]

	available_images = list_images_direct(species_dir)
	rng.shuffle(available_images)
	need = per_class_cap - len(sub_images)
	return sub_images + available_images[:need]


def scan_dataset(data_root: Path, parts_keep=("hand", "leaf", "flower", "fruit"), per_class_cap=100, seed=42) -> pd.DataFrame:
	"""Build selected dataset DataFrame with columns: path/species/label_id/source/part."""
	species_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
	species_names = [p.name for p in species_dirs]
	species_to_id = {name: i for i, name in enumerate(species_names)}

	rows = []
	for species_dir in species_dirs:
		species = species_dir.name
		selected_paths = build_selection_for_species(species_dir, parts_keep, per_class_cap, seed)
		for img in selected_paths:
			part_val = None
			for anc in img.parents:
				if anc == species_dir:
					break
				if anc.name in parts_keep:
					part_val = anc.name
					break
			source = "sub" if part_val is not None else "available"
			rows.append({
				"path": str(img.resolve()),
				"species": species,
				"label_id": species_to_id[species],
				"source": source,
				"part": part_val if part_val is not None else "available",
			})
	return pd.DataFrame(rows)


def create_dataset_unselected_csv(all_class_imgs: Dict[str, List[Path]],
								  df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
								  out_dir: Path, label_map: Dict[str, int]):
	"""Create dataset_unselected.csv for images not present in train/val/test."""
	selected_images = set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())

	total_available = sum(len(imgs) for imgs in all_class_imgs.values())
	total_selected = len(selected_images)
	total_unselected = total_available - total_selected

	unselected_path = out_dir / "dataset_unselected.csv"
	with open(unselected_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["path", "species", "label_id"])
		for species, all_imgs in all_class_imgs.items():
			label_id = label_map.get(species, -1)
			for img_path in all_imgs:
				if str(img_path) not in selected_images:
					writer.writerow([str(img_path), species, label_id])

	print(f"\n[INFO] Created {unselected_path}")
	print(f"  - Total available: {total_available} images")
	print(f"  - Selected: {total_selected} images")
	print(f"  - Unselected: {total_unselected} images")


# -----------------------------
# pHash leakage detection / fixing
# -----------------------------

def compute_phash(img_path, hash_size=8):
	if not PHASH_AVAILABLE:
		return None
	try:
		img = Image.open(img_path).convert("RGB")
		return str(imagehash.phash(img, hash_size=hash_size))
	except Exception as e:
		print(f"[WARNING] Could not compute pHash for {img_path}: {e}")
		return None


def compute_phash_for_paths(paths: List[str], hash_size=8) -> Dict[str, str]:
	if not PHASH_AVAILABLE:
		return {}
	hash_map = {}
	for path in tqdm(paths, desc="Computing pHash"):
		h = compute_phash(path, hash_size=hash_size)
		if h is not None:
			hash_map[str(path)] = h
	return hash_map


def hamming_distance_int(int1, int2):
	return (int1 ^ int2).bit_count()


def check_image_leakage_with_train(candidate_path: str, train_hashes: Dict[str, str], threshold=5) -> bool:
	if not PHASH_AVAILABLE:
		return False
	candidate_hash = compute_phash(candidate_path)
	if candidate_hash is None:
		return False
	candidate_int = int(candidate_hash, 16)
	for _, train_hash in train_hashes.items():
		train_int = int(train_hash, 16)
		if hamming_distance_int(candidate_int, train_int) <= threshold:
			return True
	return False


def check_data_leakage_phash(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
							 out_dir: Path, hash_size=8, threshold=5) -> Dict:
	"""Check exact + near cross-split duplicates via pHash."""
	if not PHASH_AVAILABLE:
		print("[WARNING] imagehash not available. Skipping pHash leakage check.")
		return {"status": "skipped", "reason": "imagehash not installed", "leakage_found": False}

	def collect_paths(df, split_name):
		return [(row["path"], split_name, row["species"]) for _, row in df.iterrows()]

	train_paths = collect_paths(df_train, "train")
	val_paths = collect_paths(df_val, "val")
	test_paths = collect_paths(df_test, "test")
	all_items = train_paths + val_paths + test_paths

	print("\n" + "=" * 80)
	print("[DATA LEAKAGE CHECK] pHash cross-split check")
	print("=" * 80)
	print(f"[INFO] Total images: {len(all_items)}")

	hash_map = {}          # path -> (hash_hex, split, class)
	hash_to_paths = {}     # hash -> [(path, split, class)]

	for path, split, cls in tqdm(all_items, desc="Computing pHash"):
		h = compute_phash(path, hash_size=hash_size)
		if h is not None:
			hash_map[path] = (h, split, cls)
			hash_to_paths.setdefault(h, []).append((path, split, cls))

	exact_duplicates = []
	exact_cross_split = []
	for h, items in hash_to_paths.items():
		if len(items) > 1:
			splits = set(x[1] for x in items)
			exact_duplicates.append({"hash": h, "count": len(items), "items": items})
			if len(splits) > 1:
				exact_cross_split.append({"hash": h, "items": items, "splits": list(splits)})

	# near-duplicate by local buckets
	hash_bits = hash_size * hash_size
	prefix_bits = min(16, hash_bits)
	shift = hash_bits - prefix_bits
	buckets = {}
	for path, (h, split, cls) in hash_map.items():
		try:
			int_hash = int(h, 16)
			prefix = int_hash >> shift
			buckets.setdefault(prefix, []).append((path, h, int_hash, split, cls))
		except Exception:
			continue

	near_duplicates, near_cross_split = [], []
	checked_pairs = set()
	for bucket_items in tqdm(buckets.values(), desc="Checking near-duplicates"):
		n = len(bucket_items)
		for i in range(n):
			for j in range(i + 1, n):
				p1, h1, int1, s1, c1 = bucket_items[i]
				p2, h2, int2, s2, c2 = bucket_items[j]
				pair_key = tuple(sorted([p1, p2]))
				if pair_key in checked_pairs:
					continue
				checked_pairs.add(pair_key)

				dist = hamming_distance_int(int1, int2)
				if 0 < dist <= threshold:
					rec = {
						"path1": p1, "split1": s1, "class1": c1,
						"path2": p2, "split2": s2, "class2": c2,
						"distance": dist,
					}
					near_duplicates.append(rec)
					if s1 != s2:
						near_cross_split.append(rec)

	leakage_found = len(exact_cross_split) > 0 or len(near_cross_split) > 0

	report_path = out_dir / "data_leakage_check.csv"
	report_rows = []

	for group in exact_cross_split:
		for path, split, cls in group["items"]:
			report_rows.append({
				"type": "exact_duplicate",
				"hash": group["hash"],
				"path": path,
				"split": split,
				"class": cls,
				"distance": 0,
				"is_cross_split": True,
			})

	for rec in near_cross_split:
		report_rows.append({
			"type": "near_duplicate",
			"hash": "",
			"path": rec["path1"],
			"split": rec["split1"],
			"class": rec["class1"],
			"distance": rec["distance"],
			"is_cross_split": True,
			"paired_with": rec["path2"],
			"paired_split": rec["split2"],
		})
		report_rows.append({
			"type": "near_duplicate",
			"hash": "",
			"path": rec["path2"],
			"split": rec["split2"],
			"class": rec["class2"],
			"distance": rec["distance"],
			"is_cross_split": True,
			"paired_with": rec["path1"],
			"paired_split": rec["split1"],
		})

	if report_rows:
		pd.DataFrame(report_rows).to_csv(report_path, index=False)
		print(f"[INFO] Leakage report saved: {report_path}")

	print(f"[SUMMARY] exact_cross_split={len(exact_cross_split)} | near_cross_split={len(near_cross_split)}")
	if leakage_found:
		print("⚠️  DATA LEAKAGE DETECTED")
	else:
		print("✅ No cross-split leakage")

	return {
		"status": "completed",
		"leakage_found": leakage_found,
		"exact_duplicate_groups": len(exact_duplicates),
		"near_duplicate_pairs": len(near_duplicates),
		"exact_cross_split": len(exact_cross_split),
		"near_cross_split": len(near_cross_split),
		"exact_cross_split_details": exact_cross_split,
		"near_cross_split_details": near_cross_split,
		"report_path": str(report_path) if report_rows else None,
		"hash_map": hash_map,
	}


def build_similarity_groups(df: pd.DataFrame, hash_size=8, threshold=5) -> List[List[str]]:
	"""Union-Find grouping by pHash distance <= threshold."""
	paths = df["path"].tolist()
	if not PHASH_AVAILABLE or len(paths) == 0:
		return [[p] for p in paths]

	hashes = {}
	for p in paths:
		h = compute_phash(p, hash_size=hash_size)
		if h is not None:
			hashes[p] = (h, int(h, 16))

	parent = {p: p for p in paths}
	rank = {p: 0 for p in paths}

	def find(x):
		if parent[x] != x:
			parent[x] = find(parent[x])
		return parent[x]

	def union(x, y):
		px, py = find(x), find(y)
		if px == py:
			return
		if rank[px] < rank[py]:
			px, py = py, px
		parent[py] = px
		if rank[px] == rank[py]:
			rank[px] += 1

	for i in range(len(paths)):
		p1 = paths[i]
		if p1 not in hashes:
			continue
		_, int1 = hashes[p1]
		for j in range(i + 1, len(paths)):
			p2 = paths[j]
			if p2 not in hashes:
				continue
			_, int2 = hashes[p2]
			if hamming_distance_int(int1, int2) <= threshold:
				union(p1, p2)

	groups_dict = {}
	for p in paths:
		root = find(p)
		groups_dict.setdefault(root, []).append(p)
	return list(groups_dict.values())


def group_aware_split(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.2,
					 hash_size=8, threshold=5, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Split by similarity groups to reduce leakage."""
	print("\n" + "=" * 80)
	print("[GROUP-AWARE SPLIT] Rebuilding train/val/test")
	print("=" * 80)

	train_rows, val_rows, test_rows = [], [], []

	for species in tqdm(df["species"].unique(), desc="Processing classes"):
		df_species = df[df["species"] == species].copy()
		groups = build_similarity_groups(df_species, hash_size=hash_size, threshold=threshold)

		n_groups = len(groups)
		n_test = max(1, int(round(n_groups * test_ratio)))
		n_val = max(1, int(round(n_groups * val_ratio)))
		n_train = n_groups - n_test - n_val

		if n_train < 1:
			n_train = 1
			n_val = max(0, n_groups - n_train - n_test)
			if n_val < 0:
				n_test = max(0, n_groups - n_train)
				n_val = 0

		rng = random.Random(hash(species) & 0xFFFFFFFF ^ seed)
		rng.shuffle(groups)

		test_groups = groups[:n_test]
		val_groups = groups[n_test:n_test + n_val]
		train_groups = groups[n_test + n_val:]

		test_paths = set(p for g in test_groups for p in g)
		val_paths = set(p for g in val_groups for p in g)

		for _, row in df_species.iterrows():
			if row["path"] in test_paths:
				test_rows.append(row)
			elif row["path"] in val_paths:
				val_rows.append(row)
			else:
				train_rows.append(row)

	return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


def handle_leakage_minor(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
						 all_class_imgs: Dict[str, List[Path]], leakage_result: Dict,
						 threshold=5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
	"""Minor leakage fix: move leaked val/test to train, replace from unselected pool."""
	print("\n" + "=" * 80)
	print("[LEAKAGE FIX] Minor leakage strategy")
	print("=" * 80)

	leaked_from_val, leaked_from_test = {}, {}

	for group in leakage_result.get("exact_cross_split_details", []):
		for path, split, cls in group["items"]:
			if split == "val":
				leaked_from_val.setdefault(cls, [])
				if path not in leaked_from_val[cls]:
					leaked_from_val[cls].append(path)
			elif split == "test":
				leaked_from_test.setdefault(cls, [])
				if path not in leaked_from_test[cls]:
					leaked_from_test[cls].append(path)

	for rec in leakage_result.get("near_cross_split_details", []):
		for side in [("1", leaked_from_val, leaked_from_test), ("2", leaked_from_val, leaked_from_test)]:
			idx = side[0]
			split_key = f"split{idx}"
			cls_key = f"class{idx}"
			path_key = f"path{idx}"
			split = rec[split_key]
			cls = rec[cls_key]
			path = rec[path_key]

			if split == "val":
				leaked_from_val.setdefault(cls, [])
				if path not in leaked_from_val[cls]:
					leaked_from_val[cls].append(path)
			elif split == "test":
				leaked_from_test.setdefault(cls, [])
				if path not in leaked_from_test[cls]:
					leaked_from_test[cls].append(path)

	selected_images = set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())
	unselected_per_class = {
		cls: [str(p) for p in imgs if str(p) not in selected_images]
		for cls, imgs in all_class_imgs.items()
	}

	train_hashes = compute_phash_for_paths(df_train["path"].tolist())

	train_rows = df_train.to_dict("records")
	val_rows = df_val.to_dict("records")
	test_rows = df_test.to_dict("records")

	stats = {"val": 0, "test": 0, "failed": 0}

	def _fix_split(leaked_dict, split_rows, split_name):
		for cls, leaked_paths in leaked_dict.items():
			for leaked_path in leaked_paths:
				# Find and move leaked image from evaluation split to train
				leaked_row = None
				for i, row in enumerate(split_rows):
					if row["path"] == leaked_path:
						leaked_row = split_rows.pop(i)
						break
				if leaked_row is None:
					continue

				train_rows.append(leaked_row)
				h = compute_phash(leaked_path)
				if h:
					train_hashes[leaked_path] = h

				found_repl = False
				for cand in list(unselected_per_class.get(cls, [])):
					if not check_image_leakage_with_train(cand, train_hashes, threshold):
						new_row = leaked_row.copy()
						new_row["path"] = cand
						split_rows.append(new_row)
						unselected_per_class[cls].remove(cand)
						stats[split_name] += 1
						found_repl = True
						break
				if not found_repl:
					stats["failed"] += 1

	_fix_split(leaked_from_val, val_rows, "val")
	_fix_split(leaked_from_test, test_rows, "test")

	print(f"[SUMMARY] replacements val={stats['val']} test={stats['test']} failed={stats['failed']}")
	return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows), stats["failed"] == 0


def handle_data_leakage(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
						df_all: pd.DataFrame, all_class_imgs: Dict[str, List[Path]],
						out_dir: Path, val_ratio=0.1, test_ratio=0.2,
						hash_size=8, threshold=5, max_iterations=3, seed=42):
	"""Main leakage handler with minor/major strategies."""
	print("\n" + "=" * 80)
	print("[DATA LEAKAGE HANDLER] Starting")
	print("=" * 80)

	iteration = 0
	while iteration < max_iterations:
		iteration += 1
		print(f"\n[Iteration {iteration}/{max_iterations}]")
		leak = check_data_leakage_phash(
			df_train, df_val, df_test, out_dir,
			hash_size=hash_size, threshold=threshold
		)

		if not leak.get("leakage_found", False):
			print("✅ No leakage detected.")
			return df_train, df_val, df_test, leak

		total_eval = len(df_val) + len(df_test)
		n_leaked = leak.get("exact_cross_split", 0) + leak.get("near_cross_split", 0)
		leak_pct = 100.0 * n_leaked / max(total_eval, 1)
		print(f"[INFO] leaked={n_leaked}/{total_eval} ({leak_pct:.2f}%)")

		if leak_pct < 5.0:
			df_train, df_val, df_test, _ = handle_leakage_minor(
				df_train, df_val, df_test,
				all_class_imgs=all_class_imgs,
				leakage_result=leak,
				threshold=threshold,
			)
		else:
			print("[DECISION] major leakage -> group-aware split")
			df_train, df_val, df_test = group_aware_split(
				df_all,
				val_ratio=val_ratio,
				test_ratio=test_ratio,
				hash_size=hash_size,
				threshold=threshold,
				seed=seed,
			)

	final_result = check_data_leakage_phash(
		df_train, df_val, df_test, out_dir,
		hash_size=hash_size, threshold=threshold
	)
	if final_result.get("leakage_found", False):
		print("⚠ WARNING: leakage still remains after max iterations.")
	else:
		print("✅ Leakage resolved.")

	return df_train, df_val, df_test, final_result


# -----------------------------
# Torch Dataset & transforms
# -----------------------------

class PlantImageDataset(Dataset):
	def __init__(self, df: pd.DataFrame, transform=None):
		self.df = df.reset_index(drop=True)
		self.transform = transform

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		img = Image.open(row["path"]).convert("RGB")
		if self.transform is not None:
			img = self.transform(img)
		return img, int(row["label_id"]), row["path"]


def get_transforms(img_size=224):
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	train_tfms = transforms.Compose([
		transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(p=0.1),
		transforms.RandomRotation(degrees=15, fill=tuple(int(x * 255) for x in mean)),
		transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])

	eval_tfms = transforms.Compose([
		transforms.Resize(img_size),
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])
	return train_tfms, eval_tfms, mean, std


# -----------------------------
# Hybrid model: SegFormer + MobileNetV3
# -----------------------------

def build_mobilenetv3_large(num_classes: int):
	try:
		import timm
		model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=num_classes)
		return model, "timm"
	except Exception:
		from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
		model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
		in_features = model.classifier[3].in_features
		model.classifier[3] = nn.Linear(in_features, num_classes)
		return model, "torchvision"


def build_segformer(model_name: str):
	try:
		from transformers import SegformerForSemanticSegmentation
	except ImportError as e:
		raise ImportError("transformers is required for SegFormer. Install: pip install transformers") from e
	model = SegformerForSemanticSegmentation.from_pretrained(model_name)
	return model


def infer_plant_class_ids(segformer_model, explicit_keywords=None) -> List[int]:
	"""
	Infer plant-related ADE labels from id2label using keyword matching.
	Fallback: returns empty list, and the model will use fallback mask strategy.
	"""
	if explicit_keywords is None:
		explicit_keywords = [
			"plant", "tree", "flower", "grass", "leaf", "palm", "bush", "forest", "garden", "vegetation"
		]
	id2label = getattr(segformer_model.config, "id2label", {}) or {}
	plant_ids = []
	for k, v in id2label.items():
		try:
			idx = int(k) if isinstance(k, str) else int(k)
		except Exception:
			continue
		name = str(v).lower()
		if any(word in name for word in explicit_keywords):
			plant_ids.append(idx)
	plant_ids = sorted(set(plant_ids))
	return plant_ids


class HybridMobileNetSegFormer(nn.Module):
	"""
	Hybrid module:
	1) SegFormer predicts semantic logits
	2) Creates foreground probability mask (plant-related)
	3) Applies mask to the input image
	4) MobileNetV3 classifies the focused image
	"""
	def __init__(self,
				 num_classes: int,
				 segformer_name: str = "nvidia/segformer-b4-finetuned-ade-512-512",
				 freeze_segformer: bool = True,
				 mask_blend: float = 1.0,
				 background_keep: float = 0.15):
		super().__init__()
		self.segformer = build_segformer(segformer_name)
		self.classifier, self.cls_backend = build_mobilenetv3_large(num_classes)

		self.mask_blend = mask_blend
		self.background_keep = background_keep

		self.plant_class_ids = infer_plant_class_ids(self.segformer)
		if len(self.plant_class_ids) > 0:
			print(f"[INFO] SegFormer plant-related class ids: {self.plant_class_ids[:20]}"
				  f"{'...' if len(self.plant_class_ids) > 20 else ''}")
		else:
			print("[INFO] Could not infer plant IDs from id2label. Using fallback mask strategy.")

		self.freeze_segformer = freeze_segformer
		if self.freeze_segformer:
			for p in self.segformer.parameters():
				p.requires_grad = False
			self.segformer.eval()

	def _build_foreground_mask(self, seg_logits: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
		logits_up = F.interpolate(seg_logits, size=(out_h, out_w), mode="bilinear", align_corners=False)
		probs = torch.softmax(logits_up, dim=1)

		if len(self.plant_class_ids) > 0:
			ids = [i for i in self.plant_class_ids if i < probs.shape[1]]
			if len(ids) > 0:
				fg_prob = probs[:, ids, :, :].sum(dim=1, keepdim=True)
			else:
				fg_prob = probs.max(dim=1, keepdim=True).values
		else:
			fg_prob = probs.max(dim=1, keepdim=True).values

		fg_prob = fg_prob.clamp(0, 1)
		return fg_prob

	def forward(self, x: torch.Tensor, return_aux: bool = False):
		# SegFormer branch
		if self.freeze_segformer:
			with torch.no_grad():
				seg_out = self.segformer(pixel_values=x)
		else:
			seg_out = self.segformer(pixel_values=x)

		fg_mask = self._build_foreground_mask(seg_out.logits, x.shape[-2], x.shape[-1])

		# Focused image = retain some background + enhance foreground region
		effective_mask = self.background_keep + (1.0 - self.background_keep) * fg_mask
		effective_mask = (1.0 - self.mask_blend) + self.mask_blend * effective_mask
		focused = x * effective_mask

		logits = self.classifier(focused)

		if return_aux:
			return logits, fg_mask, focused
		return logits


# -----------------------------
# Train / Eval
# -----------------------------

def compute_macro_metrics(y_true: np.ndarray, y_pred: np.ndarray):
	if len(y_true) == 0:
		return 0.0, 0.0, 0.0
	p, r, f1, _ = precision_recall_fscore_support(
		y_true, y_pred, average="macro", zero_division=0
	)
	return float(p), float(r), float(f1)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, amp=True):
	model.train()
	running_loss, correct, total = 0.0, 0, 0
	all_labels, all_preds = [], []

	pbar = tqdm(loader, desc="Train", leave=False)
	for imgs, labels, _ in pbar:
		imgs = imgs.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)

		use_amp = amp and torch.cuda.is_available()
		if use_amp and scaler is not None:
			with autocast(device_type="cuda"):
				logits = model(imgs)
				loss = criterion(logits, labels)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			logits = model(imgs)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

		running_loss += loss.item() * imgs.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)

		all_labels.append(labels.detach().cpu().numpy())
		all_preds.append(preds.detach().cpu().numpy())

		pbar.set_postfix({
			"loss": f"{running_loss / max(total, 1):.4f}",
			"acc": f"{100.0 * correct / max(total, 1):.2f}%",
		})

	avg_loss = running_loss / max(total, 1)
	acc = correct / max(total, 1)
	y_true = np.concatenate(all_labels) if all_labels else np.array([])
	y_pred = np.concatenate(all_preds) if all_preds else np.array([])
	mp, mr, mf1 = compute_macro_metrics(y_true, y_pred)
	return avg_loss, acc, mp, mr, mf1


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
	model.eval()
	running_loss, correct, total = 0.0, 0, 0
	all_labels, all_preds = [], []

	for imgs, labels, _ in tqdm(loader, desc=desc, leave=False):
		imgs = imgs.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		logits = model(imgs)
		loss = criterion(logits, labels)

		running_loss += loss.item() * imgs.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)

		all_labels.append(labels.cpu().numpy())
		all_preds.append(preds.cpu().numpy())

	avg_loss = running_loss / max(total, 1)
	acc = correct / max(total, 1)
	y_true = np.concatenate(all_labels) if all_labels else np.array([])
	y_pred = np.concatenate(all_preds) if all_preds else np.array([])
	mp, mr, mf1 = compute_macro_metrics(y_true, y_pred)
	return avg_loss, acc, mp, mr, mf1, y_true, y_pred


def get_model_state_dict(model):
	return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def load_model_state_dict(model, state_dict):
	if isinstance(model, nn.DataParallel):
		model.module.load_state_dict(state_dict)
	else:
		model.load_state_dict(state_dict)


# -----------------------------
# Visualization for paper
# -----------------------------

@torch.no_grad()
def save_hybrid_visualizations(model, dataset, species_list, device, out_dir: Path,
							   mean, std, n_samples=24, seed=42):
	"""
	Save qualitative figures:
	1) Original image
	2) Predicted segmentation mask (foreground probability)
	3) Masked/focus image
	4) Text panel with GT vs Pred
	"""
	vis_dir = out_dir / "visualizations_test"
	vis_dir.mkdir(parents=True, exist_ok=True)

	model.eval()
	rng = random.Random(seed)
	total = len(dataset)
	n = min(n_samples, total)
	indices = list(range(total))
	rng.shuffle(indices)
	indices = indices[:n]

	print(f"\n[VIS] Saving {n} visualization samples to: {vis_dir}")

	net = model.module if isinstance(model, nn.DataParallel) else model

	for rank, idx in enumerate(indices, start=1):
		img_t, gt_label, path = dataset[idx]
		inp = img_t.unsqueeze(0).to(device)

		logits, mask, focused = net(inp, return_aux=True)
		pred_label = int(logits.argmax(dim=1).item())

		img_np = denormalize_tensor(img_t, mean, std)
		mask_np = mask[0, 0].detach().cpu().numpy()
		focused_np = denormalize_tensor(focused[0], mean, std)

		fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))

		axes[0].imshow(img_np)
		axes[0].set_title("Original Image")
		axes[0].axis("off")

		axes[1].imshow(mask_np, cmap="viridis")
		axes[1].set_title("Foreground Mask")
		axes[1].axis("off")

		axes[2].imshow(focused_np)
		axes[2].set_title("Focused Image")
		axes[2].axis("off")

		gt_name = species_list[gt_label] if 0 <= gt_label < len(species_list) else str(gt_label)
		pred_name = species_list[pred_label] if 0 <= pred_label < len(species_list) else str(pred_label)
		axes[3].axis("off")
		axes[3].text(
			0.02, 0.85,
			f"Ground Truth:\n{gt_name}\n\nPredicted:\n{pred_name}",
			fontsize=12,
			va="top",
			bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#999999"),
		)

		save_path = vis_dir / f"sample_{rank:03d}.png"
		plt.tight_layout()
		plt.savefig(save_path, dpi=160, bbox_inches="tight")
		plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
	parser = argparse.ArgumentParser(description="Hybrid MobileNetV3-Large + SegFormer-B4 on BigPlants-100")
	parser.add_argument("--data_root", type=str, required=True)
	parser.add_argument("--out_dir", type=str, default="./outputs_mobilenetv3large-segformer")

	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--seed", type=int, default=42)

	parser.add_argument("--img_size", type=int, default=224)
	parser.add_argument("--per_class_cap", type=int, default=100)

	parser.add_argument("--val_ratio", type=float, default=0.10)
	parser.add_argument("--test_ratio", type=float, default=0.20)

	parser.add_argument("--use_weighted_sampler", action="store_true")
	parser.add_argument("--segformer_name", type=str, default="nvidia/segformer-b4-finetuned-ade-512-512")
	parser.add_argument("--unfreeze_segformer", action="store_true", help="Fine-tune SegFormer branch")
	parser.add_argument("--mask_blend", type=float, default=1.0, help="0 -> no mask, 1 -> full mask effect")
	parser.add_argument("--background_keep", type=float, default=0.15, help="Background retention after masking")

	parser.add_argument("--phash_hash_size", type=int, default=8)
	parser.add_argument("--phash_threshold", type=int, default=5)
	parser.add_argument("--max_leakage_iterations", type=int, default=3)

	parser.add_argument("--vis_samples", type=int, default=30, help="Number of test visualization samples")

	args = parser.parse_args()

	assert 0 < args.val_ratio < 0.5
	assert 0 < args.test_ratio < 0.5
	assert args.val_ratio + args.test_ratio < 1.0
	assert args.img_size == 224, "As per technical requirements, input size must be 224"

	set_seed(args.seed)

	start_time = datetime.now()
	wall_start = time.time()
	print("=" * 90)
	print(f"[START] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print("=" * 90)

	data_root = Path(args.data_root)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# 1) Scan + select dataset
	print("\n[1/8] Scanning dataset...")
	df = scan_dataset(data_root=data_root, per_class_cap=args.per_class_cap, seed=args.seed)
	assert len(df) > 0, "No valid images found"
	df.to_csv(out_dir / "dataset_selected.csv", index=False)

	species_list = sorted(df["species"].unique().tolist())
	label_map = {s: int(df[df["species"] == s]["label_id"].iloc[0]) for s in species_list}
	with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
		json.dump(label_map, f, ensure_ascii=False, indent=2)

	print(f"Selected images: {len(df)} | classes: {len(species_list)}")

	# 2) Collect all images for unselected tracking
	print("\n[2/8] Collecting ALL available images...")
	all_class_imgs = collect_all_images_from_dataset(data_root)
	total_available = sum(len(v) for v in all_class_imgs.values())
	print(f"Total available images: {total_available}")

	# 3) Split 70/10/20
	print("\n[3/8] Splitting train/val/test (70/10/20)...")
	df_trainval, df_test = train_test_split(
		df,
		test_size=args.test_ratio,
		random_state=args.seed,
		stratify=df["species"],
	)
	val_ratio_adj = args.val_ratio / (1.0 - args.test_ratio)
	df_train, df_val = train_test_split(
		df_trainval,
		test_size=val_ratio_adj,
		random_state=args.seed,
		stratify=df_trainval["species"],
	)

	for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
		split_df.to_csv(out_dir / f"{name}.csv", index=False)
		print(f"{name}: {len(split_df)}")

	total_n = len(df)
	print(
		f"Ratios -> train: {len(df_train)/total_n:.3f}, val: {len(df_val)/total_n:.3f}, test: {len(df_test)/total_n:.3f}"
	)

	# 4) dataset_unselected.csv + leakage handling
	print("\n[4/8] Creating dataset_unselected.csv + leakage handling...")
	create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

	df_train, df_val, df_test, leakage_result = handle_data_leakage(
		df_train=df_train,
		df_val=df_val,
		df_test=df_test,
		df_all=df,
		all_class_imgs=all_class_imgs,
		out_dir=out_dir,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		hash_size=args.phash_hash_size,
		threshold=args.phash_threshold,
		max_iterations=args.max_leakage_iterations,
		seed=args.seed,
	)

	for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
		split_df.to_csv(out_dir / f"{name}.csv", index=False)
	create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

	# 5) Build dataloaders
	print("\n[5/8] Building dataloaders...")
	num_classes = len(species_list)
	train_tfms, eval_tfms, mean, std = get_transforms(img_size=args.img_size)

	ds_train = PlantImageDataset(df_train, transform=train_tfms)
	ds_val = PlantImageDataset(df_val, transform=eval_tfms)
	ds_test = PlantImageDataset(df_test, transform=eval_tfms)

	if args.use_weighted_sampler:
		class_counts = df_train["label_id"].value_counts().sort_index().values
		class_weights = 1.0 / (class_counts + 1e-6)
		sample_weights = df_train["label_id"].map(lambda x: class_weights[x]).values
		sampler = WeightedRandomSampler(
			weights=torch.as_tensor(sample_weights, dtype=torch.double),
			num_samples=len(sample_weights),
			replacement=True,
		)
		train_loader = DataLoader(
			ds_train,
			batch_size=args.batch_size,
			sampler=sampler,
			num_workers=args.num_workers,
			pin_memory=True,
			persistent_workers=args.num_workers > 0,
		)
	else:
		train_loader = DataLoader(
			ds_train,
			batch_size=args.batch_size,
			shuffle=True,
			num_workers=args.num_workers,
			pin_memory=True,
			persistent_workers=args.num_workers > 0,
		)

	val_loader = DataLoader(
		ds_val,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		persistent_workers=args.num_workers > 0,
	)
	test_loader = DataLoader(
		ds_test,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		persistent_workers=args.num_workers > 0,
	)

	# 6) Build model
	print("\n[6/8] Building hybrid model...")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = HybridMobileNetSegFormer(
		num_classes=num_classes,
		segformer_name=args.segformer_name,
		freeze_segformer=not args.unfreeze_segformer,
		mask_blend=args.mask_blend,
		background_keep=args.background_keep,
	)
	model = model.to(device)

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	scaler = GradScaler(device="cuda", enabled=torch.cuda.is_available())

	print(f"Device: {device} | Multi-GPU: {torch.cuda.device_count()} | AMP: {torch.cuda.is_available()}")

	# 7) Training
	print("\n[7/8] Training...")
	best_val_acc = 0.0
	no_improve = 0
	patience = 10

	best_ckpt_path = out_dir / "best_model.pt"
	history_path = out_dir / "training_history.pt"

	history = {
		"train_loss": [], "train_acc": [], "train_macro_precision": [], "train_macro_recall": [], "train_macro_f1": [],
		"val_loss": [], "val_acc": [], "val_macro_precision": [], "val_macro_recall": [], "val_macro_f1": [],
		"epoch_time_sec": [],
		"leakage_result": ensure_serializable(leakage_result),
		"config": vars(args),
	}

	for epoch in range(1, args.epochs + 1):
		epoch_start = time.time()
		print(f"\nEpoch {epoch}/{args.epochs}")

		tr_loss, tr_acc, tr_p, tr_r, tr_f1 = train_one_epoch(
			model, train_loader, criterion, optimizer, device, scaler=scaler, amp=True
		)
		va_loss, va_acc, va_p, va_r, va_f1, _, _ = evaluate(
			model, val_loader, criterion, device, desc="Val"
		)
		scheduler.step()

		epoch_time = time.time() - epoch_start

		history["train_loss"].append(float(tr_loss))
		history["train_acc"].append(float(tr_acc))
		history["train_macro_precision"].append(float(tr_p))
		history["train_macro_recall"].append(float(tr_r))
		history["train_macro_f1"].append(float(tr_f1))

		history["val_loss"].append(float(va_loss))
		history["val_acc"].append(float(va_acc))
		history["val_macro_precision"].append(float(va_p))
		history["val_macro_recall"].append(float(va_r))
		history["val_macro_f1"].append(float(va_f1))
		history["epoch_time_sec"].append(float(epoch_time))

		print(
			f"Train | loss={tr_loss:.4f}, acc={tr_acc:.4f}, macroP={tr_p:.4f}, macroR={tr_r:.4f}, macroF1={tr_f1:.4f}"
		)
		print(
			f"Val   | loss={va_loss:.4f}, acc={va_acc:.4f}, macroP={va_p:.4f}, macroR={va_r:.4f}, macroF1={va_f1:.4f}"
		)
		print(f"Epoch time: {epoch_time:.2f}s")

		if va_acc > best_val_acc:
			best_val_acc = va_acc
			no_improve = 0
			state = {
				"epoch": epoch,
				"model_state": get_model_state_dict(model),
				"optimizer_state": optimizer.state_dict(),
				"val_acc": float(best_val_acc),
				"label_map": label_map,
				"species_list": species_list,
				"args": vars(args),
			}
			torch.save(state, best_ckpt_path)
			print(f"✅ Saved best checkpoint: {best_ckpt_path} (val_acc={best_val_acc:.4f})")
		else:
			no_improve += 1
			if no_improve >= patience:
				print(f"Early stopping (no improvement for {patience} epochs).")
				break

		torch.save(history, history_path)

	# Load best model
	if best_ckpt_path.exists():
		ckpt = torch.load(best_ckpt_path, map_location="cpu")
		load_model_state_dict(model, ckpt["model_state"])
		print(f"Loaded best checkpoint (val_acc={ckpt.get('val_acc', -1):.4f})")

	# 8) Test + export metrics/files
	print("\n[8/8] Testing + exporting results...")
	test_loss, test_acc, test_p, test_r, test_f1, y_true, y_pred = evaluate(
		model, test_loader, criterion, device, desc="Test"
	)
	print(
		f"Test  | loss={test_loss:.4f}, acc={test_acc:.4f}, macroP={test_p:.4f}, macroR={test_r:.4f}, macroF1={test_f1:.4f}"
	)

	report = classification_report(
		y_true,
		y_pred,
		labels=list(range(num_classes)),
		target_names=species_list,
		zero_division=0,
		output_dict=True,
	)
	pd.DataFrame(report).transpose().to_csv(out_dir / "test_classification_report.csv")

	cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
	cm_df = pd.DataFrame(cm, index=species_list, columns=species_list)
	cm_df.to_csv(out_dir / "confusion_matrix.csv")

	with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
		json.dump({
			"test_loss": float(test_loss),
			"test_acc": float(test_acc),
			"test_macro_precision": float(test_p),
			"test_macro_recall": float(test_r),
			"test_macro_f1": float(test_f1),
			"best_val_acc": float(best_val_acc),
			"splits": {
				"train": int(len(df_train)),
				"val": int(len(df_val)),
				"test": int(len(df_test)),
			}
		}, f, indent=2, ensure_ascii=False)

	# Save final training history
	torch.save(history, history_path)

	# Visualization for paper
	save_hybrid_visualizations(
		model=model,
		dataset=ds_test,
		species_list=species_list,
		device=device,
		out_dir=out_dir,
		mean=mean,
		std=std,
		n_samples=args.vis_samples,
		seed=args.seed,
	)

	end_time = datetime.now()
	total_seconds = int(time.time() - wall_start)
	hours = total_seconds // 3600
	minutes = (total_seconds % 3600) // 60
	seconds = total_seconds % 60

	print("\n" + "=" * 90)
	print(f"[END] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"[DURATION] {hours}h {minutes}m {seconds}s")
	print("Saved files:")
	print(f"  - {out_dir / 'dataset_selected.csv'}")
	print(f"  - {out_dir / 'dataset_unselected.csv'}")
	print(f"  - {out_dir / 'train.csv'} / val.csv / test.csv")
	print(f"  - {out_dir / 'data_leakage_check.csv'} (if leakage found)")
	print(f"  - {out_dir / 'best_model.pt'}")
	print(f"  - {out_dir / 'training_history.pt'}")
	print(f"  - {out_dir / 'test_classification_report.csv'}")
	print(f"  - {out_dir / 'confusion_matrix.csv'}")
	print(f"  - {out_dir / 'metrics_summary.json'}")
	print(f"  - {out_dir / 'visualizations_test'}")
	print("🎉 Done!")
	print("=" * 90)


if __name__ == "__main__":
	main()
