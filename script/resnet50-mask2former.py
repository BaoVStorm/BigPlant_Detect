"""
Hybrid ResNet50 + Mask2Former training script for BigPlants-100.

Main idea:
1) Use Mask2Former (Swin-L backbone) to estimate foreground regions (plant parts).
2) Apply the foreground mask to the raw input image so that background noise is
   suppressed and the classifier can focus on informative pixels (leaf, flower, fruit…).
3) Feed the mask-focused image into ResNet50 for 100-class plant classification.

This script keeps the same dataset curation and pHash leakage-handling logic style
as the standalone ResNet50 classification script, while adding the hybrid
segmentation-guided classification branch and qualitative visualization outputs.
"""

import os
import csv
import json
import time
import argparse
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from torchvision import transforms
import torchvision.transforms.functional as TF

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	accuracy_score,
	precision_recall_fscore_support,
	classification_report,
	confusion_matrix,
)

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Optional import for pHash leakage check
try:
	import imagehash
	PHASH_AVAILABLE = True
except ImportError:
	PHASH_AVAILABLE = False
	print("[WARNING] imagehash not installed. pHash leakage checks will be skipped.")
	print("          Install with: pip install imagehash")


# Optional import for Mask2Former (HuggingFace Transformers)
try:
	from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
	TRANSFORMERS_AVAILABLE = True
except ImportError:
	TRANSFORMERS_AVAILABLE = False


warnings.filterwarnings("ignore")


# ============================================================
# 1. Utilities
# ============================================================

def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def is_image_file(p: Path) -> bool:
	return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images_direct(root: Path) -> List[Path]:
	return [p for p in root.iterdir() if is_image_file(p)]


# ============================================================
# 2. Dataset Curation (selected / unselected)
# ============================================================

def collect_all_images_from_dataset(
	data_root: Path,
	parts_keep=("hand", "leaf", "flower", "fruit"),
) -> Dict[str, List[Path]]:
	"""Collect all available images for each class, no cap."""
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


def build_selection_for_species(
	species_dir: Path,
	parts_keep=("hand", "leaf", "flower", "fruit"),
	per_class_cap=100,
	seed=42,
):
	"""Select up to cap images per class, prioritizing part folders first."""
	rng = random.Random(seed)
	sub_images = []

	for part in parts_keep:
		part_dir = species_dir / part
		if part_dir.exists() and part_dir.is_dir():
			imgs = [p for p in part_dir.rglob("*") if is_image_file(p)]
			sub_images.extend(imgs)

	sub_images = list(dict.fromkeys(sub_images))
	rng.shuffle(sub_images)

	if len(sub_images) >= per_class_cap:
		return sub_images[:per_class_cap]

	available_images = list_images_direct(species_dir)
	rng.shuffle(available_images)
	need = per_class_cap - len(sub_images)
	return sub_images + available_images[:need]


def scan_dataset(
	data_root: Path,
	parts_keep=("hand", "leaf", "flower", "fruit"),
	per_class_cap=100,
	seed=42,
) -> pd.DataFrame:
	"""Create curated dataframe from BigPlants-100 directory tree."""
	species_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
	species_names = [p.name for p in species_dirs]
	species_to_id = {name: i for i, name in enumerate(species_names)}

	rows = []
	for species_dir in species_dirs:
		species = species_dir.name
		selected_paths = build_selection_for_species(
			species_dir,
			parts_keep=parts_keep,
			per_class_cap=per_class_cap,
			seed=seed,
		)
		for img in selected_paths:
			part_val = None
			for anc in img.parents:
				if anc == species_dir:
					break
				if anc.name in parts_keep:
					part_val = anc.name
					break

			rows.append(
				{
					"path": str(img.resolve()),
					"species": species,
					"label_id": species_to_id[species],
					"source": "sub" if part_val is not None else "available",
					"part": part_val if part_val is not None else "available",
				}
			)

	return pd.DataFrame(rows)


def create_dataset_unselected_csv(
	all_class_imgs: Dict[str, List[Path]],
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	out_dir: Path,
	label_map: Dict[str, int],
):
	selected_images = set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())

	total_available = sum(len(imgs) for imgs in all_class_imgs.values())
	total_selected = len(selected_images)
	total_unselected = total_available - total_selected

	out_path = out_dir / "dataset_unselected.csv"
	with open(out_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["path", "species", "label_id"])
		for species, all_imgs in all_class_imgs.items():
			label_id = label_map.get(species, -1)
			for img_path in all_imgs:
				if str(img_path) not in selected_images:
					writer.writerow([str(img_path), species, label_id])

	print(f"[INFO] Created {out_path}")
	print(f"  - Total available: {total_available}")
	print(f"  - Selected: {total_selected}")
	print(f"  - Unselected: {total_unselected}")


# ============================================================
# 3. pHash leakage handling
# ============================================================

def compute_phash(img_path, hash_size=8):
	if not PHASH_AVAILABLE:
		return None
	try:
		img = Image.open(img_path).convert("RGB")
		return str(imagehash.phash(img, hash_size=hash_size))
	except Exception as e:
		print(f"[WARNING] Could not compute pHash for {img_path}: {e}")
		return None


def hamming_distance_int(int1, int2):
	return (int1 ^ int2).bit_count()


def compute_phash_for_paths(paths: List, hash_size=8) -> Dict:
	if not PHASH_AVAILABLE:
		return {}
	hash_map = {}
	for path in tqdm(paths, desc="Computing pHash"):
		h = compute_phash(path, hash_size=hash_size)
		if h is not None:
			hash_map[str(path)] = h
	return hash_map


def check_image_leakage_with_train(candidate_path, train_hashes: Dict, threshold=5) -> bool:
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


def check_data_leakage_phash(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	out_dir: Path,
	hash_size=8,
	threshold=5,
) -> Dict:
	if not PHASH_AVAILABLE:
		print("[WARNING] imagehash not available. Skipping pHash leakage check.")
		return {"status": "skipped", "reason": "imagehash not installed"}

	print("\n" + "=" * 80)
	print("[DATA LEAKAGE CHECK] Using pHash")
	print("=" * 80)

	def collect_paths(df, split_name):
		return [(row["path"], split_name, row["species"]) for _, row in df.iterrows()]

	train_paths = collect_paths(df_train, "train")
	val_paths = collect_paths(df_val, "val")
	test_paths = collect_paths(df_test, "test")
	all_items = train_paths + val_paths + test_paths

	hash_map = {}
	hash_to_paths = {}

	for path, split, cls in tqdm(all_items, desc="Computing pHash"):
		h = compute_phash(path, hash_size=hash_size)
		if h is None:
			continue
		hash_map[path] = (h, split, cls)
		hash_to_paths.setdefault(h, []).append((path, split, cls))

	exact_duplicates = []
	exact_cross_split = []

	for h, items in hash_to_paths.items():
		if len(items) > 1:
			splits = set(i[1] for i in items)
			if len(splits) > 1:
				exact_cross_split.append({"hash": h, "items": items, "splits": list(splits)})
			exact_duplicates.append({"hash": h, "count": len(items), "items": items})

	# Near duplicate search with prefix buckets
	near_duplicates = []
	near_cross_split = []

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

	checked_pairs = set()
	for bucket_items in tqdm(buckets.values(), desc="Checking near-duplicates"):
		n = len(bucket_items)
		for i in range(n):
			for j in range(i + 1, n):
				p1, _, int1, s1, c1 = bucket_items[i]
				p2, _, int2, s2, c2 = bucket_items[j]
				pair_key = tuple(sorted([p1, p2]))
				if pair_key in checked_pairs:
					continue
				checked_pairs.add(pair_key)

				dist = hamming_distance_int(int1, int2)
				if 0 < dist <= threshold:
					rec = {
						"path1": p1,
						"split1": s1,
						"class1": c1,
						"path2": p2,
						"split2": s2,
						"class2": c2,
						"distance": dist,
					}
					near_duplicates.append(rec)
					if s1 != s2:
						near_cross_split.append(rec)

	leakage_found = len(exact_cross_split) > 0 or len(near_cross_split) > 0

	report_data = []
	for group in exact_cross_split:
		for item in group["items"]:
			report_data.append(
				{
					"type": "exact_duplicate",
					"hash": group["hash"],
					"path": item[0],
					"split": item[1],
					"class": item[2],
					"distance": 0,
					"is_cross_split": True,
				}
			)

	for record in near_cross_split:
		report_data.append(
			{
				"type": "near_duplicate",
				"hash": "",
				"path": record["path1"],
				"split": record["split1"],
				"class": record["class1"],
				"distance": record["distance"],
				"is_cross_split": True,
				"paired_with": record["path2"],
				"paired_split": record["split2"],
			}
		)
		report_data.append(
			{
				"type": "near_duplicate",
				"hash": "",
				"path": record["path2"],
				"split": record["split2"],
				"class": record["class2"],
				"distance": record["distance"],
				"is_cross_split": True,
				"paired_with": record["path1"],
				"paired_split": record["split1"],
			}
		)

	report_path = out_dir / "data_leakage_check.csv"
	if report_data:
		pd.DataFrame(report_data).to_csv(report_path, index=False)

	print(f"[INFO] leakage_found={leakage_found} | exact_cross_split={len(exact_cross_split)} | near_cross_split={len(near_cross_split)}")

	return {
		"status": "completed",
		"leakage_found": leakage_found,
		"exact_duplicate_groups": len(exact_duplicates),
		"near_duplicate_pairs": len(near_duplicates),
		"exact_cross_split": len(exact_cross_split),
		"near_cross_split": len(near_cross_split),
		"exact_cross_split_details": exact_cross_split,
		"near_cross_split_details": near_cross_split,
		"report_path": str(report_path) if report_data else None,
	}


def build_similarity_groups(df: pd.DataFrame, hash_size=8, threshold=5) -> List[List[str]]:
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

	groups = {}
	for p in paths:
		root = find(p)
		groups.setdefault(root, []).append(p)

	return list(groups.values())


def group_aware_split(
	df: pd.DataFrame,
	val_ratio=0.1,
	test_ratio=0.2,
	hash_size=8,
	threshold=5,
	seed=42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	train_rows, val_rows, test_rows = [], [], []

	for species in tqdm(df["species"].unique(), desc="Group-aware split per class"):
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

		rng = random.Random((hash(species) & 0xFFFFFFFF) ^ seed)
		rng.shuffle(groups)

		test_groups = groups[:n_test]
		val_groups = groups[n_test:n_test + n_val]
		train_groups = groups[n_test + n_val:]

		test_paths = set(p for g in test_groups for p in g)
		val_paths = set(p for g in val_groups for p in g)
		train_paths = set(p for g in train_groups for p in g)

		for _, row in df_species.iterrows():
			if row["path"] in test_paths:
				test_rows.append(row)
			elif row["path"] in val_paths:
				val_rows.append(row)
			elif row["path"] in train_paths:
				train_rows.append(row)
			else:
				train_rows.append(row)

	return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


def handle_leakage_minor(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	all_class_imgs: Dict[str, List[Path]],
	leakage_result: Dict,
	threshold=5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
	leaked_from_val = {}
	leaked_from_test = {}

	for group in leakage_result.get("exact_cross_split_details", []):
		for path, split, cls in group["items"]:
			if split == "val":
				leaked_from_val.setdefault(cls, []).append(path)
			elif split == "test":
				leaked_from_test.setdefault(cls, []).append(path)

	for record in leakage_result.get("near_cross_split_details", []):
		if record["split1"] == "val":
			leaked_from_val.setdefault(record["class1"], []).append(record["path1"])
		elif record["split1"] == "test":
			leaked_from_test.setdefault(record["class1"], []).append(record["path1"])

		if record["split2"] == "val":
			leaked_from_val.setdefault(record["class2"], []).append(record["path2"])
		elif record["split2"] == "test":
			leaked_from_test.setdefault(record["class2"], []).append(record["path2"])

	for k in list(leaked_from_val.keys()):
		leaked_from_val[k] = list(dict.fromkeys(leaked_from_val[k]))
	for k in list(leaked_from_test.keys()):
		leaked_from_test[k] = list(dict.fromkeys(leaked_from_test[k]))

	selected_images = set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())
	unselected_per_class = {}
	for cls, all_imgs in all_class_imgs.items():
		unselected_per_class[cls] = [str(p) for p in all_imgs if str(p) not in selected_images]

	train_hashes = compute_phash_for_paths(df_train["path"].tolist())

	train_rows = df_train.to_dict("records")
	val_rows = df_val.to_dict("records")
	test_rows = df_test.to_dict("records")

	failed = 0

	# Fix VAL
	for cls, leaked_paths in leaked_from_val.items():
		for leaked_path in leaked_paths:
			leaked_row = None
			for i, row in enumerate(val_rows):
				if row["path"] == leaked_path:
					leaked_row = val_rows.pop(i)
					break

			if leaked_row is None:
				continue

			train_rows.append(leaked_row)
			h = compute_phash(leaked_path)
			if h:
				train_hashes[leaked_path] = h

			replaced = False
			for candidate in list(unselected_per_class.get(cls, [])):
				if not check_image_leakage_with_train(candidate, train_hashes, threshold=threshold):
					new_row = leaked_row.copy()
					new_row["path"] = candidate
					val_rows.append(new_row)
					unselected_per_class[cls].remove(candidate)
					replaced = True
					break

			if not replaced:
				failed += 1

	# Fix TEST
	for cls, leaked_paths in leaked_from_test.items():
		for leaked_path in leaked_paths:
			leaked_row = None
			for i, row in enumerate(test_rows):
				if row["path"] == leaked_path:
					leaked_row = test_rows.pop(i)
					break

			if leaked_row is None:
				continue

			train_rows.append(leaked_row)
			h = compute_phash(leaked_path)
			if h:
				train_hashes[leaked_path] = h

			replaced = False
			for candidate in list(unselected_per_class.get(cls, [])):
				if not check_image_leakage_with_train(candidate, train_hashes, threshold=threshold):
					new_row = leaked_row.copy()
					new_row["path"] = candidate
					test_rows.append(new_row)
					unselected_per_class[cls].remove(candidate)
					replaced = True
					break

			if not replaced:
				failed += 1

	return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows), failed == 0


def handle_data_leakage(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	df_all: pd.DataFrame,
	all_class_imgs: Dict[str, List[Path]],
	out_dir: Path,
	val_ratio=0.1,
	test_ratio=0.2,
	hash_size=8,
	threshold=5,
	max_iterations=3,
	seed=42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
	iteration = 0
	while iteration < max_iterations:
		iteration += 1
		print(f"\n[Leakage Handler] Iteration {iteration}/{max_iterations}")

		leakage_result = check_data_leakage_phash(
			df_train,
			df_val,
			df_test,
			out_dir,
			hash_size=hash_size,
			threshold=threshold,
		)

		if not leakage_result.get("leakage_found", False):
			print("✅ No leakage detected.")
			return df_train, df_val, df_test, leakage_result

		total_eval = len(df_val) + len(df_test)
		n_leaked = leakage_result.get("exact_cross_split", 0) + leakage_result.get("near_cross_split", 0)
		leakage_pct = (n_leaked / max(1, total_eval)) * 100.0

		print(f"[Leakage Handler] leakage_pct={leakage_pct:.2f}%")

		if leakage_pct < 5.0:
			df_train, df_val, df_test, success = handle_leakage_minor(
				df_train,
				df_val,
				df_test,
				all_class_imgs,
				leakage_result,
				threshold=threshold,
			)
			if not success:
				print("[Leakage Handler] Minor-fix replacement partially failed.")
		else:
			print("[Leakage Handler] Rebuilding splits with group-aware strategy.")
			df_train, df_val, df_test = group_aware_split(
				df_all,
				val_ratio=val_ratio,
				test_ratio=test_ratio,
				hash_size=hash_size,
				threshold=threshold,
				seed=seed,
			)

	final_result = check_data_leakage_phash(
		df_train,
		df_val,
		df_test,
		out_dir,
		hash_size=hash_size,
		threshold=threshold,
	)
	return df_train, df_val, df_test, final_result


# ============================================================
# 4. Torch Dataset + Transforms
# ============================================================

class PlantImageDataset(Dataset):
	"""Returns raw image tensor in [0,1], label id, and image path."""

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
	"""
	Keep transform output as raw tensor [0,1].
	Classification normalization is done inside the hybrid model so that
	the segmentation and classification branches can each apply their own
	normalization parameters independently.
	"""
	train_tfms = transforms.Compose(
		[
			transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(p=0.1),
			transforms.RandomRotation(degrees=15),
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
			transforms.ToTensor(),
		]
	)
	eval_tfms = transforms.Compose(
		[
			transforms.Resize(img_size),
			transforms.CenterCrop(img_size),
			transforms.ToTensor(),
		]
	)
	return train_tfms, eval_tfms


# ============================================================
# 5. Hybrid Model: Mask2Former foreground + ResNet50 classifier
# ============================================================

def build_resnet50(num_classes: int):
	"""
	Build a ResNet50 classifier.
	Tries timm first (more flexibility / pretrained variants), then falls
	back to torchvision.
	"""
	try:
		import timm
		# Use a well-known timm ResNet50 variant with strong ImageNet weights
		model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=num_classes)
		return model, "timm"
	except Exception:
		from torchvision.models import resnet50, ResNet50_Weights
		tv_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
		in_features = tv_model.fc.in_features
		tv_model.fc = nn.Linear(in_features, num_classes)
		return tv_model, "torchvision"


class HybridResNet50Mask2Former(nn.Module):
	"""
	Hybrid architecture combining segmentation-guided attention with
	a ResNet50 classification backbone.

	Architecture overview
	---------------------
	  Segmentation branch  : Mask2Former with Swin-L backbone (frozen)
	  Classification branch: ResNet50 (trainable)
	  Fusion strategy      : mask-guided input focusing

	Forward pass
	------------
	1. The raw input image (B, 3, H, W) in [0, 1] is forwarded through the
	   *frozen* Mask2Former to obtain a per-pixel foreground probability map
	   (B, 1, H, W).
	2. The foreground map is smoothed and clamped above a configurable floor
	   value (``mask_floor``) so that background regions are suppressed
	   rather than completely zeroed out – this retains some contextual cues.
	3. The focused image ``raw_image * focus_map`` is then normalised with
	   ImageNet statistics and fed into ResNet50 for classification.

	Parameters
	----------
	num_classes : int
		Number of output classes (100 for BigPlants-100).
	mask2former_name : str
		HuggingFace model identifier for Mask2Former.
	seg_input_size : int
		Spatial resolution to which images are resized before entering the
		segmentation branch.  Mask2Former works best around 384–512 px.
	mask_floor : float
		Minimum value of the focus map.  A value of 0.15 keeps 15 % of
		background intensity, preventing total information loss.
	freeze_segmentation : bool
		If True (default), the Mask2Former parameters are frozen.
	"""

	def __init__(
		self,
		num_classes: int,
		mask2former_name: str = "facebook/mask2former-swin-large-ade-semantic",
		seg_input_size: int = 384,
		mask_floor: float = 0.15,
		freeze_segmentation: bool = True,
	):
		super().__init__()

		if not TRANSFORMERS_AVAILABLE:
			raise ImportError(
				"transformers is required for Mask2Former. Install with: pip install transformers"
			)

		# ----- Segmentation branch (Mask2Former with Swin-L) -----
		self.seg_processor = AutoImageProcessor.from_pretrained(mask2former_name)
		self.seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(mask2former_name)
		self.seg_input_size = seg_input_size
		self.mask_floor = mask_floor

		if freeze_segmentation:
			for p in self.seg_model.parameters():
				p.requires_grad = False

		# ----- Classification branch (ResNet50) -----
		self.classifier, self.cls_backend = build_resnet50(num_classes)

		# ImageNet normalization for ResNet50 branch
		self.register_buffer("cls_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer("cls_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

		# Processor normalization for Mask2Former branch
		seg_mean = torch.tensor(self.seg_processor.image_mean).view(1, 3, 1, 1)
		seg_std = torch.tensor(self.seg_processor.image_std).view(1, 3, 1, 1)
		self.register_buffer("seg_mean", seg_mean)
		self.register_buffer("seg_std", seg_std)

	@torch.no_grad()
	def _predict_foreground_mask(self, images_raw: torch.Tensor) -> torch.Tensor:
		"""
		Produce a soft foreground mask from the segmentation branch.

		Parameters
		----------
		images_raw : Tensor
			Batch of images in [0, 1] range, shape (B, 3, H, W).

		Returns
		-------
		Tensor
			Foreground focus map in [mask_floor, 1], shape (B, 1, H, W).

		Notes
		-----
		* The images are resized to ``seg_input_size`` before being fed to
		  Mask2Former, then the resulting semantic map is resized back to the
		  original (H, W).
		* ADE20K convention: class 0 == "wall / background", so every
		  non-zero class is treated as foreground (plant-related pixels).
		* An average-pooling smoothing step prevents hard mask artefacts.
		"""
		b, _, h, w = images_raw.shape

		# Resize to segmentation input size and normalise with processor stats
		seg_in = TF.resize(images_raw, [self.seg_input_size, self.seg_input_size], antialias=True)
		seg_in = (seg_in - self.seg_mean) / self.seg_std

		outputs = self.seg_model(pixel_values=seg_in)
		target_sizes = [(h, w)] * b

		# Post-process to semantic maps: list[Tensor(H, W)] of class ids
		semantic_maps = self.seg_processor.post_process_semantic_segmentation(
			outputs,
			target_sizes=target_sizes,
		)

		masks = []
		for sem in semantic_maps:
			# Binary foreground: everything that is NOT class-0 (background)
			fg = (sem != 0).float().unsqueeze(0)
			masks.append(fg)

		mask = torch.stack(masks, dim=0).to(images_raw.device)  # (B, 1, H, W)

		# Smooth the mask to soften edges and reduce segmentation noise
		mask = F.avg_pool2d(mask, kernel_size=7, stride=1, padding=3)
		# Apply floor so that background is suppressed but not fully zeroed
		mask = self.mask_floor + (1.0 - self.mask_floor) * mask
		return mask.clamp(0.0, 1.0)

	def forward(self, images_raw: torch.Tensor):
		"""
		Full forward pass of the hybrid model.

		Steps:
		  1) Generate foreground mask via frozen Mask2Former.
		  2) Element-wise multiply raw image with the mask (focus map).
		  3) Normalise the focused image and classify with ResNet50.

		Returns
		-------
		dict with keys:
		  - ``logits`` : classification logits (B, num_classes)
		  - ``mask``   : foreground focus map (B, 1, H, W)
		  - ``focused``: focused image before normalisation (B, 3, H, W)
		"""
		# 1) Foreground mask from segmentation branch
		if self.seg_model.training:
			self.seg_model.eval()
		foreground_mask = self._predict_foreground_mask(images_raw)

		# 2) Focus image: element-wise product of raw image and focus map
		focused = images_raw * foreground_mask

		# 3) Classification branch: normalise with ImageNet stats, then classify
		cls_in = (focused - self.cls_mean) / self.cls_std
		logits = self.classifier(cls_in)

		return {
			"logits": logits,
			"mask": foreground_mask,
			"focused": focused,
		}


# ============================================================
# 6. Training / Evaluation
# ============================================================

def compute_main_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	acc = accuracy_score(y_true, y_pred) if y_true.size > 0 else 0.0
	p, r, f1, _ = precision_recall_fscore_support(
		y_true,
		y_pred,
		average="macro",
		zero_division=0,
	)
	return {
		"accuracy": float(acc),
		"macro_precision": float(p),
		"macro_recall": float(r),
		"macro_f1": float(f1),
	}


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
	model.train()
	running_loss = 0.0
	all_preds, all_labels = [], []

	pbar = tqdm(loader, desc="Train", leave=False)
	for images, labels, _ in pbar:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)

		if scaler is not None and torch.cuda.is_available():
			with autocast("cuda"):
				out = model(images)
				loss = criterion(out["logits"], labels)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			out = model(images)
			loss = criterion(out["logits"], labels)
			loss.backward()
			optimizer.step()

		running_loss += loss.item() * images.size(0)
		preds = out["logits"].argmax(dim=1)
		all_preds.append(preds.detach().cpu().numpy())
		all_labels.append(labels.detach().cpu().numpy())

		pbar.set_postfix(loss=f"{loss.item():.4f}")

	y_true = np.concatenate(all_labels) if all_labels else np.array([])
	y_pred = np.concatenate(all_preds) if all_preds else np.array([])
	metrics = compute_main_metrics(y_true, y_pred)
	metrics["loss"] = running_loss / max(1, len(loader.dataset))
	return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
	model.eval()
	running_loss = 0.0
	all_preds, all_labels = [], []

	pbar = tqdm(loader, desc=desc, leave=False)
	for images, labels, _ in pbar:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		out = model(images)
		loss = criterion(out["logits"], labels)

		running_loss += loss.item() * images.size(0)
		preds = out["logits"].argmax(dim=1)
		all_preds.append(preds.cpu().numpy())
		all_labels.append(labels.cpu().numpy())

	y_true = np.concatenate(all_labels) if all_labels else np.array([])
	y_pred = np.concatenate(all_preds) if all_preds else np.array([])
	metrics = compute_main_metrics(y_true, y_pred)
	metrics["loss"] = running_loss / max(1, len(loader.dataset))
	return metrics, y_true, y_pred


# ============================================================
# 7. Visualization for paper
# ============================================================

@torch.no_grad()
def save_qualitative_examples(
	model,
	dataset: Dataset,
	device,
	class_names: List[str],
	out_dir: Path,
	n_samples: int = 40,
):
	"""
	Save visualization panels for academic paper illustration.

	For each sampled test image, produces a 1×4 subplot showing:
	  1. Original image
	  2. Foreground mask predicted by Mask2Former
	  3. Focused image (original × mask)
	  4. Ground-truth vs. predicted label

	This demonstrates that the segmentation branch helps the classifier
	attend to plant-relevant pixels (leaf, flower, fruit) instead of
	background noise (soil, sky, pots, etc.).
	"""
	out_dir.mkdir(parents=True, exist_ok=True)

	n = min(n_samples, len(dataset))
	indices = random.sample(range(len(dataset)), n)

	for i, idx in enumerate(indices):
		image, gt_label, path = dataset[idx]
		inp = image.unsqueeze(0).to(device)

		out = model(inp)
		probs = F.softmax(out["logits"], dim=1)
		pred_label = int(probs.argmax(dim=1).item())

		mask = out["mask"][0, 0].detach().cpu().numpy()
		focused = out["focused"][0].detach().cpu().permute(1, 2, 0).numpy()
		original = image.cpu().permute(1, 2, 0).numpy()

		gt_name = class_names[gt_label] if gt_label < len(class_names) else str(gt_label)
		pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)

		fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))

		axes[0].imshow(np.clip(original, 0, 1))
		axes[0].set_title("Original Image")
		axes[0].axis("off")

		axes[1].imshow(mask, cmap="viridis")
		axes[1].set_title("Foreground Mask")
		axes[1].axis("off")

		axes[2].imshow(np.clip(focused, 0, 1))
		axes[2].set_title("Focused Image")
		axes[2].axis("off")

		axes[3].axis("off")
		axes[3].text(
			0.02, 0.85,
			f"Ground Truth:\n{gt_name}\n\nPredicted:\n{pred_name}",
			fontsize=12,
			va="top",
			bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#999999"),
		)

		plt.tight_layout()
		fig.savefig(out_dir / f"sample_{i+1:03d}.png", dpi=160, bbox_inches="tight")
		plt.close(fig)


def save_confusion_matrix(y_true, y_pred, class_names: List[str], out_png: Path, out_csv: Path):
	cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
	cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
	cm_df.to_csv(out_csv)

	plt.figure(figsize=(14, 12))
	sns.heatmap(cm_df, cmap="Blues", cbar=True)
	plt.title("Confusion Matrix")
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.tight_layout()
	plt.savefig(out_png, dpi=150)
	plt.close()


# ============================================================
# 8. Main
# ============================================================

def main():
	parser = argparse.ArgumentParser(description="Hybrid ResNet50 + Mask2Former for BigPlants-100")

	# Data
	parser.add_argument("--data_root", type=str, required=True)
	parser.add_argument("--out_dir", type=str, default="./outputs_resnet50-mask2former")
	parser.add_argument("--img_size", type=int, default=224,
						help="Input image size for classification (224, 384, or 512)")
	parser.add_argument("--per_class_cap", type=int, default=100)
	parser.add_argument("--val_ratio", type=float, default=0.10)
	parser.add_argument("--test_ratio", type=float, default=0.20)
	parser.add_argument("--seed", type=int, default=42)

	# Training
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--use_weighted_sampler", action="store_true")
	parser.add_argument("--patience", type=int, default=10)

	# Hybrid specifics
	parser.add_argument("--mask2former_name", type=str, default="facebook/mask2former-swin-large-ade-semantic",
						help="HuggingFace model ID for Mask2Former segmentation branch")
	parser.add_argument("--seg_input_size", type=int, default=384,
						help="Spatial size for segmentation branch input (Mask2Former)")
	parser.add_argument("--mask_floor", type=float, default=0.15,
						help="Minimum mask value (prevents total background suppression)")
	parser.add_argument("--unfreeze_segmentation", action="store_true",
						help="If set, allow gradient flow through the segmentation branch")

	# Visualization
	parser.add_argument("--num_vis_samples", type=int, default=40,
						help="Number of qualitative examples to save from test set")

	args = parser.parse_args()

	assert 0.0 < args.val_ratio < 0.5
	assert 0.0 < args.test_ratio < 0.5
	assert args.val_ratio + args.test_ratio < 1.0

	set_seed(args.seed)

	start_time = datetime.now()
	print("=" * 88)
	print(f"[START] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"Model: Hybrid ResNet50 + Mask2Former (Swin-L)")
	print(f"Image size: {args.img_size}x{args.img_size}")
	print("=" * 88)

	data_root = Path(args.data_root)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# 1) Dataset scanning and curation
	print("\n[1/8] Scanning dataset and selecting images (cap per class)...")
	df = scan_dataset(data_root=data_root, per_class_cap=args.per_class_cap, seed=args.seed)
	assert len(df) > 0, "No valid images found."
	df.to_csv(out_dir / "dataset_selected.csv", index=False)

	species_list = sorted(df["species"].unique().tolist())
	label_map = {s: int(df[df["species"] == s]["label_id"].iloc[0]) for s in species_list}
	with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
		json.dump(label_map, f, ensure_ascii=False, indent=2)

	print(f"Selected images: {len(df)} | classes: {len(species_list)}")

	# 2) Collect all available images and split 70/10/20
	print("\n[2/8] Collecting all available images...")
	all_class_imgs = collect_all_images_from_dataset(data_root)

	print("\n[3/8] Splitting train/val/test (target 70/10/20)...")
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

	for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
		d.to_csv(out_dir / f"{name}.csv", index=False)
		print(f"  {name}: {len(d)}")

	create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

	# 3) Leakage check/fix
	print("\n[4/8] Checking and handling data leakage with pHash...")
	df_train, df_val, df_test, leakage_result = handle_data_leakage(
		df_train,
		df_val,
		df_test,
		df_all=df,
		all_class_imgs=all_class_imgs,
		out_dir=out_dir,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		hash_size=8,
		threshold=5,
		max_iterations=3,
		seed=args.seed,
	)

	# Save updated split files after leakage fix
	for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
		d.to_csv(out_dir / f"{name}.csv", index=False)

	create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

	with open(out_dir / "leakage_summary.json", "w", encoding="utf-8") as f:
		json.dump(leakage_result, f, ensure_ascii=False, indent=2, default=str)

	# 4) Datasets & loaders
	print("\n[5/8] Building datasets and loaders...")
	num_classes = len(species_list)
	train_tfms, eval_tfms = get_transforms(args.img_size)

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
		)
	else:
		train_loader = DataLoader(
			ds_train,
			batch_size=args.batch_size,
			shuffle=True,
			num_workers=args.num_workers,
			pin_memory=True,
		)

	val_loader = DataLoader(
		ds_val,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
	)
	test_loader = DataLoader(
		ds_test,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
	)

	# 5) Build model
	print("\n[6/8] Building hybrid model...")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = HybridResNet50Mask2Former(
		num_classes=num_classes,
		mask2former_name=args.mask2former_name,
		seg_input_size=args.seg_input_size,
		mask_floor=args.mask_floor,
		freeze_segmentation=not args.unfreeze_segmentation,
	)
	print(f"Classifier backend: {model.cls_backend}")

	model = model.to(device)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

	# 6) Train loop
	print("\n[7/8] Training...")
	best_val_acc = 0.0
	no_improve = 0
	best_ckpt_path = out_dir / "best_model.pt"

	history = {
		"train_loss": [],
		"train_acc": [],
		"train_macro_precision": [],
		"train_macro_recall": [],
		"train_macro_f1": [],
		"val_loss": [],
		"val_acc": [],
		"val_macro_precision": [],
		"val_macro_recall": [],
		"val_macro_f1": [],
		"epoch_seconds": [],
	}

	for epoch in range(1, args.epochs + 1):
		ep_start = time.time()

		train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
		val_m, _, _ = evaluate(model, val_loader, criterion, device, desc="Val")
		scheduler.step()

		ep_sec = time.time() - ep_start

		history["train_loss"].append(train_m["loss"])
		history["train_acc"].append(train_m["accuracy"])
		history["train_macro_precision"].append(train_m["macro_precision"])
		history["train_macro_recall"].append(train_m["macro_recall"])
		history["train_macro_f1"].append(train_m["macro_f1"])
		history["val_loss"].append(val_m["loss"])
		history["val_acc"].append(val_m["accuracy"])
		history["val_macro_precision"].append(val_m["macro_precision"])
		history["val_macro_recall"].append(val_m["macro_recall"])
		history["val_macro_f1"].append(val_m["macro_f1"])
		history["epoch_seconds"].append(ep_sec)

		print(
			f"Epoch {epoch:03d}/{args.epochs} | "
			f"Train loss={train_m['loss']:.4f}, acc={train_m['accuracy']:.4f}, f1={train_m['macro_f1']:.4f} | "
			f"Val loss={val_m['loss']:.4f}, acc={val_m['accuracy']:.4f}, f1={val_m['macro_f1']:.4f} | "
			f"epoch_time={ep_sec:.1f}s"
		)

		if val_m["accuracy"] > best_val_acc:
			best_val_acc = val_m["accuracy"]
			no_improve = 0
			state = {
				"epoch": epoch,
				"model_state": model.state_dict(),
				"optimizer_state": optimizer.state_dict(),
				"best_val_acc": best_val_acc,
				"label_map": label_map,
				"species_list": species_list,
				"args": vars(args),
			}
			torch.save(state, best_ckpt_path)
			print(f"✅ Saved best checkpoint -> {best_ckpt_path}")
		else:
			no_improve += 1
			if no_improve >= args.patience:
				print(f"Early stopping (no improvement for {args.patience} epochs).")
				break

	# Save history
	torch.save(history, out_dir / "training_history.pt")

	# 7) Test + report + confusion matrix + visualization
	print("\n[8/8] Evaluating test set and exporting reports...")
	if best_ckpt_path.exists():
		ckpt = torch.load(best_ckpt_path, map_location="cpu")
		model.load_state_dict(ckpt["model_state"])
		print(f"Loaded best checkpoint from epoch {ckpt.get('epoch', 'N/A')}")

	test_m, y_true, y_pred = evaluate(model, test_loader, criterion, device, desc="Test")

	print(
		f"Test | loss={test_m['loss']:.4f}, acc={test_m['accuracy']:.4f}, "
		f"precision={test_m['macro_precision']:.4f}, recall={test_m['macro_recall']:.4f}, f1={test_m['macro_f1']:.4f}"
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

	save_confusion_matrix(
		y_true,
		y_pred,
		class_names=species_list,
		out_png=out_dir / "confusion_matrix.png",
		out_csv=out_dir / "confusion_matrix.csv",
	)

	vis_dir = out_dir / "qualitative_examples"
	model.eval()
	save_qualitative_examples(
		model=model,
		dataset=ds_test,
		device=device,
		class_names=species_list,
		out_dir=vis_dir,
		n_samples=args.num_vis_samples,
	)

	summary = {
		"model": "Hybrid ResNet50 + Mask2Former (Swin-L)",
		"img_size": args.img_size,
		"seg_input_size": args.seg_input_size,
		"mask_floor": args.mask_floor,
		"test_loss": float(test_m["loss"]),
		"test_acc": float(test_m["accuracy"]),
		"test_macro_precision": float(test_m["macro_precision"]),
		"test_macro_recall": float(test_m["macro_recall"]),
		"test_macro_f1": float(test_m["macro_f1"]),
		"best_val_acc": float(best_val_acc),
		"splits": {
			"train": int(len(df_train)),
			"val": int(len(df_val)),
			"test": int(len(df_test)),
		},
		"num_classes": int(num_classes),
		"num_vis_samples": int(min(args.num_vis_samples, len(ds_test))),
	}
	with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)

	end_time = datetime.now()
	duration = end_time - start_time
	total_seconds = int(duration.total_seconds())
	hours = total_seconds // 3600
	minutes = (total_seconds % 3600) // 60
	seconds = total_seconds % 60

	print("\n" + "=" * 88)
	print(f"[END] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"[DURATION] {hours}h {minutes}m {seconds}s")
	print("Artifacts:")
	print("  - best_model.pt")
	print("  - training_history.pt")
	print("  - dataset_selected.csv")
	print("  - dataset_unselected.csv")
	print("  - train.csv / val.csv / test.csv")
	print("  - data_leakage_check.csv (if leakage found)")
	print("  - test_classification_report.csv")
	print("  - confusion_matrix.csv / confusion_matrix.png")
	print("  - qualitative_examples/*.png")
	print("=" * 88)
	print("Done.")


if __name__ == "__main__":
	main()
