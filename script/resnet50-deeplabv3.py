"""
Hybrid ResNet50 + DeepLabV3 training script for BigPlants-100.

Main idea:
1) DeepLabV3-ResNet50 (segmentation branch) produces a foreground mask from
   the input image, separating plant pixels (leaf, flower, fruit...) from
   background noise (soil, sky, etc.).
2) The foreground mask is applied to the input image as an attention-style
   weighting (or hard/residual masking, configurable via --mask_mode).
3) The masked image — which now emphasises only the relevant plant regions —
   is fed into a separate ResNet50 (classification branch) for 100-class
   species classification.

NOTE: The segmentation branch and the classification branch are two
*independent* ResNet50 instances.  The segmentation backbone is the one
embedded inside torchvision's DeepLabV3-ResNet50; the classification
backbone is a standard ResNet50 loaded from timm (preferred) or torchvision.

This script includes:
- Dataset curation (cap 100 images/class with part-folder priority)
- Stratified split 70/10/20 and CSV exports
- pHash-based data leakage detection + fixing strategy
- Mixed precision (AMP), DataParallel, WeightedRandomSampler
- Metrics, confusion matrix, classification reports, checkpoints
- Visualization for paper: original / mask / masked / prediction vs GT
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision import models as tv_models

# Optional pHash dependency
try:
	import imagehash
	PHASH_AVAILABLE = True
except ImportError:
	PHASH_AVAILABLE = False
	print("[WARNING] imagehash not installed. pHash leakage check disabled.")
	print("          Install with: pip install imagehash")


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def is_image_file(path: Path) -> bool:
	return path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def list_images_direct(root: Path) -> List[Path]:
	return [p for p in root.iterdir() if is_image_file(p)]


def seconds_to_hms(total_seconds: float) -> str:
	total_seconds = int(total_seconds)
	h = total_seconds // 3600
	m = (total_seconds % 3600) // 60
	s = total_seconds % 60
	return f"{h}h {m}m {s}s"


# ============================================================
# Dataset scanning and curation
# ============================================================

def collect_all_images_from_dataset(
	data_root: Path,
	parts_keep=("hand", "leaf", "flower", "fruit", "seed")
) -> Dict[str, List[Path]]:
	"""Collect all available images for each species (for unselected tracking)."""
	species_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
	all_class_imgs = {}

	for species_dir in species_dirs:
		species = species_dir.name
		all_imgs: List[Path] = []

		for part in parts_keep:
			part_dir = species_dir / part
			if part_dir.exists() and part_dir.is_dir():
				all_imgs.extend([p for p in part_dir.rglob("*") if is_image_file(p)])

		all_imgs.extend(list_images_direct(species_dir))

		# Deduplicate while preserving order
		seen = set()
		uniq = []
		for p in all_imgs:
			p_str = str(p.resolve())
			if p_str not in seen:
				seen.add(p_str)
				uniq.append(p.resolve())
		all_class_imgs[species] = uniq

	return all_class_imgs


def build_selection_for_species(
	species_dir: Path,
	parts_keep=("hand", "leaf", "flower", "fruit", "seed"),
	per_class_cap=100,
	seed=42,
) -> List[Path]:
	"""
	Select images for one class with priority:
	1) Images from preferred part folders
	2) Fill remaining slots from class root images
	"""
	rng = random.Random(seed)
	preferred_images: List[Path] = []

	for part in parts_keep:
		part_dir = species_dir / part
		if part_dir.exists() and part_dir.is_dir():
			imgs = [p.resolve() for p in part_dir.rglob("*") if is_image_file(p)]
			preferred_images.extend(imgs)

	preferred_images = list(dict.fromkeys(preferred_images))
	rng.shuffle(preferred_images)

	if len(preferred_images) >= per_class_cap:
		return preferred_images[:per_class_cap]

	available_images = [p.resolve() for p in list_images_direct(species_dir)]
	available_images = list(dict.fromkeys(available_images))
	rng.shuffle(available_images)

	need = per_class_cap - len(preferred_images)
	return preferred_images + available_images[:need]


def scan_dataset(
	data_root: Path,
	parts_keep=("hand", "leaf", "flower", "fruit", "seed"),
	per_class_cap=100,
	seed=42,
) -> pd.DataFrame:
	"""Scan BigPlants dataset and build selected dataframe."""
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

			rows.append({
				"path": str(img),
				"species": species,
				"label_id": species_to_id[species],
				"source": "sub" if part_val is not None else "available",
				"part": part_val if part_val is not None else "available",
			})

	df = pd.DataFrame(rows)
	return df


def create_dataset_unselected_csv(
	all_class_imgs: Dict[str, List[Path]],
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	out_dir: Path,
	label_map: Dict[str, int],
):
	"""Save dataset_unselected.csv containing images not selected in splits."""
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
				img_str = str(img_path)
				if img_str not in selected_images:
					writer.writerow([img_str, species, label_id])

	print(f"[INFO] Created {out_path}")
	print(f"  - Total available: {total_available}")
	print(f"  - Selected: {total_selected}")
	print(f"  - Unselected: {total_unselected}")


# ============================================================
# pHash-based leakage detection and fixing
# ============================================================

def compute_phash(img_path, hash_size=8):
	if not PHASH_AVAILABLE:
		return None
	try:
		img = Image.open(img_path).convert("RGB")
		return str(imagehash.phash(img, hash_size=hash_size))
	except Exception as e:
		print(f"[WARNING] Cannot compute pHash for {img_path}: {e}")
		return None


def compute_phash_for_paths(paths: List[str], hash_size=8) -> Dict[str, str]:
	if not PHASH_AVAILABLE:
		return {}
	out = {}
	for p in tqdm(paths, desc="Computing train pHash"):
		h = compute_phash(p, hash_size=hash_size)
		if h is not None:
			out[p] = h
	return out


def hamming_distance_int(int1: int, int2: int) -> int:
	return (int1 ^ int2).bit_count()


def check_image_leakage_with_train(candidate_path: str, train_hashes: Dict[str, str], threshold=5) -> bool:
	if not PHASH_AVAILABLE:
		return False
	candidate_hash = compute_phash(candidate_path)
	if candidate_hash is None:
		return False
	candidate_int = int(candidate_hash, 16)

	for _, train_hash in train_hashes.items():
		dist = hamming_distance_int(candidate_int, int(train_hash, 16))
		if dist <= threshold:
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
		print("[WARNING] imagehash unavailable. Skipping leakage check.")
		return {"status": "skipped", "reason": "imagehash not installed"}

	print("\n" + "=" * 80)
	print("[DATA LEAKAGE CHECK] pHash cross-split duplicate detection")
	print("=" * 80)

	def collect_paths(df: pd.DataFrame, split_name: str):
		return [(row["path"], split_name, row["species"]) for _, row in df.iterrows()]

	train_items = collect_paths(df_train, "train")
	val_items = collect_paths(df_val, "val")
	test_items = collect_paths(df_test, "test")
	all_items = train_items + val_items + test_items

	hash_map = {}         # path -> (hash_hex, split, class)
	hash_to_items = {}    # hash_hex -> list[(path, split, class)]

	for path, split, cls in tqdm(all_items, desc="Computing pHash"):
		h = compute_phash(path, hash_size=hash_size)
		if h is None:
			continue
		hash_map[path] = (h, split, cls)
		hash_to_items.setdefault(h, []).append((path, split, cls))

	exact_duplicates = []
	exact_cross_split = []
	for h, items in hash_to_items.items():
		if len(items) > 1:
			splits = set(it[1] for it in items)
			exact_duplicates.append({"hash": h, "items": items})
			if len(splits) > 1:
				exact_cross_split.append({"hash": h, "items": items, "splits": list(splits)})

	near_duplicates = []
	near_cross_split = []

	# Prefix bucketing for near-duplicate search speedup
	hash_bits = hash_size * hash_size
	prefix_bits = min(16, hash_bits)
	shift = hash_bits - prefix_bits
	buckets = {}
	for path, (h, split, cls) in hash_map.items():
		try:
			int_hash = int(h, 16)
			prefix = int_hash >> shift
			buckets.setdefault(prefix, []).append((path, int_hash, split, cls))
		except Exception:
			continue

	checked_pairs = set()
	for bucket_items in tqdm(buckets.values(), desc="Checking near-duplicates"):
		n = len(bucket_items)
		for i in range(n):
			p1, int1, s1, c1 = bucket_items[i]
			for j in range(i + 1, n):
				p2, int2, s2, c2 = bucket_items[j]
				pair = tuple(sorted((p1, p2)))
				if pair in checked_pairs:
					continue
				checked_pairs.add(pair)
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

	report_path = out_dir / "data_leakage_check.csv"
	report_data = []

	for group in exact_cross_split:
		for item in group["items"]:
			report_data.append({
				"type": "exact_duplicate",
				"hash": group["hash"],
				"path": item[0],
				"split": item[1],
				"class": item[2],
				"distance": 0,
				"is_cross_split": True,
			})

	for rec in near_cross_split:
		report_data.append({
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
		report_data.append({
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

	if report_data:
		pd.DataFrame(report_data).to_csv(report_path, index=False)
		print(f"[INFO] Leakage report saved: {report_path}")

	leakage_found = len(exact_cross_split) > 0 or len(near_cross_split) > 0
	print("\n[SUMMARY]")
	print(f"  Exact duplicate groups: {len(exact_duplicates)}")
	print(f"  Near duplicate pairs: {len(near_duplicates)}")
	print(f"  Cross-split exact: {len(exact_cross_split)}")
	print(f"  Cross-split near: {len(near_cross_split)}")
	print("  Leakage found:" + (" YES" if leakage_found else " NO"))

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
	"""Group visually similar images using pHash and Union-Find."""
	paths = df["path"].tolist()
	if not PHASH_AVAILABLE or len(paths) == 0:
		return [[p] for p in paths]

	hashes = {}
	for p in paths:
		h = compute_phash(p, hash_size=hash_size)
		if h is not None:
			hashes[p] = int(h, 16)

	parent = {p: p for p in paths}
	rank = {p: 0 for p in paths}

	def find(x):
		if parent[x] != x:
			parent[x] = find(parent[x])
		return parent[x]

	def union(x, y):
		rx, ry = find(x), find(y)
		if rx == ry:
			return
		if rank[rx] < rank[ry]:
			rx, ry = ry, rx
		parent[ry] = rx
		if rank[rx] == rank[ry]:
			rank[rx] += 1

	for i in range(len(paths)):
		p1 = paths[i]
		if p1 not in hashes:
			continue
		for j in range(i + 1, len(paths)):
			p2 = paths[j]
			if p2 not in hashes:
				continue
			dist = hamming_distance_int(hashes[p1], hashes[p2])
			if dist <= threshold:
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
	"""Split by similar-image groups to avoid cross-split leakage."""
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

		rng = random.Random((hash(species) & 0xFFFFFFFF) ^ seed)
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


def handle_leakage_minor(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	all_class_imgs: Dict[str, List[Path]],
	leakage_result: Dict,
	threshold=5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
	"""Minor leakage fix: move leaked eval images to train and replace from unselected pool."""
	leaked_val, leaked_test = {}, {}

	for group in leakage_result.get("exact_cross_split_details", []):
		for path, split, cls in group["items"]:
			if split == "val":
				leaked_val.setdefault(cls, []).append(path)
			elif split == "test":
				leaked_test.setdefault(cls, []).append(path)

	for rec in leakage_result.get("near_cross_split_details", []):
		if rec["split1"] == "val":
			leaked_val.setdefault(rec["class1"], []).append(rec["path1"])
		elif rec["split1"] == "test":
			leaked_test.setdefault(rec["class1"], []).append(rec["path1"])

		if rec["split2"] == "val":
			leaked_val.setdefault(rec["class2"], []).append(rec["path2"])
		elif rec["split2"] == "test":
			leaked_test.setdefault(rec["class2"], []).append(rec["path2"])

	# De-duplicate leaked lists
	for cls in list(leaked_val.keys()):
		leaked_val[cls] = list(dict.fromkeys(leaked_val[cls]))
	for cls in list(leaked_test.keys()):
		leaked_test[cls] = list(dict.fromkeys(leaked_test[cls]))

	selected = set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())
	unselected_per_class = {
		cls: [str(p) for p in imgs if str(p) not in selected]
		for cls, imgs in all_class_imgs.items()
	}

	train_hashes = compute_phash_for_paths(df_train["path"].tolist())

	train_rows = df_train.to_dict("records")
	val_rows = df_val.to_dict("records")
	test_rows = df_test.to_dict("records")

	failed = 0

	def move_and_replace(leaked_dict, split_rows, split_name):
		nonlocal failed
		for cls, leaked_paths in leaked_dict.items():
			for leaked_path in leaked_paths:
				moved_row = None
				for i, row in enumerate(split_rows):
					if row["path"] == leaked_path:
						moved_row = split_rows.pop(i)
						break
				if moved_row is None:
					continue

				train_rows.append(moved_row)
				h = compute_phash(leaked_path)
				if h is not None:
					train_hashes[leaked_path] = h

				replacement_found = False
				candidates = unselected_per_class.get(cls, [])
				for candidate in list(candidates):
					if not check_image_leakage_with_train(candidate, train_hashes, threshold=threshold):
						new_row = moved_row.copy()
						new_row["path"] = candidate
						split_rows.append(new_row)
						candidates.remove(candidate)
						selected.add(candidate)
						replacement_found = True
						break

				if not replacement_found:
					print(f"[WARNING] No replacement for leaked {split_name} sample in class={cls}")
					failed += 1

	move_and_replace(leaked_val, val_rows, "val")
	move_and_replace(leaked_test, test_rows, "test")

	return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows), (failed == 0)


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
	"""Main leakage handler: minor fix or group-aware split for major leakage."""
	for it in range(1, max_iterations + 1):
		print(f"\n[Leakage Iteration {it}/{max_iterations}]")
		result = check_data_leakage_phash(
			df_train, df_val, df_test, out_dir,
			hash_size=hash_size, threshold=threshold,
		)
		if not result.get("leakage_found", False):
			print("[INFO] No leakage detected.")
			return df_train, df_val, df_test, result

		total_eval = len(df_val) + len(df_test)
		n_leaked = result.get("exact_cross_split", 0) + result.get("near_cross_split", 0)
		leakage_pct = (n_leaked / max(1, total_eval)) * 100.0
		print(f"[INFO] leakage={leakage_pct:.2f}% ({n_leaked}/{total_eval})")

		if leakage_pct < 5.0:
			df_train, df_val, df_test, success = handle_leakage_minor(
				df_train, df_val, df_test,
				all_class_imgs=all_class_imgs,
				leakage_result=result,
				threshold=threshold,
			)
			if not success:
				print("[WARNING] Minor-fix incomplete. Next iteration may trigger group-aware split.")
		else:
			print("[INFO] Major leakage detected. Rebuilding split via group-aware strategy.")
			df_train, df_val, df_test = group_aware_split(
				df_all,
				val_ratio=val_ratio,
				test_ratio=test_ratio,
				hash_size=hash_size,
				threshold=threshold,
				seed=seed,
			)

	final_result = check_data_leakage_phash(df_train, df_val, df_test, out_dir, hash_size=hash_size, threshold=threshold)
	return df_train, df_val, df_test, final_result


# ============================================================
# Dataset and transforms
# ============================================================

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
		label = int(row["label_id"])
		return img, label


def get_transforms(img_size=224):
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	train_tfms = transforms.Compose([
		transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(p=0.1),
		transforms.RandomRotation(degrees=15, fill=tuple(int(x * 255) for x in mean)),
		transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])
	eval_tfms = transforms.Compose([
		transforms.Resize(img_size),
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])
	return train_tfms, eval_tfms


# ============================================================
# Hybrid model (DeepLabV3 segmentation mask + ResNet50 classifier)
# ============================================================

class HybridResNet50DeepLabV3(nn.Module):
	"""
	Hybrid architecture combining segmentation-guided attention with
	image classification.

	Architecture overview
	---------------------
	+------------------+      +-----------------+      +-------------------+
	| Input Image      | ---> | DeepLabV3       | ---> | Foreground Mask   |
	| (B, 3, H, W)    |      | (Seg. Branch)   |      | (B, 1, H, W)     |
	+------------------+      +-----------------+      +-------------------+
	         |                                                   |
	         |                          multiply / attention      |
	         +--------------------> x * mask -------------------+
	                                    |
	                             +------v--------+
	                             | ResNet50      |
	                             | (Cls. Branch) |
	                             +------+--------+
	                                    |
	                              class logits

	Masking modes
	-------------
	- "attention" (default): soft multiplication  masked = image * fg_mask
	- "hard"              : binary threshold      masked = image * (fg > 0.5)
	- "residual"          : partial suppression    masked = image * (0.5 + 0.5*fg)

	The segmentation branch uses a COCO/VOC-pretrained DeepLabV3-ResNet50
	where class-0 = background.  We derive foreground probability as
	fg = 1 - softmax(logits)[:,0:1,:,:].

	The classification branch is a separate ResNet50 initialised from
	timm (ImageNet-pretrained) or torchvision as fallback.
	"""

	def __init__(
		self,
		num_classes: int,
		seg_pretrained: bool = True,
		seg_freeze: bool = True,
		mask_mode: str = "attention",  # attention | hard | residual
	):
		super().__init__()
		self.seg_freeze = seg_freeze
		self.mask_mode = mask_mode

		# ----------------------------------------------------------
		# Segmentation branch: DeepLabV3 with a ResNet50 backbone
		# ----------------------------------------------------------
		# This model outputs 21-class segmentation logits (PASCAL VOC).
		# We only use its foreground/background separation capability.
		try:
			seg_weights = tv_models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if seg_pretrained else None
			self.seg_model = tv_models.segmentation.deeplabv3_resnet50(weights=seg_weights, aux_loss=True)
		except Exception:
			self.seg_model = tv_models.segmentation.deeplabv3_resnet50(pretrained=seg_pretrained, aux_loss=True)

		if self.seg_freeze:
			for p in self.seg_model.parameters():
				p.requires_grad = False

		# ----------------------------------------------------------
		# Classification branch: ResNet50
		# ----------------------------------------------------------
		# NOTE: This is a *separate* ResNet50 instance — NOT the same
		# backbone as the one inside DeepLabV3.  Its parameters are
		# fully trainable and it receives the mask-weighted image as
		# input.
		self.cls_backend = "torchvision"
		try:
			import timm
			self.cls_model = timm.create_model(
				"resnet50",
				pretrained=True,
				num_classes=num_classes,
			)
			self.cls_backend = "timm"
		except Exception:
			# Fallback to torchvision ResNet50
			weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
			tv_resnet = tv_models.resnet50(weights=weights)
			in_features = tv_resnet.fc.in_features
			tv_resnet.fc = nn.Linear(in_features, num_classes)
			self.cls_model = tv_resnet

	# ------------------------------------------------------------------
	# Foreground mask construction
	# ------------------------------------------------------------------
	def _build_foreground_mask(self, x: torch.Tensor, seg_logits: torch.Tensor) -> torch.Tensor:
		"""
		Build a foreground probability mask from segmentation logits.

		For torchvision pretrained DeepLabV3 (PASCAL VOC-like labels),
		channel-0 corresponds to the *background* class.
		Foreground probability = 1 - P(background).

		Returns a tensor of shape [B, 1, H, W] clamped to [0, 1].
		"""
		probs = torch.softmax(seg_logits, dim=1)
		fg = 1.0 - probs[:, :1, :, :]  # shape: [B, 1, H, W]
		# Ensure spatial dimensions match the input image
		fg = F.interpolate(fg, size=x.shape[-2:], mode="bilinear", align_corners=False)
		return fg.clamp(0.0, 1.0)

	# ------------------------------------------------------------------
	# Mask application strategies
	# ------------------------------------------------------------------
	def _apply_mask(self, x: torch.Tensor, fg_mask: torch.Tensor) -> torch.Tensor:
		"""
		Apply the foreground mask to the input image.

		- "hard"     : binary threshold — completely removes background.
		- "residual" : keeps 50% of original + 50% foreground emphasis,
		               preventing over-suppression in uncertain regions.
		- "attention": soft multiplication (default) — scaled by mask
		               probability, allowing gradient-friendly weighting.
		"""
		if self.mask_mode == "hard":
			hard_mask = (fg_mask > 0.5).float()
			return x * hard_mask
		if self.mask_mode == "residual":
			# Keep a residual path to avoid over-suppressing uncertain regions.
			return x * (0.5 + 0.5 * fg_mask)
		# Default: soft attention map.
		return x * fg_mask

	# ------------------------------------------------------------------
	# Forward pass
	# ------------------------------------------------------------------
	def forward(self, x: torch.Tensor, return_aux: bool = False):
		"""
		Forward pass of the hybrid model.

		Steps:
		  1. Run segmentation branch to get raw logits.
		  2. Convert logits -> foreground probability mask.
		  3. Apply mask to input image (element-wise multiplication).
		  4. Feed masked image into classification branch.

		Args:
			x          : Input tensor [B, 3, H, W].
			return_aux : If True, also return the fg_mask and masked image
			             (useful for visualization / debugging).

		Returns:
			logits             : Classification logits [B, num_classes].
			(fg_mask, x_masked): Returned only when return_aux=True.
		"""
		# Step 1-2: Segmentation → foreground mask
		if self.seg_freeze:
			self.seg_model.eval()
			with torch.no_grad():
				seg_logits = self.seg_model(x)["out"]
		else:
			seg_logits = self.seg_model(x)["out"]

		fg_mask = self._build_foreground_mask(x, seg_logits)

		# Step 3: Apply mask to input image
		x_masked = self._apply_mask(x, fg_mask)

		# Step 4: Classification on mask-weighted image
		logits = self.cls_model(x_masked)

		if return_aux:
			return logits, fg_mask, x_masked
		return logits


# ============================================================
# Train / Eval
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for images, labels in tqdm(loader, desc="Train", leave=False):
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)
		optimizer.zero_grad(set_to_none=True)

		with autocast("cuda", enabled=torch.cuda.is_available()):
			logits = model(images)
			loss = criterion(logits, labels)

		if scaler is not None and torch.cuda.is_available():
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()

		running_loss += loss.item() * images.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)

	return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Eval"):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	all_labels = []
	all_preds = []

	for images, labels in tqdm(loader, desc=desc, leave=False):
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		with autocast("cuda", enabled=torch.cuda.is_available()):
			logits = model(images)
			loss = criterion(logits, labels)

		preds = logits.argmax(dim=1)

		running_loss += loss.item() * images.size(0)
		correct += (preds == labels).sum().item()
		total += labels.size(0)

		all_labels.append(labels.cpu().numpy())
		all_preds.append(preds.cpu().numpy())

	y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
	y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)

	return running_loss / max(total, 1), correct / max(total, 1), y_true, y_pred


# ============================================================
# Visualization for paper
# ============================================================

def denormalize_image(tensor_img: torch.Tensor):
	"""Convert a normalized tensor [3, H, W] back to a uint8 RGB numpy array."""
	mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
	img = tensor_img.detach().cpu().permute(1, 2, 0).numpy()
	img = np.clip(img * std + mean, 0.0, 1.0)
	return (img * 255).astype(np.uint8)


@torch.no_grad()
def save_visualization_samples(
	model,
	df_test: pd.DataFrame,
	eval_transform,
	species_list: List[str],
	device,
	out_dir: Path,
	n_samples: int = 40,
	seed: int = 42,
):
	"""
	Save visual examples demonstrating how DeepLabV3 guides ResNet50.

	For each sampled test image, a 4-panel subplot is produced:
	  1) Original Image          — raw input after eval transform
	  2) Foreground Mask         — DeepLabV3 probability map (viridis)
	  3) Focused Image           — mask-weighted image fed to classifier
	  4) Prediction Info Panel   — ground truth vs predicted label

	These figures serve as evidence in the paper that the segmentation
	branch directs the classifier's attention to relevant plant parts
	(leaf, flower, fruit) while suppressing background clutter.
	"""
	vis_dir = out_dir / "visualizations"
	vis_dir.mkdir(parents=True, exist_ok=True)

	n_samples = min(n_samples, len(df_test))
	rng = random.Random(seed)
	sample_indices = rng.sample(range(len(df_test)), n_samples)

	records = []

	model.eval()
	for k, idx in enumerate(tqdm(sample_indices, desc="Saving visualizations"), start=1):
		row = df_test.iloc[idx]
		img_path = row["path"]
		gt_label = int(row["label_id"])

		pil_img = Image.open(img_path).convert("RGB")
		input_t = eval_transform(pil_img).unsqueeze(0).to(device)

		with autocast("cuda", enabled=torch.cuda.is_available()):
			logits, fg_mask, masked = model(input_t, return_aux=True)

		pred_label = int(logits.argmax(dim=1).item())

		orig_np = denormalize_image(input_t[0])
		mask_np = fg_mask[0, 0].detach().cpu().numpy()
		masked_np = denormalize_image(masked[0])

		fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))

		axes[0].imshow(orig_np)
		axes[0].set_title("Original Image")
		axes[0].axis("off")

		axes[1].imshow(mask_np, cmap="viridis", vmin=0.0, vmax=1.0)
		axes[1].set_title("Foreground Mask")
		axes[1].axis("off")

		axes[2].imshow(masked_np)
		axes[2].set_title("Focused Image")
		axes[2].axis("off")

		axes[3].axis("off")
		gt_name = species_list[gt_label] if 0 <= gt_label < len(species_list) else str(gt_label)
		pred_name = species_list[pred_label] if 0 <= pred_label < len(species_list) else str(pred_label)
		axes[3].text(
			0.02, 0.85,
			f"Ground Truth:\n{gt_name}\n\nPredicted:\n{pred_name}",
			fontsize=12,
			va="top",
			bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#999999"),
		)

		fig.tight_layout()
		save_path = vis_dir / f"sample_{k:03d}.png"
		fig.savefig(save_path, dpi=160, bbox_inches="tight")
		plt.close(fig)

		records.append({
			"sample_id": k,
			"path": img_path,
			"ground_truth": gt_name,
			"predicted": pred_name,
			"correct": pred_label == gt_label,
			"figure_path": str(save_path),
		})

	pd.DataFrame(records).to_csv(vis_dir / "visualization_index.csv", index=False)
	print(f"[INFO] Saved {len(records)} visualization samples to: {vis_dir}")


# ============================================================
# Main
# ============================================================

def main():
	parser = argparse.ArgumentParser(
		description="Hybrid ResNet50 + DeepLabV3 training for BigPlants-100"
	)
	parser.add_argument("--data_root", type=str, required=True, help="Root folder of BigPlants-100")
	parser.add_argument("--out_dir", type=str, default="./outputs_resnet50-deeplabv3")
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--img_size", type=int, default=224,
	                    help="Input resolution: 224, 384 or 512")
	parser.add_argument("--use_weighted_sampler", action="store_true")

	parser.add_argument("--val_ratio", type=float, default=0.10)
	parser.add_argument("--test_ratio", type=float, default=0.20)
	parser.add_argument("--per_class_cap", type=int, default=100)

	parser.add_argument("--hash_size", type=int, default=8)
	parser.add_argument("--hash_threshold", type=int, default=5)
	parser.add_argument("--leakage_max_iterations", type=int, default=3)

	parser.add_argument("--seg_pretrained", action="store_true",
	                    help="Use pretrained weights for DeepLabV3 segmentation branch")
	parser.add_argument("--seg_freeze", action="store_true",
	                    help="Freeze all parameters of the segmentation branch")
	parser.add_argument("--mask_mode", type=str, default="attention",
	                    choices=["attention", "hard", "residual"],
	                    help="How to apply the foreground mask to the image")

	parser.add_argument("--viz_samples", type=int, default=40)
	parser.add_argument("--patience", type=int, default=10)

	args = parser.parse_args()

	assert 0.0 < args.val_ratio < 0.5
	assert 0.0 < args.test_ratio < 0.5
	assert args.val_ratio + args.test_ratio < 1.0

	set_seed(args.seed)

	start_time = datetime.now()
	print("=" * 90)
	print(f"[START] Hybrid ResNet50 + DeepLabV3 | {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"[CONFIG] img_size={args.img_size} | mask_mode={args.mask_mode} | "
	      f"seg_freeze={args.seg_freeze} | seg_pretrained={args.seg_pretrained}")
	print("=" * 90)

	data_root = Path(args.data_root)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# 1) Dataset curation
	print("\n[1/8] Scanning dataset and selecting images per class...")
	df = scan_dataset(
		data_root,
		per_class_cap=args.per_class_cap,
		seed=args.seed,
	)
	if len(df) == 0:
		raise RuntimeError("No valid images found in dataset.")

	df.to_csv(out_dir / "dataset_selected.csv", index=False)

	species_list = sorted(df["species"].unique().tolist())
	label_map = {s: int(df[df["species"] == s]["label_id"].iloc[0]) for s in species_list}
	with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
		json.dump(label_map, f, ensure_ascii=False, indent=2)

	print(f"[INFO] Selected images: {len(df)} | classes: {len(species_list)}")

	# 2) Collect all dataset images for unselected pool
	print("\n[2/8] Collecting all available images...")
	all_class_imgs = collect_all_images_from_dataset(data_root)
	total_available = sum(len(v) for v in all_class_imgs.values())
	print(f"[INFO] Total available images in dataset: {total_available}")

	# 3) Split 70/10/20
	print("\n[3/8] Splitting train/val/test with stratification...")
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
		print(f"  {name:>5s}: {len(split_df)}")

	create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

	# 4) Leakage check/fix
	print("\n[4/8] Running pHash leakage detection and fixing...")
	df_train, df_val, df_test, leakage_result = handle_data_leakage(
		df_train, df_val, df_test,
		df_all=df,
		all_class_imgs=all_class_imgs,
		out_dir=out_dir,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		hash_size=args.hash_size,
		threshold=args.hash_threshold,
		max_iterations=args.leakage_max_iterations,
		seed=args.seed,
	)

	# Save updated splits after leakage handling
	for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
		split_df.to_csv(out_dir / f"{name}.csv", index=False)
	create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

	total_n = len(df_train) + len(df_val) + len(df_test)
	print(f"[INFO] Final split ratio: train={len(df_train)/total_n:.3f}, val={len(df_val)/total_n:.3f}, test={len(df_test)/total_n:.3f}")

	# 5) Build dataloaders
	print("\n[5/8] Building dataloaders...")
	train_tfms, eval_tfms = get_transforms(img_size=args.img_size)

	ds_train = PlantImageDataset(df_train, transform=train_tfms)
	ds_val = PlantImageDataset(df_val, transform=eval_tfms)
	ds_test = PlantImageDataset(df_test, transform=eval_tfms)

	loader_kwargs = {
		"num_workers": args.num_workers,
		"pin_memory": True,
		"persistent_workers": args.num_workers > 0,
	}

	if args.use_weighted_sampler:
		class_counts = df_train["label_id"].value_counts().sort_index().values
		class_weights = 1.0 / (class_counts + 1e-6)
		sample_weights = df_train["label_id"].map(lambda x: class_weights[x]).values
		sampler = WeightedRandomSampler(
			weights=torch.as_tensor(sample_weights, dtype=torch.double),
			num_samples=len(sample_weights),
			replacement=True,
		)
		train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, drop_last=True, **loader_kwargs)
	else:
		train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)

	val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
	test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

	# 6) Build hybrid model
	print("\n[6/8] Building hybrid model (DeepLabV3 segmentation + ResNet50 classifier)...")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = HybridResNet50DeepLabV3(
		num_classes=len(species_list),
		seg_pretrained=args.seg_pretrained,
		seg_freeze=args.seg_freeze,
		mask_mode=args.mask_mode,
	)
	model = model.to(device)

	print(f"[INFO] Classification backend: {model.cls_backend}")

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

	print(f"[INFO] Device={device}, GPUs={torch.cuda.device_count()}, AMP={torch.cuda.is_available()}")

	# 7) Train
	print("\n[7/8] Training...")
	history = {
		"epoch": [],
		"train_loss": [], "train_acc": [],
		"val_loss": [], "val_acc": [],
		"epoch_seconds": [],
		"lr": [],
	}

	best_val_acc = -1.0
	no_improve = 0
	best_ckpt_path = out_dir / "best_model.pt"

	train_start_ts = time.time()
	for epoch in range(1, args.epochs + 1):
		epoch_start = time.time()

		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
		val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc="Val")
		scheduler.step()

		epoch_secs = time.time() - epoch_start
		current_lr = optimizer.param_groups[0]["lr"]

		history["epoch"].append(epoch)
		history["train_loss"].append(float(train_loss))
		history["train_acc"].append(float(train_acc))
		history["val_loss"].append(float(val_loss))
		history["val_acc"].append(float(val_acc))
		history["epoch_seconds"].append(float(epoch_secs))
		history["lr"].append(float(current_lr))

		print(
			f"Epoch {epoch:03d}/{args.epochs} | "
			f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
			f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
			f"time={seconds_to_hms(epoch_secs)}"
		)

		if val_acc > best_val_acc:
			best_val_acc = float(val_acc)
			no_improve = 0
			ckpt = {
				"epoch": epoch,
				"model_state": model.state_dict(),
				"optimizer_state": optimizer.state_dict(),
				"scheduler_state": scheduler.state_dict(),
				"val_acc": best_val_acc,
				"label_map": label_map,
				"species_list": species_list,
				"args": vars(args),
				"leakage_result": leakage_result,
			}
			torch.save(ckpt, best_ckpt_path)
			print(f"  [*] Saved best_model.pt (val_acc={best_val_acc:.4f})")
		else:
			no_improve += 1
			if no_improve >= args.patience:
				print(f"  [!] Early stopping triggered (patience={args.patience}).")
				break

	train_total_secs = time.time() - train_start_ts
	torch.save(history, out_dir / "training_history.pt")

	# Load best checkpoint before testing
	if best_ckpt_path.exists():
		ckpt = torch.load(best_ckpt_path, map_location="cpu")
		model.load_state_dict(ckpt["model_state"])
		print(f"[INFO] Loaded best checkpoint from epoch {ckpt['epoch']} with val_acc={ckpt['val_acc']:.4f}")

	# 8) Test + metrics + visualization
	print("\n[8/8] Testing and exporting reports...")
	test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, desc="Test")

	macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
		y_true, y_pred, average="macro", zero_division=0
	)
	acc = accuracy_score(y_true, y_pred)

	rep = classification_report(
		y_true,
		y_pred,
		labels=list(range(len(species_list))),
		target_names=species_list,
		output_dict=True,
		zero_division=0,
	)
	pd.DataFrame(rep).transpose().to_csv(out_dir / "test_classification_report.csv")

	cm = confusion_matrix(y_true, y_pred, labels=list(range(len(species_list))))
	cm_df = pd.DataFrame(cm, index=species_list, columns=species_list)
	cm_df.to_csv(out_dir / "confusion_matrix.csv")

	save_visualization_samples(
		model,
		df_test=df_test.reset_index(drop=True),
		eval_transform=eval_tfms,
		species_list=species_list,
		device=device,
		out_dir=out_dir,
		n_samples=args.viz_samples,
		seed=args.seed,
	)

	summary = {
		"model": "HybridResNet50DeepLabV3",
		"test_loss": float(test_loss),
		"test_acc": float(test_acc),
		"accuracy": float(acc),
		"macro_precision": float(macro_p),
		"macro_recall": float(macro_r),
		"macro_f1": float(macro_f1),
		"best_val_acc": float(best_val_acc),
		"splits": {
			"train": int(len(df_train)),
			"val": int(len(df_val)),
			"test": int(len(df_test)),
		},
		"timing": {
			"train_seconds": float(train_total_secs),
			"train_hms": seconds_to_hms(train_total_secs),
		},
		"config": {
			"img_size": args.img_size,
			"mask_mode": args.mask_mode,
			"seg_freeze": args.seg_freeze,
			"seg_pretrained": args.seg_pretrained,
			"cls_backend": model.cls_backend if not isinstance(model, nn.DataParallel) else model.module.cls_backend,
		},
	}
	with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)

	end_time = datetime.now()
	total_secs = (end_time - start_time).total_seconds()

	print("\n" + "=" * 90)
	print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")
	print(f"[TEST] macro_precision={macro_p:.4f} macro_recall={macro_r:.4f} macro_f1={macro_f1:.4f}")
	print(f"[END] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"[TOTAL TIME] {seconds_to_hms(total_secs)} ({total_secs / 60.0:.2f} minutes)")
	print(f"[OUTPUT] {out_dir}")
	print("=" * 90)


if __name__ == "__main__":
	main()
