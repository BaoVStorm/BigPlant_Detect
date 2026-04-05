"""
EfficientNetV2-S + SegFormer-B1 for BigPlants-100 (100-class classification)
===========================================================================

Main goals:
- Reuse the strong dataset curation/splitting/leakage strategy from efficientnetv2s_standalone.py
- Add a SegFormer-B1 branch to produce foreground-aware masks
- Train EfficientNetV2-S classifier on masked images
- Export metrics/artifacts for publication-ready experiments

Example:
python efficientnetv2-segformer.py \
  --data_root /home/bigplants/dataset/bigplants-100-resized-224x224 \
  --out_dir ./outputs \
  --epochs 30 --batch_size 32 --lr 3e-4 --num_workers 8 --use_weighted_sampler
"""

import os
import csv
import json
import math
import time
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional pHash support
try:
	import imagehash
	PHASH_AVAILABLE = True
except ImportError:
	PHASH_AVAILABLE = False
	print("[WARNING] imagehash not installed. pHash leakage check disabled.")
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
	accuracy_score,
	precision_recall_fscore_support,
)
from tqdm import tqdm

try:
	from transformers import SegformerForSemanticSegmentation
	TRANSFORMERS_AVAILABLE = True
except ImportError:
	SegformerForSemanticSegmentation = None
	TRANSFORMERS_AVAILABLE = False
	print("[WARNING] transformers not installed. Install with: pip install transformers")

from torch.amp import autocast, GradScaler


# -----------------------------------------------------------------------------
# Repro / Utils
# -----------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = True


def is_image_file(p: Path) -> bool:
	return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}


def list_images_direct(root: Path) -> List[Path]:
	return [p for p in root.iterdir() if is_image_file(p)]


# -----------------------------------------------------------------------------
# Dataset collection / curation
# -----------------------------------------------------------------------------

PartsKeep = ("hand", "leaf", "flower", "fruit")


def collect_all_images_from_dataset(
	data_root: Path,
	parts_scan=("hand", "leaf", "flower", "fruit", "seed", "root"),
) -> Dict[str, List[Path]]:
	species_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
	all_class_imgs = {}

	for species_dir in species_dirs:
		species = species_dir.name
		all_imgs = []

		for part in parts_scan:
			part_dir = species_dir / part
			if part_dir.exists() and part_dir.is_dir():
				all_imgs.extend([p for p in part_dir.rglob("*") if is_image_file(p)])

		all_imgs.extend(list_images_direct(species_dir))

		seen, uniq = set(), []
		for p in all_imgs:
			if p not in seen:
				seen.add(p)
				uniq.append(p)

		all_class_imgs[species] = uniq

	return all_class_imgs


def build_selection_for_species(
	species_dir: Path,
	parts_keep: Tuple[str, ...] = PartsKeep,
	per_class_cap: int = 100,
	seed: int = 42,
) -> List[Path]:
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


def scan_dataset(
	data_root: Path,
	parts_keep: Tuple[str, ...] = PartsKeep,
	per_class_cap: int = 100,
	seed: int = 42,
) -> pd.DataFrame:
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
			source = "sub" if part_val is not None else "available"
			rows.append({
				"path": str(img.resolve()),
				"species": species,
				"label_id": species_to_id[species],
				"source": source,
				"part": part_val if part_val is not None else "available",
			})
	return pd.DataFrame(rows)


def create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map):
	selected_images = set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())
	total_available = sum(len(imgs) for imgs in all_class_imgs.values())

	unselected_path = out_dir / "dataset_unselected.csv"
	with open(unselected_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["path", "species", "label_id"])
		for species, all_imgs in all_class_imgs.items():
			label_id = label_map.get(species, -1)
			for img_path in all_imgs:
				if str(img_path) not in selected_images:
					writer.writerow([str(img_path), species, label_id])

	print(f"[INFO] Created {unselected_path}")
	print(
		f"       Total available: {total_available}, "
		f"Selected: {len(selected_images)}, "
		f"Unselected: {total_available - len(selected_images)}"
	)


# -----------------------------------------------------------------------------
# pHash Leakage Detection / Fix
# -----------------------------------------------------------------------------

def compute_phash(img_path, hash_size=8):
	if not PHASH_AVAILABLE:
		return None
	try:
		img = Image.open(img_path).convert("RGB")
		return str(imagehash.phash(img, hash_size=hash_size))
	except Exception:
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


def check_data_leakage_phash(df_train, df_val, df_test, out_dir, hash_size=8, threshold=5):
	if not PHASH_AVAILABLE:
		print("[WARNING] imagehash unavailable. Skipping leakage check.")
		return {"status": "skipped", "leakage_found": False}

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

	exact_cross_split = []
	for h, items in hash_to_paths.items():
		if len(items) > 1:
			splits = set(x[1] for x in items)
			if len(splits) > 1:
				exact_cross_split.append({"hash": h, "items": items})

	near_cross_split = []
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
			for j in range(i + 1, n):
				p1, int1, s1, c1 = bucket_items[i]
				p2, int2, s2, c2 = bucket_items[j]
				pair_key = tuple(sorted([p1, p2]))
				if pair_key in checked_pairs:
					continue
				checked_pairs.add(pair_key)
				dist = hamming_distance_int(int1, int2)
				if 0 < dist <= threshold and s1 != s2:
					near_cross_split.append({
						"path1": p1, "split1": s1, "class1": c1,
						"path2": p2, "split2": s2, "class2": c2,
						"distance": dist,
					})

	leakage_found = len(exact_cross_split) > 0 or len(near_cross_split) > 0

	report_path = out_dir / "data_leakage_check.csv"
	report_rows = []
	for group in exact_cross_split:
		for item in group["items"]:
			report_rows.append({
				"type": "exact_duplicate",
				"path": item[0],
				"split": item[1],
				"class": item[2],
				"distance": 0,
				"is_cross_split": True,
			})
	for rec in near_cross_split:
		report_rows.append({
			"type": "near_duplicate",
			"path": rec["path1"],
			"split": rec["split1"],
			"class": rec["class1"],
			"distance": rec["distance"],
			"is_cross_split": True,
			"paired_with": rec["path2"],
		})
		report_rows.append({
			"type": "near_duplicate",
			"path": rec["path2"],
			"split": rec["split2"],
			"class": rec["class2"],
			"distance": rec["distance"],
			"is_cross_split": True,
			"paired_with": rec["path1"],
		})

	if report_rows:
		pd.DataFrame(report_rows).to_csv(report_path, index=False)
		print(f"[INFO] Leakage report saved: {report_path}")

	return {
		"status": "completed",
		"leakage_found": leakage_found,
		"exact_cross_split": len(exact_cross_split),
		"near_cross_split": len(near_cross_split),
		"exact_cross_split_details": exact_cross_split,
		"near_cross_split_details": near_cross_split,
	}


def build_similarity_groups(df: pd.DataFrame, hash_size=8, threshold=5):
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
		for j in range(i + 1, len(paths)):
			p2 = paths[j]
			if p2 not in hashes:
				continue
			if hamming_distance_int(hashes[p1], hashes[p2]) <= threshold:
				union(p1, p2)

	groups_dict = {}
	for p in paths:
		root = find(p)
		groups_dict.setdefault(root, []).append(p)
	return list(groups_dict.values())


def group_aware_split(df, val_ratio=0.1, test_ratio=0.2, hash_size=8, threshold=5, seed=42):
	train_rows, val_rows, test_rows = [], [], []
	for species in tqdm(df["species"].unique(), desc="Group-aware split by class"):
		df_species = df[df["species"] == species].copy()
		groups = build_similarity_groups(df_species, hash_size=hash_size, threshold=threshold)
		n_groups = len(groups)

		n_test_groups = max(1, int(round(n_groups * test_ratio)))
		n_val_groups = max(1, int(round(n_groups * val_ratio)))
		n_train_groups = n_groups - n_test_groups - n_val_groups
		if n_train_groups < 1:
			n_train_groups = 1
			n_val_groups = max(0, n_groups - n_train_groups - n_test_groups)

		rng = random.Random((hash(species) & 0xFFFFFFFF) ^ seed)
		rng.shuffle(groups)

		test_paths = set(p for g in groups[:n_test_groups] for p in g)
		val_paths = set(p for g in groups[n_test_groups:n_test_groups + n_val_groups] for p in g)

		for _, row in df_species.iterrows():
			if row["path"] in test_paths:
				test_rows.append(row)
			elif row["path"] in val_paths:
				val_rows.append(row)
			else:
				train_rows.append(row)

	return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


def handle_leakage_minor(df_train, df_val, df_test, all_class_imgs, leakage_result, threshold=5):
	leaked_from_val, leaked_from_test = {}, {}

	for group in leakage_result.get("exact_cross_split_details", []):
		for path, split, cls in group["items"]:
			if split == "val":
				leaked_from_val.setdefault(cls, []).append(path)
			elif split == "test":
				leaked_from_test.setdefault(cls, []).append(path)

	for record in leakage_result.get("near_cross_split_details", []):
		for split_key, cls_key, path_key in [("split1", "class1", "path1"), ("split2", "class2", "path2")]:
			split = record[split_key]
			cls = record[cls_key]
			path = record[path_key]
			if split == "val":
				leaked_from_val.setdefault(cls, []).append(path)
			elif split == "test":
				leaked_from_test.setdefault(cls, []).append(path)

	leaked_from_val = {k: list(dict.fromkeys(v)) for k, v in leaked_from_val.items()}
	leaked_from_test = {k: list(dict.fromkeys(v)) for k, v in leaked_from_test.items()}

	selected_images = set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())
	unselected_per_class = {
		cls: [str(p) for p in imgs if str(p) not in selected_images]
		for cls, imgs in all_class_imgs.items()
	}

	train_hashes = compute_phash_for_paths(df_train["path"].tolist())
	train_rows = df_train.to_dict("records")
	val_rows = df_val.to_dict("records")
	test_rows = df_test.to_dict("records")
	failed = 0

	def process_split(leaked_dict, split_rows):
		nonlocal failed
		replaced = 0
		for cls, leaked_paths in leaked_dict.items():
			for leaked_path in leaked_paths:
				for i, row in enumerate(split_rows):
					if row["path"] == leaked_path:
						leaked_row = split_rows.pop(i)
						train_rows.append(leaked_row)
						h = compute_phash(leaked_path)
						if h is not None:
							train_hashes[leaked_path] = h

						found = False
						for cand in list(unselected_per_class.get(cls, [])):
							if not check_image_leakage_with_train(cand, train_hashes, threshold):
								new_row = leaked_row.copy()
								new_row["path"] = cand
								split_rows.append(new_row)
								unselected_per_class[cls].remove(cand)
								replaced += 1
								found = True
								break
						if not found:
							failed += 1
						break
		return replaced

	rep_val = process_split(leaked_from_val, val_rows)
	rep_test = process_split(leaked_from_test, test_rows)
	print(f"[LEAKAGE FIX] val replaced={rep_val}, test replaced={rep_test}, failed={failed}")

	return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows), failed == 0


def handle_data_leakage(
	df_train, df_val, df_test, df_all, all_class_imgs, out_dir,
	val_ratio=0.1, test_ratio=0.2, hash_size=8, threshold=5,
	max_iterations=3, seed=42,
):
	print("\n" + "=" * 80)
	print("[DATA LEAKAGE HANDLER] Start")
	print("=" * 80)

	for iteration in range(1, max_iterations + 1):
		print(f"\n[Iteration {iteration}/{max_iterations}]")
		leakage_result = check_data_leakage_phash(df_train, df_val, df_test, out_dir, hash_size, threshold)

		if not leakage_result.get("leakage_found", False):
			print("✅ No data leakage detected.")
			return df_train, df_val, df_test, leakage_result

		total_eval = len(df_val) + len(df_test)
		n_leaked = leakage_result.get("exact_cross_split", 0) + leakage_result.get("near_cross_split", 0)
		leakage_pct = (n_leaked / max(1, total_eval)) * 100
		print(f"[INFO] Leakage: {n_leaked} records (~{leakage_pct:.2f}% of val+test)")

		if leakage_pct < 5.0:
			df_train, df_val, df_test, _ = handle_leakage_minor(
				df_train, df_val, df_test, all_class_imgs, leakage_result, threshold=threshold
			)
		else:
			print("[DECISION] Leakage >= 5% -> rebuild by group-aware split")
			df_train, df_val, df_test = group_aware_split(
				df_all,
				val_ratio=val_ratio,
				test_ratio=test_ratio,
				hash_size=hash_size,
				threshold=threshold,
				seed=seed,
			)

	final_result = check_data_leakage_phash(df_train, df_val, df_test, out_dir, hash_size, threshold)
	if final_result.get("leakage_found", False):
		print("⚠ WARNING: Some leakage still remains after max iterations.")
	else:
		print("✅ Leakage resolved.")
	return df_train, df_val, df_test, final_result


# -----------------------------------------------------------------------------
# Dataset + transforms
# -----------------------------------------------------------------------------

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
		return img, int(row["label_id"])


def get_transforms(img_size=224):
	train_tfms = transforms.Compose([
		transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(p=0.1),
		transforms.RandomRotation(degrees=15, fill=0),
		transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
		transforms.ToTensor(),
	])
	eval_tfms = transforms.Compose([
		transforms.Resize(img_size),
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
	])
	return train_tfms, eval_tfms


# -----------------------------------------------------------------------------
# Model: EfficientNetV2-S + SegFormer foreground gating
# -----------------------------------------------------------------------------

def build_efficientnetv2_s(num_classes: int):
	try:
		from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
		model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
		if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
			last = model.classifier[-1]
			if isinstance(last, nn.Linear):
				model.classifier[-1] = nn.Linear(last.in_features, num_classes)
			else:
				for i in reversed(range(len(model.classifier))):
					if isinstance(model.classifier[i], nn.Linear):
						model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
						break
		return model, "torchvision"
	except Exception:
		pass

	try:
		import timm
		for name in ["tf_efficientnetv2_s", "efficientnetv2_s", "tf_efficientnetv2_s_in21k"]:
			try:
				m = timm.create_model(name, pretrained=True, num_classes=num_classes)
				return m, f"timm:{name}"
			except Exception:
				continue
	except Exception:
		pass

	raise RuntimeError("Could not build EfficientNetV2-S. Install torchvision or timm.")


class EffNetV2SegFormerClassifier(nn.Module):
	def __init__(
		self,
		num_classes: int,
		segformer_name: str = "nvidia/segformer-b1-finetuned-ade-512-512",
		seg_input_size: int = 512,
		seg_threshold: float = 0.55,
		seg_temperature: float = 12.0,
		min_keep_bg: float = 0.15,
		freeze_segformer: bool = True,
	):
		super().__init__()

		self.classifier, self.cls_backend = build_efficientnetv2_s(num_classes)
		self.seg_threshold = seg_threshold
		self.seg_temperature = seg_temperature
		self.min_keep_bg = min_keep_bg
		self.seg_input_size = seg_input_size

		if not TRANSFORMERS_AVAILABLE:
			raise ImportError("transformers is required for SegFormer. Please: pip install transformers")

		self.segformer = SegformerForSemanticSegmentation.from_pretrained(segformer_name)
		self.seg_backend = segformer_name

		if freeze_segformer:
			for p in self.segformer.parameters():
				p.requires_grad = False
			self.segformer.eval()

		mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
		std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
		self.register_buffer("mean", mean)
		self.register_buffer("std", std)

	def _build_foreground_mask(self, x_raw: torch.Tensor) -> torch.Tensor:
		h, w = x_raw.shape[-2], x_raw.shape[-1]

		x_seg = F.interpolate(
			x_raw,
			size=(self.seg_input_size, self.seg_input_size),
			mode="bilinear",
			align_corners=False,
		)
		x_seg = (x_seg - self.mean) / self.std

		with torch.no_grad():
			seg_out = self.segformer(pixel_values=x_seg)
			logits = seg_out.logits

		probs = torch.softmax(logits, dim=1)
		conf = probs.max(dim=1, keepdim=True).values

		conf = F.interpolate(conf, size=(h, w), mode="bilinear", align_corners=False)
		conf_min = conf.amin(dim=(2, 3), keepdim=True)
		conf_max = conf.amax(dim=(2, 3), keepdim=True)
		conf_norm = (conf - conf_min) / (conf_max - conf_min + 1e-6)

		soft_mask = torch.sigmoid((conf_norm - self.seg_threshold) * self.seg_temperature)
		return soft_mask

	def forward(self, x_raw: torch.Tensor):
		fg_mask = self._build_foreground_mask(x_raw)
		x_focus = x_raw * (self.min_keep_bg + (1.0 - self.min_keep_bg) * fg_mask)
		x_cls = (x_focus - self.mean) / self.std
		logits = self.classifier(x_cls)
		return logits, fg_mask


# -----------------------------------------------------------------------------
# Train / Eval
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
	model.train()
	if hasattr(model, "segformer"):
		model.segformer.eval()

	running_loss, correct, total = 0.0, 0, 0
	for imgs, labels in tqdm(loader, desc="Train", leave=False):
		imgs = imgs.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)

		if scaler is not None and torch.cuda.is_available():
			with autocast("cuda"):
				logits, _ = model(imgs)
				loss = criterion(logits, labels)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			logits, _ = model(imgs)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

		running_loss += loss.item() * imgs.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)

	return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
	model.eval()
	running_loss, correct, total = 0.0, 0, 0
	all_labels, all_preds = [], []

	for imgs, labels in tqdm(loader, desc=desc, leave=False):
		imgs = imgs.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		logits, _ = model(imgs)
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
	return avg_loss, acc, y_true, y_pred


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path):
	fig = plt.figure(figsize=(16, 14))
	plt.imshow(cm, interpolation="nearest", cmap="Blues")
	plt.title("Confusion Matrix")
	plt.colorbar()

	if len(class_names) <= 30:
		ticks = np.arange(len(class_names))
		plt.xticks(ticks, class_names, rotation=90, fontsize=7)
		plt.yticks(ticks, class_names, fontsize=7)
	else:
		plt.xticks([])
		plt.yticks([])

	plt.ylabel("True label")
	plt.xlabel("Predicted label")
	plt.tight_layout()
	plt.savefig(out_path, dpi=220)
	plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--data_root", type=str, required=True)
	parser.add_argument("--out_dir", type=str, default="./outputs")
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--use_weighted_sampler", action="store_true")

	parser.add_argument("--img_size", type=int, default=224)
	parser.add_argument("--per_class_cap", type=int, default=100)
	parser.add_argument("--parts_keep", type=str, default="hand,leaf,flower,fruit")

	parser.add_argument("--val_ratio", type=float, default=0.10)
	parser.add_argument("--test_ratio", type=float, default=0.20)

	parser.add_argument("--segformer_name", type=str, default="nvidia/segformer-b1-finetuned-ade-512-512")
	parser.add_argument("--seg_input_size", type=int, default=512)
	parser.add_argument("--seg_threshold", type=float, default=0.55)
	parser.add_argument("--seg_temperature", type=float, default=12.0)
	parser.add_argument("--min_keep_bg", type=float, default=0.15)
	parser.add_argument("--unfreeze_segformer", action="store_true")

	parser.add_argument("--leak_hash_size", type=int, default=8)
	parser.add_argument("--leak_threshold", type=int, default=5)
	parser.add_argument("--max_leak_fix_iter", type=int, default=3)

	parser.add_argument("--patience", type=int, default=10)
	args = parser.parse_args()

	assert 0.0 < args.test_ratio < 0.5
	assert 0.0 < args.val_ratio < 0.5
	assert args.val_ratio + args.test_ratio < 1.0

	set_seed(args.seed)

	start_time = datetime.now()
	print("=" * 90)
	print(f"[START] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print("=" * 90)

	data_root = Path(args.data_root)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	parts_keep = tuple([x.strip() for x in args.parts_keep.split(",") if x.strip()])

	print("\n[STEP] Scan dataset & select samples per class...")
	df = scan_dataset(
		data_root=data_root,
		parts_keep=parts_keep,
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
	print(f"[INFO] Selected: {len(df)} images | classes: {len(species_list)}")

	print("\n[STEP] Collect all available images for unselected tracking...")
	all_class_imgs = collect_all_images_from_dataset(data_root)
	total_available = sum(len(v) for v in all_class_imgs.values())
	print(f"[INFO] Total available images: {total_available}")

	print("\n[STEP] Initial stratified split (target 70/10/20)...")
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
		print(f"  {name}: {len(split_df)}")

	create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

	print("\n[STEP] Leakage check + fix...")
	df_train, df_val, df_test, leakage_result = handle_data_leakage(
		df_train=df_train,
		df_val=df_val,
		df_test=df_test,
		df_all=df,
		all_class_imgs=all_class_imgs,
		out_dir=out_dir,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		hash_size=args.leak_hash_size,
		threshold=args.leak_threshold,
		max_iterations=args.max_leak_fix_iter,
		seed=args.seed,
	)

	for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
		split_df.to_csv(out_dir / f"{name}.csv", index=False)
	create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

	print("\n[STEP] Build datasets/loaders...")
	train_tfms, eval_tfms = get_transforms(img_size=args.img_size)
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

	print("\n[STEP] Build model (EffNetV2-S + SegFormer-B1)...")
	num_classes = len(species_list)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = EffNetV2SegFormerClassifier(
		num_classes=num_classes,
		segformer_name=args.segformer_name,
		seg_input_size=args.seg_input_size,
		seg_threshold=args.seg_threshold,
		seg_temperature=args.seg_temperature,
		min_keep_bg=args.min_keep_bg,
		freeze_segformer=not args.unfreeze_segformer,
	)
	model = model.to(device)

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

	print(f"[INFO] Device: {device}")
	print("[INFO] Start training...")

	history = {
		"epoch": [],
		"train_loss": [],
		"train_acc": [],
		"val_loss": [],
		"val_acc": [],
		"epoch_seconds": [],
		"lr": [],
	}

	best_val_acc = 0.0
	no_improve = 0
	best_ckpt = out_dir / "best_model.pt"

	for epoch in range(1, args.epochs + 1):
		ep_start = time.time()
		ep_start_dt = datetime.now()

		print("\n" + "-" * 90)
		print(f"Epoch {epoch}/{args.epochs} | start: {ep_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
		val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc="Val")
		scheduler.step()

		ep_seconds = time.time() - ep_start
		ep_end_dt = datetime.now()

		current_lr = optimizer.param_groups[0]["lr"]
		history["epoch"].append(epoch)
		history["train_loss"].append(float(train_loss))
		history["train_acc"].append(float(train_acc))
		history["val_loss"].append(float(val_loss))
		history["val_acc"].append(float(val_acc))
		history["epoch_seconds"].append(float(ep_seconds))
		history["lr"].append(float(current_lr))

		print(
			f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
			f"Val: loss={val_loss:.4f}, acc={val_acc:.4f} | "
			f"lr={current_lr:.6g}"
		)
		print(
			f"Epoch time: {ep_seconds:.2f}s | "
			f"end: {ep_end_dt.strftime('%Y-%m-%d %H:%M:%S')}"
		)

		torch.save(history, out_dir / "training_history.pt")

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			no_improve = 0
			state = {
				"epoch": epoch,
				"model_state": model.state_dict(),
				"optimizer_state": optimizer.state_dict(),
				"scheduler_state": scheduler.state_dict(),
				"val_acc": best_val_acc,
				"label_map": label_map,
				"species_list": species_list,
				"args": vars(args),
			}
			torch.save(state, best_ckpt)
			print(f"✅ Saved best_model.pt (val_acc={best_val_acc:.4f})")
		else:
			no_improve += 1
			if no_improve >= args.patience:
				print(f"Early stopping: no improvement for {args.patience} epochs.")
				break

	if best_ckpt.exists():
		ckpt = torch.load(best_ckpt, map_location="cpu")
		model.load_state_dict(ckpt["model_state"])
		print(f"[INFO] Loaded best checkpoint (val_acc={ckpt.get('val_acc', -1):.4f})")

	print("\n[STEP] Final test evaluation...")
	test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, desc="Test")

	macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
		y_true, y_pred, average="macro", zero_division=0
	)

	print("\n" + "=" * 90)
	print(f"Test Accuracy       : {test_acc:.4f}")
	print(f"Test Loss           : {test_loss:.4f}")
	print(f"Macro Precision     : {macro_p:.4f}")
	print(f"Macro Recall        : {macro_r:.4f}")
	print(f"Macro F1            : {macro_f1:.4f}")
	print("=" * 90)

	report_dict = classification_report(
		y_true,
		y_pred,
		labels=list(range(num_classes)),
		target_names=species_list,
		zero_division=0,
		output_dict=True,
	)
	pd.DataFrame(report_dict).transpose().to_csv(out_dir / "test_classification_report.csv")

	cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
	pd.DataFrame(cm, index=species_list, columns=species_list).to_csv(out_dir / "confusion_matrix.csv")
	save_confusion_matrix(cm, species_list, out_dir / "confusion_matrix.png")

	metrics_summary = {
		"test_loss": float(test_loss),
		"test_acc": float(test_acc),
		"macro_precision": float(macro_p),
		"macro_recall": float(macro_r),
		"macro_f1": float(macro_f1),
		"best_val_acc": float(best_val_acc),
		"splits": {
			"train": int(len(df_train)),
			"val": int(len(df_val)),
			"test": int(len(df_test)),
		},
		"leakage": {
			"status": leakage_result.get("status", "unknown"),
			"leakage_found": bool(leakage_result.get("leakage_found", False)),
			"exact_cross_split": int(leakage_result.get("exact_cross_split", 0)),
			"near_cross_split": int(leakage_result.get("near_cross_split", 0)),
		},
	}
	with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
		json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

	end_time = datetime.now()
	duration = end_time - start_time
	total_seconds = int(duration.total_seconds())
	hours = total_seconds // 3600
	minutes = (total_seconds % 3600) // 60
	seconds = total_seconds % 60

	print("\n" + "=" * 90)
	print(f"[END] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"[DURATION] {hours}h {minutes}m {seconds}s ({duration.total_seconds()/60:.2f} minutes)")
	print("Saved artifacts:")
	print("  - dataset_selected.csv")
	print("  - dataset_unselected.csv")
	print("  - train.csv, val.csv, test.csv")
	print("  - data_leakage_check.csv (if leakage found)")
	print("  - training_history.pt")
	print("  - best_model.pt")
	print("  - test_classification_report.csv")
	print("  - confusion_matrix.csv")
	print("  - confusion_matrix.png")
	print("  - metrics_summary.json")
	print("=" * 90)
	print("🎉 Done!")


if __name__ == "__main__":
	main()
