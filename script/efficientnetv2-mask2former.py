"""
EfficientNetV2-S + Mask2Former (foreground-guided classification)
=================================================================

BigPlants-100 pipeline for 100-class plant classification.

What this script does:
1) Curate dataset with per-class cap and part-priority strategy.
2) Split train/val/test.
3) Detect and fix cross-split leakage by pHash.
4) Build Mask2Former foreground-guided images (optional, cached).
5) Train EfficientNetV2-S classifier.
6) Export: train.csv, val.csv, test.csv, dataset_selected.csv,
   dataset_unselected.csv, classification report, confusion matrix,
   training_history.pt, best_model.pt, metrics_summary.json.
"""

import os
import csv
import json
import time
import argparse
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

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
	print("[WARNING] imagehash not installed. pHash leakage detection disabled.")

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	classification_report,
	confusion_matrix,
	ConfusionMatrixDisplay,
)
from tqdm import tqdm


# ============================================================
# 1) Utilities
# ============================================================

def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = True


def is_image_file(p: Path) -> bool:
	return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}


def list_images_direct(root: Path) -> List[Path]:
	if not root.exists() or not root.is_dir():
		return []
	return [p for p in root.iterdir() if is_image_file(p)]


def parse_parts_keep(parts_csv: str) -> Tuple[str, ...]:
	vals = [x.strip() for x in parts_csv.split(",") if x.strip()]
	return tuple(vals)


# ============================================================
# 2) Dataset scan / selection
# ============================================================

def collect_all_images_from_dataset(
	data_root: Path,
	parts_keep=("hand", "leaf", "flower", "fruit", "seed", "root"),
) -> Dict[str, List[Path]]:
	species_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
	all_class_imgs: Dict[str, List[Path]] = {}

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
	parts_keep: Tuple[str, ...],
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
	parts_keep: Tuple[str, ...],
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
			rows.append(
				{
					"path": str(img.resolve()),
					"species": species,
					"label_id": species_to_id[species],
					"source": source,
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
	selected_images = set(
		df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist()
	)
	total_available = sum(len(imgs) for imgs in all_class_imgs.values())

	out_csv = out_dir / "dataset_unselected.csv"
	with open(out_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["path", "species", "label_id"])
		for species, all_imgs in all_class_imgs.items():
			label_id = label_map.get(species, -1)
			for img_path in all_imgs:
				if str(img_path) not in selected_images:
					writer.writerow([str(img_path), species, label_id])

	print(
		f"[INFO] Created {out_csv} | total={total_available}, "
		f"selected={len(selected_images)}, unselected={total_available - len(selected_images)}"
	)


# ============================================================
# 3) pHash leakage detection / fixing
# ============================================================

def compute_phash(img_path: str, hash_size: int = 8):
	if not PHASH_AVAILABLE:
		return None
	try:
		img = Image.open(img_path).convert("RGB")
		return str(imagehash.phash(img, hash_size=hash_size))
	except Exception as e:
		print(f"[WARNING] pHash failed for {img_path}: {e}")
		return None


def hamming_distance_int(int1: int, int2: int) -> int:
	return (int1 ^ int2).bit_count()


def check_data_leakage_phash(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	out_dir: Path,
	hash_size: int = 8,
	threshold: int = 5,
) -> Dict:
	if not PHASH_AVAILABLE:
		return {"status": "skipped", "reason": "imagehash_not_installed"}

	print("\n" + "=" * 80)
	print("[DATA LEAKAGE CHECK] pHash cross-split scan")
	print("=" * 80)

	all_items = []
	for split_name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
		for _, row in df.iterrows():
			all_items.append((row["path"], split_name, row["species"]))

	hash_map = {}
	hash_to_paths = {}
	for path, split, cls in tqdm(all_items, desc="Computing pHash"):
		h = compute_phash(path, hash_size)
		if h is None:
			continue
		hash_map[path] = (h, split, cls)
		hash_to_paths.setdefault(h, []).append((path, split, cls))

	exact_cross_split = []
	for h, items in hash_to_paths.items():
		if len(items) <= 1:
			continue
		splits = {x[1] for x in items}
		if len(splits) > 1:
			exact_cross_split.append({"hash": h, "items": items, "splits": list(splits)})

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
			pass

	near_cross_split = []
	checked_pairs = set()
	for bucket_items in tqdm(buckets.values(), desc="Checking near-duplicates"):
		n = len(bucket_items)
		for i in range(n):
			for j in range(i + 1, n):
				p1, int1, s1, c1 = bucket_items[i]
				p2, int2, s2, c2 = bucket_items[j]
				if s1 == s2:
					continue
				pair_key = tuple(sorted([p1, p2]))
				if pair_key in checked_pairs:
					continue
				checked_pairs.add(pair_key)
				dist = hamming_distance_int(int1, int2)
				if 0 < dist <= threshold:
					near_cross_split.append(
						{
							"path1": p1,
							"split1": s1,
							"class1": c1,
							"path2": p2,
							"split2": s2,
							"class2": c2,
							"distance": dist,
						}
					)

	leakage_found = len(exact_cross_split) > 0 or len(near_cross_split) > 0

	report_rows = []
	for group in exact_cross_split:
		for item in group["items"]:
			report_rows.append(
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
	for rec in near_cross_split:
		report_rows.append(
			{
				"type": "near_duplicate",
				"path": rec["path1"],
				"split": rec["split1"],
				"class": rec["class1"],
				"distance": rec["distance"],
				"is_cross_split": True,
				"paired_with": rec["path2"],
			}
		)
		report_rows.append(
			{
				"type": "near_duplicate",
				"path": rec["path2"],
				"split": rec["split2"],
				"class": rec["class2"],
				"distance": rec["distance"],
				"is_cross_split": True,
				"paired_with": rec["path1"],
			}
		)

	if report_rows:
		pd.DataFrame(report_rows).to_csv(out_dir / "data_leakage_check.csv", index=False)

	print(
		f"[LEAKAGE] exact_cross={len(exact_cross_split)} | "
		f"near_cross={len(near_cross_split)} | found={leakage_found}"
	)
	return {
		"status": "completed",
		"leakage_found": leakage_found,
		"exact_cross_split": len(exact_cross_split),
		"near_cross_split": len(near_cross_split),
		"exact_cross_split_details": exact_cross_split,
		"near_cross_split_details": near_cross_split,
	}


def check_image_leakage_with_train(candidate_path: str, train_hashes: Dict[str, str], threshold: int = 5) -> bool:
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


def compute_phash_for_paths(paths: List[str], hash_size: int = 8) -> Dict[str, str]:
	if not PHASH_AVAILABLE:
		return {}
	out = {}
	for p in tqdm(paths, desc="Computing train pHash"):
		h = compute_phash(p, hash_size)
		if h is not None:
			out[p] = h
	return out


def handle_leakage_minor(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	all_class_imgs: Dict[str, List[Path]],
	leakage_result: Dict,
	threshold: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
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
		for split_key, cls_key, path_key in [
			("split1", "class1", "path1"),
			("split2", "class2", "path2"),
		]:
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

	stats_failed = 0

	def move_and_replace(leaked_map, src_rows):
		nonlocal stats_failed, train_rows, train_hashes
		for cls, leaked_paths in leaked_map.items():
			for leaked_path in leaked_paths:
				found_idx = None
				for i, row in enumerate(src_rows):
					if row["path"] == leaked_path:
						found_idx = i
						break
				if found_idx is None:
					continue

				leaked_row = src_rows.pop(found_idx)
				train_rows.append(leaked_row)
				h = compute_phash(leaked_path)
				if h:
					train_hashes[leaked_path] = h

				replaced = False
				for cand in unselected_per_class.get(cls, []):
					if not check_image_leakage_with_train(cand, train_hashes, threshold):
						new_row = leaked_row.copy()
						new_row["path"] = cand
						src_rows.append(new_row)
						unselected_per_class[cls].remove(cand)
						replaced = True
						break
				if not replaced:
					stats_failed += 1

	move_and_replace(leaked_from_val, val_rows)
	move_and_replace(leaked_from_test, test_rows)

	return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows), stats_failed == 0


def build_similarity_groups(df: pd.DataFrame, hash_size: int = 8, threshold: int = 5) -> List[List[str]]:
	paths = df["path"].tolist()
	if not PHASH_AVAILABLE or len(paths) == 0:
		return [[p] for p in paths]

	hashes = {}
	for p in paths:
		h = compute_phash(p, hash_size)
		if h is not None:
			hashes[p] = int(h, 16)

	parent = {p: p for p in paths}
	rank = {p: 0 for p in paths}

	def find(x):
		if parent[x] != x:
			parent[x] = find(parent[x])
		return parent[x]

	def union(a, b):
		ra, rb = find(a), find(b)
		if ra == rb:
			return
		if rank[ra] < rank[rb]:
			ra, rb = rb, ra
		parent[rb] = ra
		if rank[ra] == rank[rb]:
			rank[ra] += 1

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

	groups = {}
	for p in paths:
		root = find(p)
		groups.setdefault(root, []).append(p)
	return list(groups.values())


def stable_species_seed(species: str, seed: int) -> int:
	h = hashlib.sha1(species.encode("utf-8")).hexdigest()
	return (int(h[:8], 16) ^ seed) & 0xFFFFFFFF


def group_aware_split(
	df: pd.DataFrame,
	val_ratio: float = 0.1,
	test_ratio: float = 0.2,
	hash_size: int = 8,
	threshold: int = 5,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	train_rows, val_rows, test_rows = [], [], []

	for species in tqdm(df["species"].unique(), desc="Group-aware split"):
		df_species = df[df["species"] == species].copy()
		groups = build_similarity_groups(df_species, hash_size, threshold)
		n_groups = len(groups)

		n_test = max(1, int(round(n_groups * test_ratio)))
		n_val = max(1, int(round(n_groups * val_ratio)))
		n_train = n_groups - n_test - n_val
		if n_train < 1:
			n_train = 1
			n_val = max(0, n_groups - n_train - n_test)

		rng = random.Random(stable_species_seed(species, seed))
		rng.shuffle(groups)

		test_paths = set(p for g in groups[:n_test] for p in g)
		val_paths = set(p for g in groups[n_test:n_test + n_val] for p in g)

		for _, row in df_species.iterrows():
			if row["path"] in test_paths:
				test_rows.append(row)
			elif row["path"] in val_paths:
				val_rows.append(row)
			else:
				train_rows.append(row)

	return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


def handle_data_leakage(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	df_all: pd.DataFrame,
	all_class_imgs: Dict[str, List[Path]],
	out_dir: Path,
	val_ratio: float = 0.1,
	test_ratio: float = 0.2,
	hash_size: int = 8,
	threshold: int = 5,
	max_iterations: int = 3,
	seed: int = 42,
):
	print("\n" + "=" * 80)
	print("[DATA LEAKAGE HANDLER] start")
	print("=" * 80)

	for it in range(1, max_iterations + 1):
		print(f"\n[Iteration {it}/{max_iterations}]")
		leakage = check_data_leakage_phash(df_train, df_val, df_test, out_dir, hash_size, threshold)

		if not leakage.get("leakage_found", False):
			print("✅ No leakage found")
			return df_train, df_val, df_test, leakage

		total_eval = len(df_val) + len(df_test)
		n_leaked = leakage.get("exact_cross_split", 0) + leakage.get("near_cross_split", 0)
		leakage_pct = (n_leaked / max(1, total_eval)) * 100
		print(f"[INFO] Leakage {n_leaked}/{max(1,total_eval)} ({leakage_pct:.2f}%)")

		if leakage_pct < 5.0:
			print("[ACTION] Minor leakage -> move leaked to train + refill")
			df_train, df_val, df_test, _ = handle_leakage_minor(
				df_train, df_val, df_test, all_class_imgs, leakage, threshold
			)
		else:
			print("[ACTION] Major leakage -> group-aware re-split")
			df_train, df_val, df_test = group_aware_split(
				df_all,
				val_ratio=val_ratio,
				test_ratio=test_ratio,
				hash_size=hash_size,
				threshold=threshold,
				seed=seed,
			)

	final_leakage = check_data_leakage_phash(df_train, df_val, df_test, out_dir, hash_size, threshold)
	return df_train, df_val, df_test, final_leakage


# ============================================================
# 4) Mask2Former foreground extractor (pseudo-mask guidance)
# ============================================================

class Mask2FormerForegroundExtractor:
	def __init__(self, model_id: str, device: torch.device, score_threshold: float = 0.35):
		self.available = False
		self.model_id = model_id
		self.device = device
		self.score_threshold = score_threshold

		try:
			from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

			self.processor = AutoImageProcessor.from_pretrained(model_id)
			self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
			self.model.to(device)
			self.model.eval()
			self.available = True
			print(f"[INFO] Loaded Mask2Former: {model_id}")
		except Exception as e:
			self.processor = None
			self.model = None
			print(f"[WARNING] Could not load Mask2Former ({model_id}): {e}")
			print("          Fallback: use original images (no masking).")

	@torch.no_grad()
	def foreground_mask(self, image: Image.Image) -> np.ndarray:
		h, w = image.height, image.width
		if not self.available:
			return np.ones((h, w), dtype=np.uint8)

		inputs = self.processor(images=image, return_tensors="pt")
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		outputs = self.model(**inputs)

		try:
			panoptic = self.processor.post_process_panoptic_segmentation(
				outputs,
				target_sizes=[(h, w)],
				threshold=self.score_threshold,
			)[0]

			seg = panoptic["segmentation"].cpu().numpy()
			info = panoptic.get("segments_info", [])
			if len(info) == 0:
				return np.ones((h, w), dtype=np.uint8)

			center_y, center_x = h / 2.0, w / 2.0
			scored = []
			for s in info:
				sid = s["id"]
				score = float(s.get("score", 1.0))
				if score < self.score_threshold:
					continue
				m = (seg == sid)
				area = int(m.sum())
				if area <= 0:
					continue
				ys, xs = np.where(m)
				cy, cx = ys.mean(), xs.mean()
				dist = ((cy - center_y) ** 2 + (cx - center_x) ** 2) ** 0.5
				scored.append((area, -dist, sid))

			if not scored:
				return np.ones((h, w), dtype=np.uint8)

			scored.sort(reverse=True)
			keep_ids = [x[2] for x in scored[:3]]
			fg = np.isin(seg, keep_ids).astype(np.uint8)

			fg_ratio = float(fg.mean())
			if fg_ratio < 0.01 or fg_ratio > 0.98:
				return np.ones((h, w), dtype=np.uint8)
			return fg
		except Exception:
			return np.ones((h, w), dtype=np.uint8)


def masked_output_path(masked_dir: Path, img_path: str) -> Path:
	p = Path(img_path)
	h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:16]
	safe_stem = p.stem.replace(" ", "_")
	return masked_dir / f"{safe_stem}_{h}.jpg"


def apply_mask_to_image(img: Image.Image, mask01: np.ndarray, bg_value: int = 127) -> Image.Image:
    """
    Strategy update: Instead of completely removing the background with a solid color,
    we blur the background to retain context while making the foreground stand out.
    """
    from PIL import ImageFilter
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    
    # Create a blurred version of the original image
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=5))
    blurred_arr = np.array(blurred_img, dtype=np.uint8)
    
    m = mask01.astype(np.uint8)
    if m.ndim == 2:
        m = m[..., None]
    
    # Blend: keep foreground as-is, take background from the blurred version
    out = arr * m + blurred_arr * (1 - m)
    return Image.fromarray(out.astype(np.uint8))


def build_masked_images_cache(
	all_paths: List[str],
	masked_dir: Path,
	extractor: Mask2FormerForegroundExtractor,
	overwrite: bool = False,
) -> Dict[str, str]:
	masked_dir.mkdir(parents=True, exist_ok=True)
	path_map: Dict[str, str] = {}

	for p in tqdm(all_paths, desc="Building Mask2Former masked cache"):
		out_p = masked_output_path(masked_dir, p)
		path_map[p] = str(out_p)

		if out_p.exists() and not overwrite:
			continue

		try:
			img = Image.open(p).convert("RGB")
			fg = extractor.foreground_mask(img)
			masked = apply_mask_to_image(img, fg)
			out_p.parent.mkdir(parents=True, exist_ok=True)
			masked.save(out_p, format="JPEG", quality=95)
		except Exception as e:
			print(f"[WARNING] Mask cache failed for {p}: {e}")
			try:
				Image.open(p).convert("RGB").save(out_p, format="JPEG", quality=95)
			except Exception:
				path_map[p] = p

	return path_map


# ============================================================
# 5) Dataset / transforms
# ============================================================

class PlantImageDataset(Dataset):
	def __init__(self, df: pd.DataFrame, transform=None, input_col: str = "input_path"):
		self.df = df.reset_index(drop=True)
		self.transform = transform
		self.input_col = input_col

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		img = Image.open(row[self.input_col]).convert("RGB")
		if self.transform is not None:
			img = self.transform(img)
		return img, int(row["label_id"])


def get_transforms(img_size=224):
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	train_tfms = transforms.Compose(
		[
			transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(p=0.1),
			transforms.RandomRotation(degrees=15, fill=tuple(int(x * 255) for x in mean)),
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)

	eval_tfms = transforms.Compose(
		[
			transforms.Resize(img_size),
			transforms.CenterCrop(img_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)

	return train_tfms, eval_tfms


# ============================================================
# 6) EfficientNetV2-S model
# ============================================================

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
						in_f = model.classifier[i].in_features
						model.classifier[i] = nn.Linear(in_f, num_classes)
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

	raise RuntimeError("Cannot build EfficientNetV2-S (torchvision/timm unavailable).")


# ============================================================
# 7) Train / eval
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
	model.train()
	running_loss, correct, total = 0.0, 0, 0

	for imgs, labels in tqdm(loader, desc="Train", leave=False):
		imgs = imgs.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)

		if scaler is not None and torch.cuda.is_available():
			with autocast("cuda"):
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

	return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
	model.eval()
	running_loss, correct, total = 0.0, 0, 0
	all_labels, all_preds = [], []

	for imgs, labels in tqdm(loader, desc=desc, leave=False):
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

	y_true = np.concatenate(all_labels) if all_labels else np.array([])
	y_pred = np.concatenate(all_preds) if all_preds else np.array([])

	return (
		running_loss / max(total, 1),
		correct / max(total, 1),
		y_true,
		y_pred,
	)


def save_confusion_matrix(y_true, y_pred, labels, class_names, out_dir: Path):
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
	cm_df.to_csv(out_dir / "confusion_matrix.csv")

	fig, ax = plt.subplots(figsize=(14, 12))
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
	disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
	plt.tight_layout()
	plt.savefig(out_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
	plt.close()


# ============================================================
# 8) Main
# ============================================================

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_root", type=str, required=True)
	parser.add_argument("--out_dir", type=str, default="./outputs_efficientnetv2_mask2former")
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--img_size", type=int, default=224)
	parser.add_argument("--per_class_cap", type=int, default=100)
	parser.add_argument("--val_ratio", type=float, default=0.10)
	parser.add_argument("--test_ratio", type=float, default=0.20)
	parser.add_argument("--use_weighted_sampler", action="store_true")

	parser.add_argument(
		"--parts_keep",
		type=str,
		default="hand,leaf,flower,fruit",
		help="Comma-separated subfolders to prioritize before root-level images.",
	)

	parser.add_argument("--use_mask2former", action="store_true")
	parser.add_argument(
		"--mask2former_model_id",
		type=str,
		default="facebook/mask2former-swin-small-coco-panoptic",
	)
	parser.add_argument("--mask_score_threshold", type=float, default=0.35)
	parser.add_argument("--overwrite_mask_cache", action="store_true")

	args = parser.parse_args()

	assert 0.0 < args.test_ratio < 0.5
	assert 0.0 < args.val_ratio < 0.5
	assert args.val_ratio + args.test_ratio < 1.0

	set_seed(args.seed)

	start_time = datetime.now()
	print("=" * 80)
	print(f"[START] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print("=" * 80)

	data_root = Path(args.data_root)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	parts_keep = parse_parts_keep(args.parts_keep)

	# 1) Scan + selected set
	print("\n[1/8] Scanning dataset...")
	df = scan_dataset(
		data_root=data_root,
		parts_keep=parts_keep,
		per_class_cap=args.per_class_cap,
		seed=args.seed,
	)
	if len(df) == 0:
		raise RuntimeError("No valid images found.")

	df.to_csv(out_dir / "dataset_selected.csv", index=False)

	species_list = sorted(df["species"].unique().tolist())
	label_map = {s: int(df[df["species"] == s]["label_id"].iloc[0]) for s in species_list}
	with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
		json.dump(label_map, f, ensure_ascii=False, indent=2)

	print(f"Selected={len(df)} images | classes={len(species_list)}")

	# 2) Collect all images for unselected tracking
	print("\n[2/8] Collecting all available images...")
	all_class_imgs = collect_all_images_from_dataset(data_root)
	print(f"Total available images: {sum(len(v) for v in all_class_imgs.values())}")

	# 3) Split
	print("\n[3/8] Split train/val/test...")
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

	for name, dfx in [("train", df_train), ("val", df_val), ("test", df_test)]:
		dfx.to_csv(out_dir / f"{name}.csv", index=False)
		print(f"{name}: {len(dfx)}")

	create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

	# 4) Leakage handling
	print("\n[4/8] Leakage detection/fix...")
	df_train, df_val, df_test, leakage_result = handle_data_leakage(
		df_train=df_train,
		df_val=df_val,
		df_test=df_test,
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

	# Save updated splits
	for name, dfx in [("train", df_train), ("val", df_val), ("test", df_test)]:
		dfx.to_csv(out_dir / f"{name}.csv", index=False)

	create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

	# 5) Mask2Former guidance
	print("\n[5/8] Preparing model inputs...")
	for dfx in [df_train, df_val, df_test]:
		dfx["input_path"] = dfx["path"]

	if args.use_mask2former:
		device_mask = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		extractor = Mask2FormerForegroundExtractor(
			model_id=args.mask2former_model_id,
			device=device_mask,
			score_threshold=args.mask_score_threshold,
		)

		if extractor.available:
			masked_dir = out_dir / "mask2former_cache"
			unique_paths = sorted(
				set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())
			)
			path_map = build_masked_images_cache(
				all_paths=unique_paths,
				masked_dir=masked_dir,
				extractor=extractor,
				overwrite=args.overwrite_mask_cache,
			)
			df_train["input_path"] = df_train["path"].map(lambda x: path_map.get(x, x))
			df_val["input_path"] = df_val["path"].map(lambda x: path_map.get(x, x))
			df_test["input_path"] = df_test["path"].map(lambda x: path_map.get(x, x))
			print("[INFO] Mask2Former-guided images are enabled.")
		else:
			print("[INFO] Mask2Former unavailable -> using original images.")

	# Save final split CSVs with input_path
	for name, dfx in [("train", df_train), ("val", df_val), ("test", df_test)]:
		dfx.to_csv(out_dir / f"{name}.csv", index=False)

	# 6) DataLoader
	print("\n[6/8] Building loaders...")
	train_tfms, eval_tfms = get_transforms(img_size=args.img_size)
	ds_train = PlantImageDataset(df_train, transform=train_tfms, input_col="input_path")
	ds_val = PlantImageDataset(df_val, transform=eval_tfms, input_col="input_path")
	ds_test = PlantImageDataset(df_test, transform=eval_tfms, input_col="input_path")

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

	# 7) Model
	print("\n[7/8] Building model...")
	num_classes = len(species_list)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model, backend = build_efficientnetv2_s(num_classes)
	model = model.to(device)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

	# 8) Train + eval + exports
	print("\n[8/8] Training...")
	best_val_acc = 0.0
	best_ckpt = out_dir / "best_model.pt"
	patience = 10
	no_improve = 0

	history = {
		"epoch": [],
		"train_loss": [],
		"train_acc": [],
		"val_loss": [],
		"val_acc": [],
		"epoch_time_sec": [],
	}

	print(f"Start training on {device} | backend={backend}")
	for epoch in range(1, args.epochs + 1):
		epoch_start = time.time()
		print("\n" + "-" * 80)
		print(f"Epoch {epoch}/{args.epochs}")
		print(f"Epoch start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

		train_loss, train_acc = train_one_epoch(
			model, train_loader, criterion, optimizer, device, scaler
		)
		val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc="Val")
		scheduler.step()

		epoch_time = time.time() - epoch_start
		print(f"Train | loss={train_loss:.4f}, acc={train_acc:.4f}")
		print(f"Val   | loss={val_loss:.4f}, acc={val_acc:.4f}")
		print(f"Epoch time: {epoch_time:.2f}s")

		history["epoch"].append(epoch)
		history["train_loss"].append(float(train_loss))
		history["train_acc"].append(float(train_acc))
		history["val_loss"].append(float(val_loss))
		history["val_acc"].append(float(val_acc))
		history["epoch_time_sec"].append(float(epoch_time))

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			no_improve = 0
			state = {
				"epoch": epoch,
				"model_state": model.state_dict(),
				"optimizer_state": optimizer.state_dict(),
				"val_acc": best_val_acc,
				"label_map": label_map,
				"backend": backend,
				"args": vars(args),
			}
			torch.save(state, best_ckpt)
			print(f"✅ Saved best checkpoint -> {best_ckpt}")
		else:
			no_improve += 1
			if no_improve >= patience:
				print(f"Early stopping (no improvement for {patience} epochs).")
				break

		torch.save(history, out_dir / "training_history.pt")

	torch.save(history, out_dir / "training_history.pt")

	if best_ckpt.exists():
		ckpt = torch.load(best_ckpt, map_location="cpu")
		model.load_state_dict(ckpt["model_state"])
		print(f"Loaded best checkpoint (val_acc={ckpt.get('val_acc', -1):.4f})")

	test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, desc="Test")
	print(f"\nTest | loss={test_loss:.4f}, acc={test_acc:.4f}")

	report_dict = classification_report(
		y_true,
		y_pred,
		labels=list(range(num_classes)),
		target_names=species_list,
		zero_division=0,
		output_dict=True,
	)
	pd.DataFrame(report_dict).transpose().to_csv(out_dir / "test_classification_report.csv")

	save_confusion_matrix(
		y_true=y_true,
		y_pred=y_pred,
		labels=list(range(num_classes)),
		class_names=species_list,
		out_dir=out_dir,
	)

	with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
		json.dump(
			{
				"test_loss": float(test_loss),
				"test_acc": float(test_acc),
				"best_val_acc": float(best_val_acc),
				"splits": {
					"train": int(len(df_train)),
					"val": int(len(df_val)),
					"test": int(len(df_test)),
				},
				"leakage": leakage_result,
			},
			f,
			indent=2,
			ensure_ascii=False,
		)

	end_time = datetime.now()
	duration = end_time - start_time
	total_seconds = int(duration.total_seconds())
	hours = total_seconds // 3600
	minutes = (total_seconds % 3600) // 60
	seconds = total_seconds % 60

	print("\n" + "=" * 80)
	print(f"[END] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"[DURATION] {hours}h {minutes}m {seconds}s")
	print("=" * 80)
	print("Done.")


if __name__ == "__main__":
	main()
