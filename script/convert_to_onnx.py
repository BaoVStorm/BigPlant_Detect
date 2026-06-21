"""Convert the three new BigPlant models (DINOv2, Organ-Aware V-MoE, Hybrid)
to ONNX.

The model class definitions live in the training scripts inside this folder.
We load those modules dynamically (their heavy code is guarded by
``if __name__ == "__main__":``) so the *exact* architecture is reused and the
checkpoint ``state_dict`` keys match without any reimplementation.

ONNX export notes
-----------------
* DINOv2 is a plain ``backbone -> linear head`` graph: exported directly,
  single ``image`` input, single ``logits`` output.
* Organ-Aware V-MoE and the Hybrid model contain a Switch-MoE whose training
  ``forward`` uses data-dependent Python loops (``nonzero``/capacity drop/
  per-token weight gather). That control flow is NOT trace-safe and cannot be
  exported faithfully. For inference (eval, ``top_k == 1``) the routing reduces
  to "send each token to its arg-max expert with weight 1.0" and the capacity
  limit is never hit at batch size 1. The wrappers below reproduce exactly that
  behaviour with a dense, fully-static graph (every expert is evaluated on every
  token, then gathered) which is ONNX/TensorRT friendly and numerically equal to
  the eval forward at batch 1.
* The MoE/Hybrid graphs take two inputs ``(image, organ_prior)`` and return
  ``(logits, aux_org)`` to match ``app/services/onnx_export._OnnxWrapper`` and the
  runtime contract in ``app/services/predictor.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"
MODEL_DIR = ROOT / "model"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_training_module(filename: str, alias: str):
    """Import a (possibly hyphenated) training script by file path."""
    path = SCRIPT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Training script not found: {path}")
    spec = importlib.util.spec_from_file_location(alias, os.fspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_ckpt(path: str) -> dict:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint format in {path}: {type(ckpt)}")
    return ckpt


def _num_classes(ckpt: dict) -> int:
    if ckpt.get("n_classes"):
        return int(ckpt["n_classes"])
    c2i = ckpt.get("class_to_idx") or {}
    if not c2i:
        raise ValueError("Checkpoint missing n_classes and class_to_idx")
    return len(c2i)


def _args(ckpt: dict) -> dict:
    return ckpt.get("args", {}) or {}


def _load_state(model: nn.Module, ckpt: dict) -> None:
    state = ckpt.get("model_state_dict") or ckpt.get("model_state")
    if state is None:
        raise ValueError("Checkpoint missing model_state_dict/model_state")
    state = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Buffers such as switch.expert_usage are training-only stats; safe to drop.
    missing = [k for k in missing if not k.endswith("expert_usage")]
    unexpected = [k for k in unexpected if not k.endswith("expert_usage")]
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    if not missing and not unexpected:
        print("  [ok] state_dict matched exactly")


def _verify(onnx_path: str, inputs: dict, torch_logits: torch.Tensor) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - optional dep
        print(f"  [skip] onnxruntime not available, skipping verification ({exc})")
        return
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {k: v.cpu().numpy().astype("float32") for k, v in inputs.items()}
    out = sess.run(None, feed)[0]
    diff = float(np.max(np.abs(out - torch_logits.detach().cpu().numpy())))
    print(f"  [verify] max|onnx - torch| logits = {diff:.3e}")


# ---------------------------------------------------------------------------
# ONNX-safe wrappers for the MoE-based models (eval, top_k == 1)
# ---------------------------------------------------------------------------
def _dense_moe_patch_out(switch: nn.Module, patches: torch.Tensor, organ: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eval-time Switch-MoE output, fully static.

    patches: (B, T, D); organ: (B, T, O)
    Returns (out (B, T, D), probs (B, T, E)).
    Equivalent to the training forward at eval / top_k=1 / batch 1.
    """
    B, T, D = patches.shape
    flat = patches.reshape(B * T, D)
    flat_prior = organ.reshape(B * T, -1)

    probs = F.softmax(switch.router(flat, flat_prior), dim=-1)  # (B*T, E)
    sel = torch.argmax(probs, dim=-1)  # (B*T,)

    # Dense: evaluate every expert on every token, then gather the selected one.
    expert_stack = torch.stack([expert(flat) for expert in switch.experts], dim=1)  # (B*T, E, D)
    gather_idx = sel.view(-1, 1, 1).expand(-1, 1, D)
    out = torch.gather(expert_stack, 1, gather_idx).squeeze(1)  # (B*T, D)
    return out.reshape(B, T, D), probs.reshape(B, T, -1)


class OrganMoEOnnxWrapper(nn.Module):
    """Wrap OrganAwareSwitchViT -> (logits, aux_org)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, organ_prior: torch.Tensor):
        m = self.model
        tokens = m.backbone.forward_features(image)  # (B, N+1, D)
        cls = tokens[:, 0:1, :]
        patches = tokens[:, 1:, :]
        B, T, _ = patches.shape
        organ = organ_prior.unsqueeze(1).expand(B, T, -1)

        out, _ = _dense_moe_patch_out(m.switch, patches, organ)
        tokens2 = torch.cat([cls, out], dim=1)
        cls_final = m.ln(tokens2[:, 0, :])
        logits = m.cls_head(cls_final)
        aux_org = m.aux_head(cls_final)
        return logits, aux_org


class HybridOnnxWrapper(nn.Module):
    """Wrap HybridPlantModel -> (logits, aux_org)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, organ_prior: torch.Tensor):
        m = self.model
        ob = m.organ_branch
        tokens = ob.backbone.forward_features(image)
        cls = tokens[:, 0, :]
        patches = tokens[:, 1:, :]
        B, T, _ = patches.shape
        organ = organ_prior.unsqueeze(1).expand(B, T, -1)

        out, _ = _dense_moe_patch_out(ob.switch, patches, organ)
        pooled_patch = out.mean(dim=1)
        fused_cls = ob.ln(cls + pooled_patch)
        organ_logits = ob.cls_head(fused_cls)
        aux_org = ob.aux_head(cls)

        dino_logits = m.dino_branch(image)
        seg_logits = m.segformer_branch(image)
        fused = m.fusion(torch.cat([organ_logits, dino_logits, seg_logits], dim=-1))
        return fused, aux_org


# ---------------------------------------------------------------------------
# per-model export entry points
# ---------------------------------------------------------------------------
def export_dinov2(ckpt_path: str, out_path: str, opset: int) -> None:
    print(f"[dinov2] checkpoint: {ckpt_path}")
    mod = _load_training_module("dinov2.py", "_bp_dinov2_train")
    ckpt = _load_ckpt(ckpt_path)
    n_classes = _num_classes(ckpt)
    args = _args(ckpt)

    model = mod.DINOv2Classifier(
        model_name=ckpt.get("model_name", args.get("model_name", "dinov2_vitb14")),
        n_classes=n_classes,
        pretrained=False,
        freeze_backbone=bool(ckpt.get("freeze_backbone", True)),
        dropout=float(ckpt.get("dropout", args.get("dropout", 0.1))),
    ).eval()
    _load_state(model, ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    dummy = torch.randn(1, 3, 224, 224).to(device)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with torch.no_grad():
        torch_logits = model(dummy)
        torch.onnx.export(
            model,
            (dummy,),
            out_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )
    print(f"[dinov2] -> {out_path}")
    _verify(out_path, {"image": dummy}, torch_logits)


def export_organ_moe(ckpt_path: str, out_path: str, opset: int) -> None:
    print(f"[organ-aware-v-moe] checkpoint: {ckpt_path}")
    mod = _load_training_module("organ-aware-v-moe.py", "_bp_organ_moe_train")
    ckpt = _load_ckpt(ckpt_path)
    n_classes = _num_classes(ckpt)
    args = _args(ckpt)
    organ_dim = int(ckpt.get("organ_dim", args.get("n_org_clusters", 5)))

    model = mod.OrganAwareSwitchViT(
        vit_name=args.get("vit_name", "vit_base_patch16_224"),
        n_classes=n_classes,
        organ_dim=organ_dim,
        n_experts=int(ckpt.get("n_experts", args.get("n_experts", 8))),
        d_ff_expert=int(args.get("d_ff_expert", 1024)),
        top_k=int(args.get("top_k", 1)),
        pretrained=False,
    ).eval()
    _load_state(model, ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = OrganMoEOnnxWrapper(model).eval().to(device)
    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    dummy_prior = torch.full((1, organ_dim), 1.0 / organ_dim).to(device)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with torch.no_grad():
        torch_logits, _ = wrapper(dummy_img, dummy_prior)
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_prior),
            out_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image", "organ_prior"],
            output_names=["logits", "aux_org"],
            dynamic_axes={
                "image": {0: "batch"},
                "organ_prior": {0: "batch"},
                "logits": {0: "batch"},
                "aux_org": {0: "batch"},
            },
        )
    print(f"[organ-aware-v-moe] -> {out_path}")
    _verify(out_path, {"image": dummy_img, "organ_prior": dummy_prior}, torch_logits)


def export_hybrid(ckpt_path: str, out_path: str, opset: int) -> None:
    print(f"[hybrid-model] checkpoint: {ckpt_path}")
    mod = _load_training_module("hybrid-model.py", "_bp_hybrid_train")
    ckpt = _load_ckpt(ckpt_path)
    n_classes = _num_classes(ckpt)
    args = _args(ckpt)
    organ_dim = int(ckpt.get("organ_dim", args.get("n_org_clusters", 5)))

    model = mod.HybridPlantModel(
        n_classes=n_classes,
        organ_dim=organ_dim,
        n_experts=int(ckpt.get("n_experts", args.get("n_experts", 8))),
        d_ff_expert=int(args.get("d_ff_expert", 1024)),
        vit_name=args.get("vit_name", "vit_base_patch16_224"),
        dino_model_name=args.get("dino_model_name", "dinov2_vitb14"),
        segformer_model_name=args.get("segformer_model_name", "nvidia/segformer-b1-finetuned-ade-512-512"),
        dropout=float(args.get("dropout", 0.1)),
        top_k=int(args.get("top_k", 1)),
        freeze_dino=not bool(args.get("unfreeze_dino", False)),
        freeze_segformer=not bool(args.get("unfreeze_segformer", False)),
    ).eval()
    _load_state(model, ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = HybridOnnxWrapper(model).eval().to(device)
    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    dummy_prior = torch.full((1, organ_dim), 1.0 / organ_dim).to(device)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with torch.no_grad():
        torch_logits, _ = wrapper(dummy_img, dummy_prior)
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_prior),
            out_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image", "organ_prior"],
            output_names=["logits", "aux_org"],
            dynamic_axes={
                "image": {0: "batch"},
                "organ_prior": {0: "batch"},
                "logits": {0: "batch"},
                "aux_org": {0: "batch"},
            },
        )
    print(f"[hybrid-model] -> {out_path}")
    _verify(out_path, {"image": dummy_img, "organ_prior": dummy_prior}, torch_logits)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "dinov2": (MODEL_DIR / "dinov2" / "dinov2.pt", MODEL_DIR / "dinov2" / "dinov2.onnx"),
    "organ-aware-v-moe": (
        MODEL_DIR / "organ-aware-v-moe" / "organ-aware-v-moe.pt",
        MODEL_DIR / "organ-aware-v-moe" / "organ-aware-v-moe.onnx",
    ),
    "hybrid": (
        MODEL_DIR / "hybridmodel" / "hybrid-model.pt",
        MODEL_DIR / "hybridmodel" / "hybrid-model.onnx",
    ),
}

_EXPORTERS = {
    "dinov2": export_dinov2,
    "organ-aware-v-moe": export_organ_moe,
    "hybrid": export_hybrid,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export new BigPlant models to ONNX")
    parser.add_argument(
        "--model",
        choices=["dinov2", "organ-aware-v-moe", "hybrid", "all"],
        default="all",
        help="which model to export",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="override checkpoint path")
    parser.add_argument("--out", type=str, default=None, help="override output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    targets = list(_EXPORTERS) if args.model == "all" else [args.model]
    if len(targets) > 1 and (args.ckpt or args.out):
        parser.error("--ckpt/--out can only be used with a single --model")

    for name in targets:
        default_ckpt, default_out = _DEFAULTS[name]
        ckpt = args.ckpt or os.fspath(default_ckpt)
        out = args.out or os.fspath(default_out)
        if not os.path.isfile(ckpt):
            print(f"[{name}] checkpoint not found, skipping: {ckpt}")
            continue
        _EXPORTERS[name](ckpt, out, args.opset)


if __name__ == "__main__":
    main()
