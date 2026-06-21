"""Inference + ONNX-export adapter for the Hybrid Plant model.

Re-uses ``HybridPlantModel`` from ``script/hybrid-model.py`` unchanged so the
checkpoint ``state_dict`` keys match. Only the forward path is re-implemented to
be ONNX-friendly:

* The DINOv2 branch and the SegFormer branch already trace cleanly, so they are
  called as-is.
* The Organ-Aware Switch-ViT branch contains the same data-dependent Switch
  routing loops as the standalone V-MoE model, so its routing is replaced by a
  **dense** top-1 formulation (every expert evaluated, selected output gathered)
  that is mathematically identical to eval top-1 routing but free of
  data-dependent control flow.
* The model's ``forward`` returns a dict; the adapter flattens it to the
  ``(logits, aux_org, probs, entropy)`` 4-tuple expected by the predictor and by
  ``app.services.onnx_export._OnnxWrapper``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _THIS_DIR / "hybrid-model.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("hybrid_model_train", os.fspath(_TRAIN_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_train = _load_training_module()
HybridPlantModel = _train.HybridPlantModel


def _dense_switch(switch: nn.Module, tokens: torch.Tensor, organ_tokens: torch.Tensor):
    """ONNX-safe equivalent of ``SwitchMoE.forward`` for eval top-1 routing."""
    B, T, D = tokens.shape
    flat = tokens.reshape(B * T, D)
    flat_prior = organ_tokens.reshape(B * T, -1)

    logits = switch.router(flat, flat_prior)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)

    sel = probs.argmax(dim=-1)
    expert_outs = torch.stack([expert(flat) for expert in switch.experts], dim=1)
    gather_idx = sel.view(-1, 1, 1).expand(-1, 1, D)
    out = expert_outs.gather(1, gather_idx).squeeze(1)

    return out.reshape(B, T, D), probs.reshape(B, T, -1), entropy.reshape(B, T)


class HybridInferenceAdapter(nn.Module):
    """Predictor / ONNX compatible adapter for ``HybridPlantModel``."""

    def __init__(self, model: "HybridPlantModel"):
        super().__init__()
        self.model = model

    def _organ_branch(self, x: torch.Tensor, prior: torch.Tensor):
        ob = self.model.organ_branch
        if hasattr(ob.backbone, "forward_features"):
            tokens = ob.backbone.forward_features(x)
        else:
            tokens = ob.backbone(x)

        if tokens.dim() == 2:
            cls = tokens
            patches = tokens.unsqueeze(1)
        else:
            cls = tokens[:, 0, :]
            patches = tokens[:, 1:, :]

        B, T, _ = patches.shape
        organ_tokens = prior.unsqueeze(1).expand(B, T, -1)
        patches_out, probs, entropy = _dense_switch(ob.switch, patches, organ_tokens)

        pooled_patch = patches_out.mean(dim=1)
        fused_cls = ob.ln(cls + pooled_patch)
        species_logits = ob.cls_head(fused_cls)
        aux_org_logits = ob.aux_head(cls)
        return species_logits, aux_org_logits, probs, entropy

    def forward(self, x: torch.Tensor, prior: torch.Tensor, training: bool = False):
        m = self.model
        organ_logits, aux_org_logits, probs, entropy = self._organ_branch(x, prior)
        dino_logits = m.dino_branch(x)
        seg_logits = m.segformer_branch(x)

        fused = m.fusion(torch.cat([organ_logits, dino_logits, seg_logits], dim=-1))
        return fused, aux_org_logits, probs, entropy
