"""Inference + ONNX-export adapter for the Organ-Aware Switch-ViT (V-MoE).

The training model (``script/organ-aware-v-moe.py``) is re-used as-is so the
checkpoint ``state_dict`` keys match exactly. Only the *forward* path is
re-implemented here in an ONNX-friendly way.

Why a custom forward?
---------------------
``SwitchMoE.forward`` performs Switch routing with Python control flow that is
data-dependent at runtime: ``mask.nonzero()``, a per-expert capacity ``if``
branch, and an inner ``for i, token_idx in enumerate(idx)`` loop. Under
``torch.onnx.export`` (tracing) those loops are *unrolled to the shape seen for
the dummy input*, which produces an incorrect graph for any other input.

For inference the model runs in eval mode with ``top_k`` experts (saved value,
typically 1). We reproduce that exactly with a **dense** formulation: every
expert is evaluated on every patch token and the per-token top-1 expert output
is gathered. This is mathematically identical to the eval top-1 Switch routing
at batch size 1 (where the capacity limit is never hit) but contains no
data-dependent control flow, so it traces to a clean, static ONNX graph.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _THIS_DIR / "organ-aware-v-moe.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("organ_aware_v_moe_train", os.fspath(_TRAIN_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_train = _load_training_module()
OrganAwareSwitchViT = _train.OrganAwareSwitchViT


def _dense_switch(switch: nn.Module, tokens: torch.Tensor, organ_tokens: torch.Tensor):
    """ONNX-safe equivalent of ``SwitchMoE.forward`` for eval top-1 routing.

    Args:
        switch: a ``SwitchMoE`` module (provides ``.router`` and ``.experts``).
        tokens: patch tokens ``(B, T, D)``.
        organ_tokens: organ prior broadcast per patch ``(B, T, organ_dim)``.

    Returns:
        out ``(B, T, D)``, probs ``(B, T, E)``, entropy ``(B, T)``.
    """
    B, T, D = tokens.shape
    flat = tokens.reshape(B * T, D)
    flat_prior = organ_tokens.reshape(B * T, -1)

    logits = switch.router(flat, flat_prior)          # (B*T, E)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)  # (B*T,)

    # Dense top-1 routing: evaluate every expert, gather the selected one.
    sel = probs.argmax(dim=-1)                          # (B*T,)
    expert_outs = torch.stack([expert(flat) for expert in switch.experts], dim=1)  # (B*T, E, D)
    gather_idx = sel.view(-1, 1, 1).expand(-1, 1, D)
    out = expert_outs.gather(1, gather_idx).squeeze(1)  # (B*T, D)

    return (
        out.reshape(B, T, D),
        probs.reshape(B, T, -1),
        entropy.reshape(B, T),
    )


class OrganAwareVMoEInferenceAdapter(nn.Module):
    """Predictor / ONNX compatible adapter for ``OrganAwareSwitchViT``."""

    def __init__(self, model: "OrganAwareSwitchViT"):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, prior: torch.Tensor, training: bool = False):
        m = self.model

        if hasattr(m.backbone, "forward_features"):
            tokens = m.backbone.forward_features(x)
        else:
            tokens = m.backbone(x)

        cls = tokens[:, 0:1, :]            # (B, 1, D)
        patches = tokens[:, 1:, :]         # (B, T, D)
        B, T, _ = patches.shape

        organ_tokens = prior.unsqueeze(1).expand(B, T, -1)
        out, probs, entropy = _dense_switch(m.switch, patches, organ_tokens)

        tokens2 = torch.cat([cls, out], dim=1)
        cls_final = m.ln(tokens2[:, 0, :])
        logits = m.cls_head(cls_final)
        aux_org = m.aux_head(cls_final)
        return logits, aux_org, probs, entropy
