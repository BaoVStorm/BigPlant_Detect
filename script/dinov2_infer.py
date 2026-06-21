"""Inference + ONNX-export adapter for the DINOv2 classifier.

Mirrors the convention of the other ``*_infer.py`` modules: it re-uses the
exact model class defined in the training script (``script/dinov2.py``) so the
checkpoint ``state_dict`` keys line up 1:1, and exposes a thin inference
adapter whose ``forward`` returns the 4-tuple ``(logits, aux_org, probs,
entropy)`` expected by ``app.services.predictor``.

DINOv2 is an image-only classifier (no organ prior, no auxiliary head), so the
adapter ignores the ``prior`` argument and reports ``None`` for the aux / MoE
routing slots. For ONNX it is exported as a single-input / single-output graph
via ``export_image_model_to_onnx``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _THIS_DIR / "dinov2.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("dinov2_train", os.fspath(_TRAIN_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_train = _load_training_module()
DINOv2Classifier = _train.DINOv2Classifier


class DINOv2InferenceAdapter(nn.Module):
    """Adapter exposing the predictor-compatible 4-tuple interface."""

    def __init__(self, model: "DINOv2Classifier"):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None, training: bool = False):
        logits = self.model(x)
        return logits, None, None, None
