from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


class _OnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, organ_prior: torch.Tensor):
        logits, aux_org, _, _ = self.model(image, organ_prior, training=False)
        return logits, aux_org


def export_model_to_onnx(
    model: nn.Module,
    organ_dim: int,
    out_path: str,
    opset: int = 17,
) -> str:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    original_device = next(model.parameters()).device

    wrapper = _OnnxWrapper(model).eval()
    wrapper.to("cpu")

    dummy_x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    dummy_prior = torch.full((1, organ_dim), 1.0 / organ_dim, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_prior),
        os.fspath(out_file),
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

    wrapper.to(original_device)
    if was_training:
        model.train()
    else:
        model.eval()

    return os.fspath(out_file)


def export_image_model_to_onnx(
    model: nn.Module,
    out_path: str,
    opset: int = 17,
    input_name: str = "image",
    output_name: str = "logits",
) -> str:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    original_device = next(model.parameters()).device

    model.eval()
    model.to("cpu")

    dummy_x = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_x,),
        os.fspath(out_file),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={
            input_name: {0: "batch"},
            output_name: {0: "batch"},
        },
    )

    model.to(original_device)
    if was_training:
        model.train()
    else:
        model.eval()

    return os.fspath(out_file)
