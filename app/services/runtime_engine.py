from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import os
import ctypes

import numpy as np
import torch


@dataclass
class RuntimeInfo:
    backend: str
    details: Dict[str, Any]


class TensorRTRuntime:
    def __init__(
        self,
        onnx_path: str,
        engine_cache_dir: str,
        use_fp16: bool,
        strict: bool,
        workspace_gb: int,
        device_id: int,
    ):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError(
                "onnxruntime is required for TensorRT backend. Install onnxruntime-gpu."
            ) from exc

        self._ort = ort
        available = set(ort.get_available_providers())
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(
                "TensorRT EP is not available in onnxruntime. "
                "Install TensorRT + onnxruntime-gpu that includes TensorrtExecutionProvider."
            )

        if os.name == "nt":
            try:
                ctypes.WinDLL("nvinfer_10.dll")
            except OSError as exc:
                raise RuntimeError(
                    "TensorRT runtime DLL not found: nvinfer_10.dll. "
                    "Install TensorRT 10.x and add its lib folder to PATH."
                ) from exc

        trt_options = {
            "device_id": device_id,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": engine_cache_dir,
            "trt_fp16_enable": use_fp16,
            "trt_max_workspace_size": int(workspace_gb) * 1024 * 1024 * 1024,
        }

        providers = [("TensorrtExecutionProvider", trt_options)]
        if not strict:
            if "CUDAExecutionProvider" in available:
                providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
            providers.append("CPUExecutionProvider")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
        active_providers = self.session.get_providers()
        if strict and "TensorrtExecutionProvider" not in active_providers:
            raise RuntimeError(
                "TensorRT strict mode is enabled but session did not activate TensorRT provider. "
                f"Active providers: {active_providers}"
            )

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None, None, None]:
        x_np = x.detach().to("cpu").numpy().astype(np.float32, copy=False)
        prior_np = prior.detach().to("cpu").numpy().astype(np.float32, copy=False)

        outputs = self.session.run(None, {"image": x_np, "organ_prior": prior_np})
        logits = torch.from_numpy(outputs[0]).to(x.device)
        aux_org = torch.from_numpy(outputs[1]).to(x.device) if len(outputs) > 1 else None
        return logits, aux_org, None, None

    @property
    def info(self) -> RuntimeInfo:
        return RuntimeInfo(
            backend="tensorrt",
            details={
                "providers": self.session.get_providers(),
            },
        )
