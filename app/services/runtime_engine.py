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
            self._ensure_tensorrt_windows_dlls()

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

    @staticmethod
    def _ensure_tensorrt_windows_dlls() -> None:
        """
        Ensure TensorRT runtime DLL is loadable on Windows.

        We try common DLL names and search dirs from:
        - TRT_DLL_DIR (direct bin folder)
        - TRT_ROOT (+ "bin")
        - PATH
        """
        dll_names = [
            "nvinfer_10.dll",
            "nvinfer_9.dll",
            "nvinfer_8.dll",
            "nvinfer.dll",
        ]

        search_dirs = []

        trt_dll_dir = os.getenv("TRT_DLL_DIR", "").strip().strip('"')
        if trt_dll_dir:
            search_dirs.append(trt_dll_dir)

        trt_root = os.getenv("TRT_ROOT", "").strip().strip('"')
        if trt_root:
            search_dirs.append(os.path.join(trt_root, "bin"))

        for p in os.environ.get("PATH", "").split(os.pathsep):
            p = p.strip().strip('"')
            if p:
                search_dirs.append(p)

        # De-duplicate while preserving order
        seen = set()
        uniq_dirs = []
        for d in search_dirs:
            if d not in seen:
                seen.add(d)
                uniq_dirs.append(d)

        if hasattr(os, "add_dll_directory"):
            for d in uniq_dirs:
                if os.path.isdir(d):
                    try:
                        os.add_dll_directory(d)
                    except Exception:
                        pass

        # 1) try full-path load from known dirs first (more reliable)
        for d in uniq_dirs:
            for name in dll_names:
                full = os.path.join(d, name)
                if os.path.isfile(full):
                    try:
                        ctypes.WinDLL(full)
                        return
                    except OSError:
                        continue

        # 2) then try by name
        for name in dll_names:
            try:
                ctypes.WinDLL(name)
                return
            except OSError:
                continue

        raise RuntimeError(
            "TensorRT runtime DLL not loadable on Windows. "
            "Expected one of: nvinfer_10.dll / nvinfer_9.dll / nvinfer_8.dll. "
            "Set TRT_DLL_DIR to TensorRT bin directory (e.g. D:\\App\\TensorRT-10.x\\bin) "
            "or add that bin folder to PATH."
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
