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
        self._ort = _import_onnxruntime_with_trt_check()
        self.session = _create_trt_session(
            ort=self._ort,
            onnx_path=onnx_path,
            engine_cache_dir=engine_cache_dir,
            use_fp16=use_fp16,
            strict=strict,
            workspace_gb=workspace_gb,
            device_id=device_id,
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
        core_names = [
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
        loaded_core = None
        for d in uniq_dirs:
            for name in core_names:
                full = os.path.join(d, name)
                if os.path.isfile(full):
                    try:
                        ctypes.WinDLL(full)
                        loaded_core = name
                        break
                    except OSError:
                        continue
            if loaded_core is not None:
                break

        # 2) then try by name
        if loaded_core is None:
            for name in core_names:
                try:
                    ctypes.WinDLL(name)
                    loaded_core = name
                    break
                except OSError:
                    continue

        if loaded_core is None:
            raise RuntimeError(
                "TensorRT runtime DLL not loadable on Windows. "
                "Expected one of: nvinfer_10.dll / nvinfer_9.dll / nvinfer_8.dll. "
                "Set TRT_DLL_DIR to TensorRT bin directory (e.g. D:\\App\\TensorRT-10.x\\bin) "
                "or add that bin folder to PATH."
            )

        # 3) verify TensorRT plugin/parser DLLs for ORT provider.
        # This catches common startup-time fallback to CPU with unclear message.
        major_suffix = ""
        if loaded_core.startswith("nvinfer_") and loaded_core.endswith(".dll"):
            major_suffix = loaded_core.replace("nvinfer_", "").replace(".dll", "")

        plugin_candidates = []
        parser_candidates = []
        if major_suffix:
            plugin_candidates.append(f"nvinfer_plugin_{major_suffix}.dll")
            parser_candidates.append(f"nvonnxparser_{major_suffix}.dll")
        plugin_candidates.append("nvinfer_plugin.dll")
        parser_candidates.append("nvonnxparser.dll")

        for group, label in [
            (plugin_candidates, "plugin"),
            (parser_candidates, "parser"),
        ]:
            group_loaded = False
            for name in group:
                # Try full-path load from known dirs first
                for d in uniq_dirs:
                    full = os.path.join(d, name)
                    if os.path.isfile(full):
                        try:
                            ctypes.WinDLL(full)
                            group_loaded = True
                            break
                        except OSError:
                            continue
                if group_loaded:
                    break
                # Fallback: try by name only
                try:
                    ctypes.WinDLL(name)
                    group_loaded = True
                    break
                except OSError:
                    continue
            if not group_loaded:
                raise RuntimeError(
                    f"TensorRT {label} DLL missing (tried: {group}). "
                    "Add TensorRT bin folder to PATH/TRT_DLL_DIR."
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


def _import_onnxruntime_with_trt_check():
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(
            "onnxruntime is required for TensorRT backend. Install onnxruntime-gpu."
        ) from exc

    available = set(ort.get_available_providers())
    if "TensorrtExecutionProvider" not in available:
        raise RuntimeError(
            "TensorRT EP is not available in onnxruntime. "
            "Install TensorRT + onnxruntime-gpu that includes TensorrtExecutionProvider."
        )

    if os.name == "nt":
        TensorRTRuntime._ensure_tensorrt_windows_dlls()

    return ort


def _create_trt_session(
    ort,
    onnx_path: str,
    engine_cache_dir: str,
    use_fp16: bool,
    strict: bool,
    workspace_gb: int,
    device_id: int,
):
    available = set(ort.get_available_providers())

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

    session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
    active_providers = session.get_providers()
    if strict and "TensorrtExecutionProvider" not in active_providers:
        raise RuntimeError(
            "TensorRT strict mode is enabled but session did not activate TensorRT provider. "
            f"Active providers: {active_providers}"
        )

    return session


class EfficientSegformerTensorRTRuntime:
    def __init__(
        self,
        base_model,
        onnx_path: str,
        engine_cache_dir: str,
        use_fp16: bool,
        strict: bool,
        workspace_gb: int,
        device_id: int,
    ):
        self.base_model = base_model.eval()
        self._ort = _import_onnxruntime_with_trt_check()
        self.session = _create_trt_session(
            ort=self._ort,
            onnx_path=onnx_path,
            engine_cache_dir=engine_cache_dir,
            use_fp16=use_fp16,
            strict=strict,
            workspace_gb=workspace_gb,
            device_id=device_id,
        )

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None):
        with torch.no_grad():
            if hasattr(self.base_model, "build_classifier_input"):
                x_cls = self.base_model.build_classifier_input(x)
            elif (
                hasattr(self.base_model, "seg_model")
                and hasattr(self.base_model, "_build_foreground_mask")
                and hasattr(self.base_model, "_apply_mask")
            ):
                if bool(getattr(self.base_model, "seg_freeze", False)):
                    self.base_model.seg_model.eval()
                    seg_ctx = torch.no_grad()
                else:
                    seg_ctx = torch.enable_grad()

                with seg_ctx:
                    seg_logits = self.base_model.seg_model(x)["out"]
                fg_mask = self.base_model._build_foreground_mask(x, seg_logits)
                x_cls = self.base_model._apply_mask(x, fg_mask)
            else:
                fg_mask = self.base_model._build_foreground_mask(x)
                x_focus = x * (
                    self.base_model.min_keep_bg + (1.0 - self.base_model.min_keep_bg) * fg_mask
                )
                x_cls = (x_focus - self.base_model.mean) / self.base_model.std

        x_np = x_cls.detach().to("cpu").numpy().astype(np.float32, copy=False)
        outputs = self.session.run(None, {"image": x_np})
        logits = torch.from_numpy(outputs[0]).to(x.device)
        return logits, None, None, None

    @property
    def info(self) -> RuntimeInfo:
        return RuntimeInfo(
            backend="tensorrt",
            details={
                "providers": self.session.get_providers(),
                "hybrid": True,
                "hybrid_detail": "segmentation branch in pytorch + classifier in tensorrt",
            },
        )
