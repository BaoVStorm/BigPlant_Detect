import os
import importlib.util
from typing import Any, Dict, List, Tuple

import torch

from app.services.onnx_export import export_image_model_to_onnx
from app.services.runtime_engine import EfficientSegformerTensorRTRuntime


SUPPORTED_MODEL_SCRIPT = "mobilenetv3large-deeplabv3"


def _normalize_model_script(model_script: str) -> str:
    return (model_script or "").strip().lower()


def _resolve_model_class(model_script: str):
    script_name = _normalize_model_script(model_script)
    if script_name != SUPPORTED_MODEL_SCRIPT:
        raise ValueError(
            f"Unsupported MODEL_SCRIPT='{model_script}'. "
            f"Only '{SUPPORTED_MODEL_SCRIPT}' is supported."
        )

    try:
        from script.mobilenetv3large_deeplabv3_infer import (
            MobileNetV3DeepLabV3Classifier,
            MobileNetDeepLabV3InferenceAdapter,
        )

        return MobileNetV3DeepLabV3Classifier, MobileNetDeepLabV3InferenceAdapter
    except Exception:
        pass

    train_script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "script",
        "mobilenetv3large-deeplabv3.py",
    )
    if not os.path.isfile(train_script_path):
        raise RuntimeError(
            "Cannot find mobilenetv3large-deeplabv3 runtime class. "
            "Expected script/mobilenetv3large_deeplabv3_infer.py or script/mobilenetv3large-deeplabv3.py"
        )

    spec = importlib.util.spec_from_file_location("mobilenetv3large_deeplabv3_train", train_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from: {train_script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "HybridMobileNetV3DeepLabV3"):
        raise RuntimeError("Class HybridMobileNetV3DeepLabV3 not found in training script")

    model_cls = module.HybridMobileNetV3DeepLabV3

    class _Adapter(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None, training: bool = False):
            logits = self.model(x)
            return logits, None, None, None

    return model_cls, _Adapter


def _extract_classifier_module(base_model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(base_model, "classifier"):
        return base_model.classifier
    if hasattr(base_model, "cls_model"):
        return base_model.cls_model
    raise RuntimeError("Base model does not expose classifier/cls_model for ONNX export")


def load_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise ValueError("Invalid checkpoint format. Expect a dict checkpoint.")

    if "model_state_dict" in ckpt or "model_state" in ckpt:
        return ckpt

    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return {"model_state_dict": ckpt}

    raise ValueError(
        "Invalid checkpoint format. Expect dict with 'model_state_dict' or 'model_state'."
    )


def build_model_from_ckpt(
    ckpt: Dict[str, Any],
    device: torch.device,
    model_script: str,
) -> Tuple[torch.nn.Module, List[str], Dict[str, Any]]:
    script_name = _normalize_model_script(model_script)
    if script_name != SUPPORTED_MODEL_SCRIPT:
        raise ValueError(
            f"Unsupported MODEL_SCRIPT='{model_script}'. "
            f"Only '{SUPPORTED_MODEL_SCRIPT}' is supported."
        )

    saved_args = ckpt.get("args", {}) or {}

    species_list = ckpt.get("species_list")
    label_map = ckpt.get("label_map") or {}
    if species_list:
        class_names = list(species_list)
    elif label_map:
        idx_to_class = {int(v): k for k, v in label_map.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        raise ValueError("Checkpoint missing species_list/label_map for mobilenetv3large-deeplabv3")

    num_classes = len(class_names)
    model_cls, adapter_cls = _resolve_model_class(script_name)

    seg_pretrained = bool(saved_args.get("seg_pretrained", True))
    seg_freeze = bool(saved_args.get("seg_freeze", True))
    mask_mode = str(saved_args.get("mask_mode", "attention"))

    base_model = model_cls(
        num_classes=num_classes,
        seg_pretrained=seg_pretrained,
        seg_freeze=seg_freeze,
        mask_mode=mask_mode,
    )

    state_dict = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing model_state/model_state_dict")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    if any(k.startswith("cls_model.") for k in state_dict.keys()) and hasattr(base_model, "classifier") and not hasattr(base_model, "cls_model"):
        state_dict = {
            ("classifier." + k[len("cls_model.") :]) if k.startswith("cls_model.") else k: v
            for k, v in state_dict.items()
        }

    try:
        base_model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        if hasattr(base_model, "classifier"):
            classifier_state = {
                k[len("classifier.") :]: v
                for k, v in state_dict.items()
                if k.startswith("classifier.")
            }
            if classifier_state:
                base_model.classifier.load_state_dict(classifier_state, strict=True)

        if hasattr(base_model, "cls_model"):
            cls_state = {
                k[len("cls_model.") :]: v
                for k, v in state_dict.items()
                if k.startswith("cls_model.")
            }
            if cls_state:
                base_model.cls_model.load_state_dict(cls_state, strict=True)
            else:
                raise
        else:
            raise

    base_model.to(device).eval()

    model = adapter_cls(base_model).to(device).eval()
    meta = {
        "model_script": script_name,
        "model_name": SUPPORTED_MODEL_SCRIPT,
        "n_classes": num_classes,
        "seg_pretrained": seg_pretrained,
        "seg_freeze": seg_freeze,
        "mask_mode": mask_mode,
        "organ_dim": 1,
        "preprocess_mode": "imagenet_norm",
        "backend": "pytorch",
    }
    return model, class_names, meta


def build_runtime_from_ckpt(
    ckpt: Dict[str, Any],
    device: torch.device,
    model_script: str,
    infer_backend: str,
    onnx_path: str,
    trt_engine_cache_dir: str,
    trt_fp16: bool,
    trt_strict: bool,
    trt_workspace_gb: int,
    trt_device_id: int,
):
    script_name = _normalize_model_script(model_script)
    if script_name != SUPPORTED_MODEL_SCRIPT:
        raise ValueError(
            f"Unsupported MODEL_SCRIPT='{model_script}'. "
            f"Only '{SUPPORTED_MODEL_SCRIPT}' is supported."
        )

    model, class_names, meta = build_model_from_ckpt(
        ckpt,
        device=device,
        model_script=script_name,
    )

    backend = (infer_backend or "pytorch").lower().strip()
    if backend == "pytorch":
        return model, class_names, meta

    if backend != "tensorrt":
        raise ValueError(f"Unsupported INFER_BACKEND='{infer_backend}'. Use 'pytorch' or 'tensorrt'.")

    if device.type != "cuda":
        raise RuntimeError("TensorRT backend requires DEVICE to be CUDA.")

    if not hasattr(model, "model"):
        raise RuntimeError("Internal error: hybrid runtime adapter missing base model")

    base_model = model.model
    classifier_module = _extract_classifier_module(base_model)
    if not os.path.isfile(onnx_path):
        export_image_model_to_onnx(classifier_module, out_path=onnx_path)

    runtime = EfficientSegformerTensorRTRuntime(
        base_model=base_model,
        onnx_path=onnx_path,
        engine_cache_dir=trt_engine_cache_dir,
        use_fp16=trt_fp16,
        strict=trt_strict,
        workspace_gb=trt_workspace_gb,
        device_id=trt_device_id,
    )

    runtime_meta = dict(meta)
    runtime_meta["backend"] = "tensorrt"
    runtime_meta["onnx_path"] = onnx_path
    runtime_meta["trt_engine_cache_dir"] = trt_engine_cache_dir
    runtime_meta["trt_fp16"] = trt_fp16
    runtime_meta["trt_strict"] = trt_strict
    runtime_meta["runtime"] = runtime.info.details
    return runtime, class_names, runtime_meta
