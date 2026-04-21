import os
from typing import Any, Dict, List, Tuple

import torch

from app.services.onnx_export import export_image_model_to_onnx
from app.services.runtime_engine import EfficientSegformerTensorRTRuntime


SUPPORTED_MODEL_SCRIPTS = {
    "mobilenetv3large-deeplabv3",
    "mobilenetv3large-mask2former",
    "resnet50-segformer",
    "resnet50-deeplabv3",
    "resnet50-mask2former",
}


def _normalize_model_script(model_script: str) -> str:
    return (model_script or "").strip().lower()


def _ensure_supported_model_script(model_script: str) -> str:
    script_name = _normalize_model_script(model_script)
    if script_name not in SUPPORTED_MODEL_SCRIPTS:
        raise ValueError(
            f"Unsupported MODEL_SCRIPT='{model_script}'. "
            "Use 'mobilenetv3large-deeplabv3', 'mobilenetv3large-mask2former', 'resnet50-segformer', 'resnet50-deeplabv3', or 'resnet50-mask2former'."
        )
    return script_name


def _resolve_model_class(model_script: str):
    script_name = _ensure_supported_model_script(model_script)

    if script_name == "mobilenetv3large-deeplabv3":
        from script.mobilenetv3large_deeplabv3_infer import (
            MobileNetDeepLabV3InferenceAdapter,
            MobileNetV3DeepLabV3Classifier,
        )

        return MobileNetV3DeepLabV3Classifier, MobileNetDeepLabV3InferenceAdapter

    if script_name == "resnet50-segformer":
        from script.resnet50_segformer_infer import (
            ResNet50SegFormerClassifier,
            ResNet50SegFormerInferenceAdapter,
        )

        return ResNet50SegFormerClassifier, ResNet50SegFormerInferenceAdapter

    if script_name == "resnet50-deeplabv3":
        from script.resnet50_deeplabv3_infer import (
            ResNet50DeepLabV3Classifier,
            ResNet50DeepLabV3InferenceAdapter,
        )

        return ResNet50DeepLabV3Classifier, ResNet50DeepLabV3InferenceAdapter

    if script_name == "resnet50-mask2former":
        from script.resnet50_mask2former_infer import (
            ResNet50Mask2FormerClassifier,
            ResNet50Mask2FormerInferenceAdapter,
        )

        return ResNet50Mask2FormerClassifier, ResNet50Mask2FormerInferenceAdapter

    from script.mobilenetv3large_mask2former_infer import (
        MobileNetMask2FormerInferenceAdapter,
        MobileNetV3Mask2FormerClassifier,
    )

    return MobileNetV3Mask2FormerClassifier, MobileNetMask2FormerInferenceAdapter


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


def _class_names_from_ckpt(ckpt: Dict[str, Any], model_script: str) -> List[str]:
    species_list = ckpt.get("species_list")
    label_map = ckpt.get("label_map") or {}
    if species_list:
        return list(species_list)
    if label_map:
        idx_to_class = {int(v): k for k, v in label_map.items()}
        return [idx_to_class[i] for i in range(len(idx_to_class))]
    raise ValueError(f"Checkpoint missing species_list/label_map for {model_script}")


def _load_state_dict(base_model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    if any(k.startswith("cls_model.") for k in state_dict.keys()) and hasattr(base_model, "classifier") and not hasattr(base_model, "cls_model"):
        state_dict = {
            ("classifier." + k[len("cls_model.") :]) if k.startswith("cls_model.") else k: v
            for k, v in state_dict.items()
        }

    try:
        base_model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    loaded_partial = False
    if hasattr(base_model, "classifier"):
        classifier_state = {
            k[len("classifier.") :]: v
            for k, v in state_dict.items()
            if k.startswith("classifier.")
        }
        if classifier_state:
            base_model.classifier.load_state_dict(classifier_state, strict=True)
            loaded_partial = True

    if hasattr(base_model, "cls_model"):
        cls_state = {
            k[len("cls_model.") :]: v
            for k, v in state_dict.items()
            if k.startswith("cls_model.")
        }
        if cls_state:
            base_model.cls_model.load_state_dict(cls_state, strict=True)
            loaded_partial = True

    if not loaded_partial:
        base_model.load_state_dict(state_dict, strict=True)


def build_model_from_ckpt(
    ckpt: Dict[str, Any],
    device: torch.device,
    model_script: str,
) -> Tuple[torch.nn.Module, List[str], Dict[str, Any]]:
    script_name = _ensure_supported_model_script(model_script)
    saved_args = ckpt.get("args", {}) or {}
    class_names = _class_names_from_ckpt(ckpt, script_name)

    num_classes = len(class_names)
    model_cls, adapter_cls = _resolve_model_class(script_name)

    if script_name == "mobilenetv3large-deeplabv3":
        base_model = model_cls(
            num_classes=num_classes,
            seg_pretrained=bool(saved_args.get("seg_pretrained", True)),
            seg_freeze=bool(saved_args.get("seg_freeze", True)),
            mask_mode=str(saved_args.get("mask_mode", "attention")),
        )
        preprocess_mode = "imagenet_norm"
        meta_extra = {
            "seg_pretrained": bool(saved_args.get("seg_pretrained", True)),
            "seg_freeze": bool(saved_args.get("seg_freeze", True)),
            "mask_mode": str(saved_args.get("mask_mode", "attention")),
        }
    elif script_name == "resnet50-segformer":
        unfreeze_segformer = bool(saved_args.get("unfreeze_segformer", False))
        freeze_segformer = not unfreeze_segformer
        base_model = model_cls(
            num_classes=num_classes,
            segformer_name=str(
                saved_args.get(
                    "segformer_name",
                    "nvidia/segformer-b4-finetuned-ade-512-512",
                )
            ),
            freeze_segformer=freeze_segformer,
            mask_blend=float(saved_args.get("mask_blend", 1.0)),
            background_keep=float(saved_args.get("background_keep", 0.15)),
        )
        preprocess_mode = "imagenet_norm"
        meta_extra = {
            "segformer_name": str(
                saved_args.get(
                    "segformer_name",
                    "nvidia/segformer-b4-finetuned-ade-512-512",
                )
            ),
            "freeze_segformer": freeze_segformer,
            "mask_blend": float(saved_args.get("mask_blend", 1.0)),
            "background_keep": float(saved_args.get("background_keep", 0.15)),
        }
    elif script_name == "resnet50-deeplabv3":
        base_model = model_cls(
            num_classes=num_classes,
            seg_pretrained=bool(saved_args.get("seg_pretrained", False)),
            seg_freeze=bool(saved_args.get("seg_freeze", False)),
            mask_mode=str(saved_args.get("mask_mode", "attention")),
        )
        preprocess_mode = "imagenet_norm"
        meta_extra = {
            "seg_pretrained": bool(saved_args.get("seg_pretrained", False)),
            "seg_freeze": bool(saved_args.get("seg_freeze", False)),
            "mask_mode": str(saved_args.get("mask_mode", "attention")),
        }
    elif script_name == "resnet50-mask2former":
        unfreeze_seg = bool(saved_args.get("unfreeze_segmentation", False))
        freeze_segmentation = not unfreeze_seg
        base_model = model_cls(
            num_classes=num_classes,
            mask2former_name=str(
                saved_args.get(
                    "mask2former_name",
                    "facebook/mask2former-swin-large-ade-semantic",
                )
            ),
            seg_input_size=int(saved_args.get("seg_input_size", 384)),
            mask_floor=float(saved_args.get("mask_floor", 0.15)),
            freeze_segmentation=freeze_segmentation,
        )
        preprocess_mode = "raw_01"
        meta_extra = {
            "mask2former_name": str(
                saved_args.get(
                    "mask2former_name",
                    "facebook/mask2former-swin-large-ade-semantic",
                )
            ),
            "seg_input_size": int(saved_args.get("seg_input_size", 384)),
            "mask_floor": float(saved_args.get("mask_floor", 0.15)),
            "freeze_segmentation": freeze_segmentation,
        }
    else:
        unfreeze_seg = bool(saved_args.get("unfreeze_segmentation", False))
        freeze_segmentation = not unfreeze_seg
        base_model = model_cls(
            num_classes=num_classes,
            mask2former_name=str(
                saved_args.get(
                    "mask2former_name",
                    "facebook/mask2former-swin-large-ade-semantic",
                )
            ),
            seg_input_size=int(saved_args.get("seg_input_size", 384)),
            mask_floor=float(saved_args.get("mask_floor", 0.15)),
            freeze_segmentation=freeze_segmentation,
        )
        preprocess_mode = "raw_01"
        meta_extra = {
            "mask2former_name": str(
                saved_args.get(
                    "mask2former_name",
                    "facebook/mask2former-swin-large-ade-semantic",
                )
            ),
            "seg_input_size": int(saved_args.get("seg_input_size", 384)),
            "mask_floor": float(saved_args.get("mask_floor", 0.15)),
            "freeze_segmentation": freeze_segmentation,
        }

    state_dict = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing model_state/model_state_dict")
    _load_state_dict(base_model, state_dict)

    base_model.to(device).eval()
    model = adapter_cls(base_model).to(device).eval()

    meta = {
        "model_script": script_name,
        "model_name": script_name,
        "n_classes": num_classes,
        "organ_dim": 1,
        "preprocess_mode": preprocess_mode,
        "backend": "pytorch",
    }
    meta.update(meta_extra)
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
    script_name = _ensure_supported_model_script(model_script)
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
