import os
from typing import Dict, Any, List, Tuple
import torch

from app.services.onnx_export import export_model_to_onnx, export_image_model_to_onnx
from app.services.runtime_engine import TensorRTRuntime, EfficientSegformerTensorRTRuntime


def _resolve_model_class(model_script: str):
    script_name = (model_script or "organ_aware_switch_vit").strip().lower()

    if script_name == "organ_aware_switch_vit":
        try:
            from script.organ_aware_switch_vit import OrganAwareSwitchViT
        except Exception:
            from organ_aware_switch_vit import OrganAwareSwitchViT
        return OrganAwareSwitchViT

    if script_name == "efficientnetv2-segformer":
        from script.efficientnetv2_segformer_infer import (
            EffNetV2SegFormerClassifier,
            EfficientSegformerInferenceAdapter,
        )
        return EffNetV2SegFormerClassifier, EfficientSegformerInferenceAdapter

    if script_name == "efficientnetv2-mask2former":
        from script.efficientnetv2_mask2former_infer import (
            EffNetV2Mask2FormerClassifier,
            EfficientMask2FormerInferenceAdapter,
        )
        return EffNetV2Mask2FormerClassifier, EfficientMask2FormerInferenceAdapter

    if script_name == "mobilenetv3large-segformer":
        from script.mobilenetv3large_segformer_infer import (
            MobileNetV3LargeSegFormerClassifier,
            MobileNetSegFormerInferenceAdapter,
        )
        return MobileNetV3LargeSegFormerClassifier, MobileNetSegFormerInferenceAdapter

    if script_name == "mobilenetv3large-deeplabv3":
        from script.mobilenetv3large_deeplabv3_infer import (
            MobileNetV3DeepLabV3Classifier,
            MobileNetDeepLabV3InferenceAdapter,
        )
        return MobileNetV3DeepLabV3Classifier, MobileNetDeepLabV3InferenceAdapter

    raise ValueError(
        f"Unsupported MODEL_SCRIPT='{model_script}'. "
        "Use 'organ_aware_switch_vit', 'efficientnetv2-segformer', 'efficientnetv2-mask2former', 'mobilenetv3large-segformer', or 'mobilenetv3large-deeplabv3'."
    )


def load_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise ValueError("Invalid checkpoint format. Expect a dict checkpoint.")

    if "model_state_dict" in ckpt or "model_state" in ckpt:
        return ckpt

    # Some users save raw state_dict directly; allow downstream loader to report
    # missing metadata with a more specific error.
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
    script_name = (model_script or "organ_aware_switch_vit").strip().lower()

    if script_name == "efficientnetv2-segformer":
        return build_efficientnetv2_segformer_from_ckpt(ckpt=ckpt, device=device, model_script=model_script)
    if script_name == "efficientnetv2-mask2former":
        return build_efficientnetv2_mask2former_from_ckpt(ckpt=ckpt, device=device, model_script=model_script)
    if script_name == "mobilenetv3large-segformer":
        return build_mobilenetv3large_segformer_from_ckpt(ckpt=ckpt, device=device, model_script=model_script)
    if script_name == "mobilenetv3large-deeplabv3":
        return build_mobilenetv3large_deeplabv3_from_ckpt(ckpt=ckpt, device=device, model_script=model_script)

    saved_args = ckpt.get("args", {}) or {}
    class_to_idx = ckpt.get("class_to_idx")
    if not class_to_idx:
        raise ValueError("Checkpoint missing class_to_idx")

    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    n_classes = len(class_names)
    organ_dim = int(ckpt.get("organ_dim", saved_args.get("n_org_clusters", 5)))
    n_experts = int(ckpt.get("n_experts", saved_args.get("n_experts", 8)))

    vit_name = saved_args.get("vit_name", "vit_base_patch16_224")
    d_ff_expert = int(saved_args.get("d_ff_expert", 1024))
    top_k = int(saved_args.get("top_k", 1))

    model_cls = _resolve_model_class(model_script)
    if isinstance(model_cls, tuple):
        raise RuntimeError("Internal model loader mismatch for MODEL_SCRIPT=organ_aware_switch_vit")
    model = model_cls(
        vit_name=vit_name,
        n_classes=n_classes,
        organ_dim=organ_dim,
        n_experts=n_experts,
        d_ff_expert=d_ff_expert,
        top_k=top_k,
        pretrained=False,
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    meta = {
        "vit_name": vit_name,
        "n_classes": n_classes,
        "organ_dim": organ_dim,
        "n_experts": n_experts,
        "d_ff_expert": d_ff_expert,
        "router_top_k": top_k,
        "model_script": model_script,
        "backend": "pytorch",
    }
    return model, class_names, meta


def build_efficientnetv2_segformer_from_ckpt(
    ckpt: Dict[str, Any],
    device: torch.device,
    model_script: str,
) -> Tuple[torch.nn.Module, List[str], Dict[str, Any]]:
    saved_args = ckpt.get("args", {}) or {}

    species_list = ckpt.get("species_list")
    label_map = ckpt.get("label_map") or {}
    if species_list:
        class_names = list(species_list)
    elif label_map:
        idx_to_class = {int(v): k for k, v in label_map.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        raise ValueError("Checkpoint missing species_list/label_map for efficientnetv2-segformer")

    num_classes = len(class_names)
    model_cls, adapter_cls = _resolve_model_class("efficientnetv2-segformer")

    segformer_name = saved_args.get("segformer_name", "nvidia/segformer-b1-finetuned-ade-512-512")
    seg_input_size = int(saved_args.get("seg_input_size", 512))
    seg_threshold = float(saved_args.get("seg_threshold", saved_args.get("mask_score_threshold", 0.5)))
    seg_temperature = float(saved_args.get("seg_temperature", 12.0))
    min_keep_bg = float(saved_args.get("min_keep_bg", 0.15))

    base_model = model_cls(
        num_classes=num_classes,
        segformer_name=segformer_name,
        seg_input_size=seg_input_size,
        seg_threshold=seg_threshold,
        seg_temperature=seg_temperature,
        min_keep_bg=min_keep_bg,
        freeze_segformer=True,
    )

    state_dict = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing model_state/model_state_dict")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    try:
        base_model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        try:
            base_model.classifier.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            if not any(k.startswith("classifier.") for k in state_dict.keys()):
                raise
            classifier_state = {
                k[len("classifier.") :]: v
                for k, v in state_dict.items()
                if k.startswith("classifier.")
            }
            if not classifier_state:
                raise
            base_model.classifier.load_state_dict(classifier_state, strict=True)

    base_model.to(device).eval()

    model = adapter_cls(base_model).to(device).eval()
    meta = {
        "model_script": model_script,
        "model_name": "efficientnetv2s-segformerb4",
        "n_classes": num_classes,
        "segformer_name": segformer_name,
        "seg_input_size": seg_input_size,
        "seg_threshold": seg_threshold,
        "seg_temperature": seg_temperature,
        "min_keep_bg": min_keep_bg,
        "organ_dim": 1,
        "preprocess_mode": "raw_01",
        "backend": "pytorch",
    }
    return model, class_names, meta


def build_efficientnetv2_mask2former_from_ckpt(
    ckpt: Dict[str, Any],
    device: torch.device,
    model_script: str,
) -> Tuple[torch.nn.Module, List[str], Dict[str, Any]]:
    saved_args = ckpt.get("args", {}) or {}

    species_list = ckpt.get("species_list")
    label_map = ckpt.get("label_map") or {}
    if species_list:
        class_names = list(species_list)
    elif label_map:
        idx_to_class = {int(v): k for k, v in label_map.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        raise ValueError("Checkpoint missing species_list/label_map for efficientnetv2-mask2former")

    num_classes = len(class_names)
    model_cls, adapter_cls = _resolve_model_class("efficientnetv2-mask2former")

    mask2former_model_id = saved_args.get(
        "mask2former_model_id",
        "facebook/mask2former-swin-small-coco-panoptic",
    )
    mask_score_threshold = float(saved_args.get("mask_score_threshold", 0.35))

    base_model = model_cls(
        num_classes=num_classes,
        mask2former_model_id=mask2former_model_id,
        mask_score_threshold=mask_score_threshold,
        freeze_mask2former=True,
    )

    state_dict = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing model_state/model_state_dict")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    try:
        base_model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        try:
            base_model.classifier.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            if not any(k.startswith("classifier.") for k in state_dict.keys()):
                raise
            classifier_state = {
                k[len("classifier.") :]: v
                for k, v in state_dict.items()
                if k.startswith("classifier.")
            }
            if not classifier_state:
                raise
            base_model.classifier.load_state_dict(classifier_state, strict=True)

    base_model.to(device).eval()

    model = adapter_cls(base_model).to(device).eval()
    meta = {
        "model_script": model_script,
        "model_name": "efficientnetv2s-mask2former",
        "n_classes": num_classes,
        "mask2former_model_id": mask2former_model_id,
        "mask_score_threshold": mask_score_threshold,
        "organ_dim": 1,
        "preprocess_mode": "raw_01",
        "backend": "pytorch",
    }
    return model, class_names, meta


def build_mobilenetv3large_segformer_from_ckpt(
    ckpt: Dict[str, Any],
    device: torch.device,
    model_script: str,
) -> Tuple[torch.nn.Module, List[str], Dict[str, Any]]:
    saved_args = ckpt.get("args", {}) or {}

    species_list = ckpt.get("species_list")
    label_map = ckpt.get("label_map") or {}
    if species_list:
        class_names = list(species_list)
    elif label_map:
        idx_to_class = {int(v): k for k, v in label_map.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        raise ValueError("Checkpoint missing species_list/label_map for mobilenetv3large-segformer")

    num_classes = len(class_names)
    model_cls, adapter_cls = _resolve_model_class("mobilenetv3large-segformer")

    segformer_name = saved_args.get("segformer_name", "nvidia/segformer-b4-finetuned-ade-512-512")
    mask_blend = float(saved_args.get("mask_blend", 1.0))
    background_keep = float(saved_args.get("background_keep", 0.15))

    base_model = model_cls(
        num_classes=num_classes,
        segformer_name=segformer_name,
        mask_blend=mask_blend,
        background_keep=background_keep,
        freeze_segformer=True,
    )

    state_dict = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing model_state/model_state_dict")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    try:
        base_model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        try:
            base_model.classifier.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            if not any(k.startswith("classifier.") for k in state_dict.keys()):
                raise
            classifier_state = {
                k[len("classifier.") :]: v
                for k, v in state_dict.items()
                if k.startswith("classifier.")
            }
            if not classifier_state:
                raise
            base_model.classifier.load_state_dict(classifier_state, strict=True)

    base_model.to(device).eval()

    model = adapter_cls(base_model).to(device).eval()
    meta = {
        "model_script": model_script,
        "model_name": "mobilenetv3large-segformerb4",
        "n_classes": num_classes,
        "segformer_name": segformer_name,
        "mask_blend": mask_blend,
        "background_keep": background_keep,
        "organ_dim": 1,
        "preprocess_mode": "raw_01",
        "backend": "pytorch",
    }
    return model, class_names, meta


def build_mobilenetv3large_deeplabv3_from_ckpt(
    ckpt: Dict[str, Any],
    device: torch.device,
    model_script: str,
) -> Tuple[torch.nn.Module, List[str], Dict[str, Any]]:
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
    model_cls, adapter_cls = _resolve_model_class("mobilenetv3large-deeplabv3")

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

    # Training script uses `cls_model.*` while API runtime uses `classifier.*`.
    if any(k.startswith("cls_model.") for k in state_dict.keys()):
        state_dict = {
            ("classifier." + k[len("cls_model.") :]) if k.startswith("cls_model.") else k: v
            for k, v in state_dict.items()
        }

    try:
        base_model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        classifier_state = {
            k[len("classifier.") :]: v
            for k, v in state_dict.items()
            if k.startswith("classifier.")
        }
        if not classifier_state:
            raise
        base_model.classifier.load_state_dict(classifier_state, strict=True)

    base_model.to(device).eval()

    model = adapter_cls(base_model).to(device).eval()
    meta = {
        "model_script": model_script,
        "model_name": "mobilenetv3large-deeplabv3",
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
    model, class_names, meta = build_model_from_ckpt(
        ckpt,
        device=device,
        model_script=model_script,
    )
    backend = (infer_backend or "pytorch").lower().strip()

    if backend == "pytorch":
        return model, class_names, meta

    if (model_script or "").strip().lower() in {
        "efficientnetv2-segformer",
        "efficientnetv2-mask2former",
        "mobilenetv3large-segformer",
        "mobilenetv3large-deeplabv3",
    }:
        if device.type != "cuda":
            raise RuntimeError("TensorRT backend requires DEVICE to be CUDA.")

        if not hasattr(model, "model"):
            raise RuntimeError("Internal error: hybrid runtime adapter missing base model")

        base_model = model.model
        if not os.path.isfile(onnx_path):
            export_image_model_to_onnx(base_model.classifier, out_path=onnx_path)

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

    if backend != "tensorrt":
        raise ValueError(f"Unsupported INFER_BACKEND='{infer_backend}'. Use 'pytorch' or 'tensorrt'.")

    if device.type != "cuda":
        raise RuntimeError("TensorRT backend requires DEVICE to be CUDA.")

    if not os.path.isfile(onnx_path):
        export_model_to_onnx(model, organ_dim=meta["organ_dim"], out_path=onnx_path)

    runtime = TensorRTRuntime(
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
