import os
from typing import Dict, Any, List, Tuple
import torch

from app.services.onnx_export import export_model_to_onnx
from app.services.runtime_engine import TensorRTRuntime

# Import your model (same as your current)
from organ_aware_switch_vit import OrganAwareSwitchViT


def load_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Invalid checkpoint format. Expect dict with model_state_dict.")
    return ckpt


def build_model_from_ckpt(
    ckpt: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, List[str], Dict[str, Any]]:
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

    model = OrganAwareSwitchViT(
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
        "backend": "pytorch",
    }
    return model, class_names, meta


def build_runtime_from_ckpt(
    ckpt: Dict[str, Any],
    device: torch.device,
    infer_backend: str,
    onnx_path: str,
    trt_engine_cache_dir: str,
    trt_fp16: bool,
    trt_strict: bool,
    trt_workspace_gb: int,
    trt_device_id: int,
):
    model, class_names, meta = build_model_from_ckpt(ckpt, device=device)
    backend = (infer_backend or "pytorch").lower().strip()

    if backend == "pytorch":
        return model, class_names, meta

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
