import os
from fastapi import FastAPI

from app.core.config import (
    CHECKPOINT_PATH,
    DEVICE,
    MODEL_SCRIPT,
    INFER_BACKEND,
    ONNX_PATH,
    TRT_ENGINE_CACHE_DIR,
    TRT_DEVICE_ID,
    TRT_FP16,
    TRT_STRICT,
    TRT_WORKSPACE_GB,
)
from app.services.device import choose_device
from app.services.model_loader import load_checkpoint, build_runtime_from_ckpt
from app.api.routes import router, set_runtime


def create_app() -> FastAPI:
    app = FastAPI(title="Organ-Aware Switch-ViT API", version="1.0")

    device = choose_device(DEVICE)

    if not os.path.isfile(CHECKPOINT_PATH):
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH} (set env CKPT=...)")

    ckpt = load_checkpoint(CHECKPOINT_PATH)
    model, class_names, meta = build_runtime_from_ckpt(
        ckpt,
        device=device,
        model_script=MODEL_SCRIPT,
        infer_backend=INFER_BACKEND,
        onnx_path=ONNX_PATH,
        trt_engine_cache_dir=TRT_ENGINE_CACHE_DIR,
        trt_fp16=TRT_FP16,
        trt_strict=TRT_STRICT,
        trt_workspace_gb=TRT_WORKSPACE_GB,
        trt_device_id=TRT_DEVICE_ID,
    )

    set_runtime(model, class_names, meta, device)
    app.include_router(router)

    return app


app = create_app()

#!/usr/bin/env python3
# pip install fastapi uvicorn python-multipart pillow torch torchvision timm
# Run:
#   uvicorn api_server:app --host 0.0.0.0 --port 8000

# run
# uvicorn api_server:app --host 0.0.0.0 --port 8000
# python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

# local
# D:\App\anaconda3\envs\pt_gpu\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
