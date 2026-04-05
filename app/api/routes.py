import io
import time
import base64
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image

from app.core.config import DEFAULT_TOPK
from app.services.preprocessing import preprocess_pil
from app.services.predictor import predict_one


router = APIRouter()

# These will be set by init in main.py
MODEL = None
CLASS_NAMES = None
META = None
DEVICE = None


def set_runtime(model, class_names, meta, device):
    global MODEL, CLASS_NAMES, META, DEVICE
    MODEL = model
    CLASS_NAMES = class_names
    META = meta
    DEVICE = device


def _preprocess_mode() -> str:
    if not isinstance(META, dict):
        return "imagenet_norm"
    return str(META.get("preprocess_mode", "imagenet_norm"))


@router.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model": META}


@router.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
    topk: int = DEFAULT_TOPK,
    two_pass: bool = True,
):
    t_total0 = time.perf_counter()
    data = await file.read()

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    t_pre0 = time.perf_counter()
    x = preprocess_pil(img, device=DEVICE, mode=_preprocess_mode())
    t_pre1 = time.perf_counter()

    out = predict_one(
        MODEL,
        x,
        CLASS_NAMES,
        organ_dim=META["organ_dim"],
        topk=topk,
        two_pass=two_pass,
    )

    t_total1 = time.perf_counter()

    out["timing_ms"]["preprocess_ms"] = (t_pre1 - t_pre0) * 1000.0
    out["timing_ms"]["total_ms"] = (t_total1 - t_total0) * 1000.0
    out["input"] = {"filename": file.filename, "content_type": file.content_type}
    out["meta"] = {"device": str(DEVICE), "model": META}
    return out


class PredictB64Request(BaseModel):
    image_b64: str
    topk: Optional[int] = DEFAULT_TOPK
    two_pass: Optional[bool] = True


@router.post("/predict/base64")
def predict_base64(req: PredictB64Request):
    t_total0 = time.perf_counter()
    try:
        raw = base64.b64decode(req.image_b64, validate=True)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    t_pre0 = time.perf_counter()
    x = preprocess_pil(img, device=DEVICE, mode=_preprocess_mode())
    t_pre1 = time.perf_counter()

    out = predict_one(
        MODEL,
        x,
        CLASS_NAMES,
        organ_dim=META["organ_dim"],
        topk=req.topk or DEFAULT_TOPK,
        two_pass=bool(req.two_pass),
    )

    t_total1 = time.perf_counter()
    out["timing_ms"]["preprocess_ms"] = (t_pre1 - t_pre0) * 1000.0
    out["timing_ms"]["total_ms"] = (t_total1 - t_total0) * 1000.0
    out["meta"] = {"device": str(DEVICE), "model": META}
    return out
