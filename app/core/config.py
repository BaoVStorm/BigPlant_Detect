import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_env_file_fallback(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


if load_dotenv is not None:
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
else:
    _load_env_file_fallback(_ENV_PATH)

# CHECKPOINT_PATH can be overridden by env CKPT
CHECKPOINT_PATH = os.getenv("CKPT", "./model/best_model.pt")

# Model script selector: organ_aware_switch_vit|efficientnetv2-segformer
MODEL_SCRIPT = os.getenv("MODEL_SCRIPT", "organ_aware_switch_vit").strip()

# Inference backend: pytorch|tensorrt
INFER_BACKEND = os.getenv("INFER_BACKEND", "pytorch").strip().lower()

# TensorRT/ONNX settings
ONNX_PATH = os.getenv("ONNX_PATH", "./model/best_model.onnx")
TRT_ENGINE_CACHE_DIR = os.getenv("TRT_ENGINE_CACHE_DIR", "./model/trt_cache")
TRT_DEVICE_ID = int(os.getenv("TRT_DEVICE_ID", "0"))
TRT_FP16 = os.getenv("TRT_FP16", "1") in {"1", "true", "True", "yes", "on"}
TRT_STRICT = os.getenv("TRT_STRICT", "1") in {"1", "true", "True", "yes", "on"}
TRT_WORKSPACE_GB = int(os.getenv("TRT_WORKSPACE_GB", "4"))

# DEVICE: auto|cpu|cuda|cuda:0
DEVICE = os.getenv("DEVICE", "auto")

# TOPK default
DEFAULT_TOPK = int(os.getenv("TOPK", "5"))
