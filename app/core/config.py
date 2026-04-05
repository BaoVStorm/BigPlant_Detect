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

# Model script selector: organ_aware_switch_vit|efficientnetv2-segformer
MODEL_SCRIPT = os.getenv("MODEL_SCRIPT", "organ_aware_switch_vit").strip()

MODEL_DIR = os.getenv("MODEL_DIR", "./model").strip()
MODEL_SCRIPT_DIR = os.path.join(MODEL_DIR, MODEL_SCRIPT)


def _resolve_model_path(model_script_dir: str) -> str:
    """
    Resolve model checkpoint path without CKPT env.

    Priority:
    1) MODEL env (file name inside model_script_dir or absolute path)
    2) best_model.pt inside model_script_dir
    3) <MODEL_SCRIPT>.pt inside model_script_dir
    """
    model_env = os.getenv("MODEL", "").strip().strip('"')
    if model_env:
        if os.path.isabs(model_env):
            return model_env
        return os.path.join(model_script_dir, model_env)

    best_path = os.path.join(model_script_dir, "best_model.pt")
    if os.path.isfile(best_path):
        return best_path

    script_name = (MODEL_SCRIPT or "").strip().lower()
    script_named_path = os.path.join(model_script_dir, f"{script_name}.pt")
    if os.path.isfile(script_named_path):
        return script_named_path

    return best_path


def _default_onnx_path(model_script: str, model_script_dir: str) -> str:
    script_name = (model_script or "organ_aware_switch_vit").strip().lower()
    return os.path.join(model_script_dir, f"{script_name}.onnx")


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


CHECKPOINT_PATH = _resolve_model_path(MODEL_SCRIPT_DIR)

# Inference backend: pytorch|tensorrt
INFER_BACKEND = os.getenv("INFER_BACKEND", "pytorch").strip().lower()

# TensorRT/ONNX settings
ONNX_PATH = _env_or_default("ONNX_PATH", _default_onnx_path(MODEL_SCRIPT, MODEL_SCRIPT_DIR))
TRT_ENGINE_CACHE_DIR = _env_or_default(
    "TRT_ENGINE_CACHE_DIR",
    os.path.join(MODEL_SCRIPT_DIR, f"{MODEL_SCRIPT.strip().lower()}_trt_cache"),
)
TRT_DEVICE_ID = int(os.getenv("TRT_DEVICE_ID", "0"))
TRT_FP16 = os.getenv("TRT_FP16", "1") in {"1", "true", "True", "yes", "on"}
TRT_STRICT = os.getenv("TRT_STRICT", "1") in {"1", "true", "True", "yes", "on"}
TRT_WORKSPACE_GB = int(os.getenv("TRT_WORKSPACE_GB", "4"))

# DEVICE: auto|cpu|cuda|cuda:0
DEVICE = os.getenv("DEVICE", "auto")

# TOPK default
DEFAULT_TOPK = int(os.getenv("TOPK", "5"))
