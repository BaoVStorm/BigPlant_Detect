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

# Supported: mobilenetv3large-deeplabv3 | mobilenetv3large-mask2former | resnet50-segformer
MODEL_SCRIPT = os.getenv("MODEL_SCRIPT", "mobilenetv3large-deeplabv3").strip()

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
        script_local = os.path.join(model_script_dir, model_env)
        if os.path.isfile(script_local):
            return script_local
        model_root_local = os.path.join(MODEL_DIR, model_env)
        if os.path.isfile(model_root_local):
            return model_root_local
        return script_local

    best_path = os.path.join(model_script_dir, "best_model.pt")
    if os.path.isfile(best_path):
        return best_path

    script_name = (MODEL_SCRIPT or "").strip().lower()
    script_named_path = os.path.join(model_script_dir, f"{script_name}.pt")
    if os.path.isfile(script_named_path):
        return script_named_path

    if script_name == "mobilenetv3large-mask2former":
        candidates = [
            os.path.join(model_script_dir, "mobilenetv3large-mask2former.pt"),
            os.path.join(model_script_dir, "mobilenetv3large-mask2former-512x512.pt"),
            os.path.join(model_script_dir, "mobilenetv3large-mask2former-384x384.pt"),
            os.path.join(MODEL_DIR, "mobilenetv3large-mask2former.pt"),
            os.path.join(MODEL_DIR, "mobilenetv3large-mask2former-512x512.pt"),
            os.path.join(MODEL_DIR, "mobilenetv3large-mask2former-384x384.pt"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
    elif script_name == "mobilenetv3large-deeplabv3":
        candidates = [
            os.path.join(model_script_dir, "mobilenetv3large-deeplabv3.pt"),
            os.path.join(model_script_dir, "mobilenetv3large-deeplabv3-512x512.pt"),
            os.path.join(model_script_dir, "mobilenetv3large-deeplabv3-384x384.pt"),
            os.path.join(MODEL_DIR, "mobilenetv3large-deeplabv3.pt"),
            os.path.join(MODEL_DIR, "mobilenetv3large-deeplabv3-512x512.pt"),
            os.path.join(MODEL_DIR, "mobilenetv3large-deeplabv3-384x384.pt"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
    elif script_name == "resnet50-segformer":
        candidates = [
            os.path.join(model_script_dir, "resnet50-segformer.pt"),
            os.path.join(model_script_dir, "resnet50-segformer-512x512.pt"),
            os.path.join(model_script_dir, "resnet50-segformer-384x384.pt"),
            os.path.join(MODEL_DIR, "resnet50-segformer.pt"),
            os.path.join(MODEL_DIR, "resnet50-segformer-512x512.pt"),
            os.path.join(MODEL_DIR, "resnet50-segformer-384x384.pt"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
    else:
        raise ValueError(
            f"Unsupported MODEL_SCRIPT='{MODEL_SCRIPT}'. "
            "Use 'mobilenetv3large-deeplabv3', 'mobilenetv3large-mask2former', or 'resnet50-segformer'."
        )

    return best_path


def _checkpoint_stem(ckpt_path: str, fallback_model_script: str) -> str:
    ckpt_name = os.path.splitext(os.path.basename((ckpt_path or "").strip()))[0].strip()
    if ckpt_name:
        return ckpt_name
    return (fallback_model_script or "mobilenetv3large-deeplabv3").strip().lower()


def _default_onnx_path(model_script_dir: str, ckpt_path: str, fallback_model_script: str) -> str:
    stem = _checkpoint_stem(ckpt_path, fallback_model_script)
    return os.path.join(model_script_dir, f"{stem}.onnx")


def _default_trt_cache_dir(model_script_dir: str, ckpt_path: str, fallback_model_script: str) -> str:
    stem = _checkpoint_stem(ckpt_path, fallback_model_script)
    return os.path.join(model_script_dir, f"{stem}_trt_cache")


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
ONNX_PATH = _env_or_default(
    "ONNX_PATH",
    _default_onnx_path(MODEL_SCRIPT_DIR, CHECKPOINT_PATH, MODEL_SCRIPT),
)
TRT_ENGINE_CACHE_DIR = _env_or_default(
    "TRT_ENGINE_CACHE_DIR",
    _default_trt_cache_dir(MODEL_SCRIPT_DIR, CHECKPOINT_PATH, MODEL_SCRIPT),
)
TRT_DEVICE_ID = int(os.getenv("TRT_DEVICE_ID", "0"))
TRT_FP16 = os.getenv("TRT_FP16", "1") in {"1", "true", "True", "yes", "on"}
TRT_STRICT = os.getenv("TRT_STRICT", "1") in {"1", "true", "True", "yes", "on"}
TRT_WORKSPACE_GB = int(os.getenv("TRT_WORKSPACE_GB", "4"))

# DEVICE: auto|cpu|cuda|cuda:0
DEVICE = os.getenv("DEVICE", "auto")

# TOPK default
DEFAULT_TOPK = int(os.getenv("TOPK", "5"))
