# Project Context

## Tong quan

- Repo: FastAPI inference service cho phan loai cay.
- 2 backend:
  - `pytorch`
  - `tensorrt` (ONNX Runtime + TensorRT EP)

## Entry points chinh

- `api_server.py`
- `app/main.py`

## Runtime flow

1. Doc `.env` tai `app/core/config.py`.
2. Resolve checkpoint theo `MODEL_SCRIPT`, `MODEL_DIR`, `MODEL`.
3. Load ckpt + build runtime qua `app/services/model_loader.py`.
4. Gan runtime cho router qua `app/api/routes.py::set_runtime`.

## Thu muc quan trong

- `app/`: API + runtime logic
- `script/`: infer adapters + train scripts
- `model/`: checkpoints
- `notebooks/`: benchmark notebooks
