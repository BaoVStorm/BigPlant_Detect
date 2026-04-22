# Skill: Add New Model Script

## Muc tieu

Tich hop model script moi vao he thong inference cho ca `pytorch` va `tensorrt`.

## Checklist

1. Tao infer adapter moi trong `script/*_infer.py`.
2. Dang ky script trong `SUPPORTED_MODEL_SCRIPTS` (`app/services/model_loader.py`).
3. Them nhanh resolve class trong `_resolve_model_class(...)`.
4. Parse `ckpt["args"]` an toan trong `build_model_from_ckpt(...)`.
5. Set `meta["preprocess_mode"]` dung theo training pipeline.
6. Dam bao export ONNX lay dung classifier module.
7. Bo sung candidate checkpoint trong `app/core/config.py`.
8. Cap nhat `.env.example` voi `MODEL_SCRIPT` moi.
9. Khong thay doi endpoint API.

## Verify toi thieu

- `GET /health` tra ve backend + model meta dung.
- Chay 1 anh voi `pytorch` va `tensorrt` deu tra ket qua hop le.
