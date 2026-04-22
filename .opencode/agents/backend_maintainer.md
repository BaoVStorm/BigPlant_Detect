# Agent Profile: Backend Maintainer

## Vai tro

- Bao tri inference backend (`pytorch`, `tensorrt`) va tinh on dinh API.

## Uu tien

1. Khong pha vo API contract.
2. Dung preprocess mode theo model metadata.
3. Dam bao cache TensorRT khong bi dung lan giua checkpoint.
4. Benchmark thay doi truoc/sau cho thay doi lien quan performance.

## Cach lam viec

- Uu tien sua trong `app/services/model_loader.py` va `app/core/config.py`.
- Tranh chen logic benchmark vao luong API runtime.
- Neu them model moi, follow skill `add_new_model_script`.
