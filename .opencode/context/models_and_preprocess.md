# Models and Preprocess

## Supported MODEL_SCRIPT

- `mobilenetv3large-deeplabv3`
- `mobilenetv3large-mask2former`
- `mobilenetv3large-segformer`
- `resnet50-deeplabv3`
- `resnet50-mask2former`
- `resnet50-segformer`

Nguon tham chieu: `app/services/model_loader.py`.

## Preprocess mode theo model

- `imagenet_norm`:
  - `mobilenetv3large-deeplabv3`
  - `mobilenetv3large-segformer`
  - `resnet50-deeplabv3`
  - `resnet50-segformer`
- `raw_01`:
  - `mobilenetv3large-mask2former`
  - `resnet50-mask2former`

Nguon tham chieu: `meta["preprocess_mode"]` trong `build_model_from_ckpt(...)`.

## Resolve checkpoint

Logic auto-resolve o `app/core/config.py` theo uu tien:

1. `MODEL` (absolute hoac relative)
2. `best_model.pt`
3. `<MODEL_SCRIPT>.pt`
4. Danh sach candidate rieng cho tung model script
