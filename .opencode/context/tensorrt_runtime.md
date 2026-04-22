# TensorRT Runtime Context

## Rule phan backend

- `INFER_BACKEND=pytorch`: su dung model `.pt` truc tiep, khong can ONNX runtime.
- `INFER_BACKEND=tensorrt`: export ONNX (neu chua co), tao ORT session TensorRT EP.

## Cache isolation (quan trong)

- `ONNX_PATH` va `TRT_ENGINE_CACHE_DIR` mac dinh duoc tao theo `checkpoint stem`.
- Muc tieu: tranh dung lan engine khi doi qua lai 384/512.

Nguon tham chieu: `app/core/config.py`.

## Hybrid TensorRT

- Cac model segmentation-guided chay hybrid:
  - segmentation branch: PyTorch
  - classifier branch: TensorRT

Nguon tham chieu: `app/services/runtime_engine.py`, `app/services/model_loader.py`.
