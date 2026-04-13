# TensorRT Setup Guide (for this API)

This project now supports two backends with the same API routes and port usage:

- `INFER_BACKEND=pytorch` (default)
- `INFER_BACKEND=tensorrt`

Your endpoints remain unchanged:

- `GET /health`
- `POST /predict/file`
- `POST /predict/base64`

And your run command can stay the same:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 1) Prerequisites

1. NVIDIA GPU + recent driver (`nvidia-smi` must work).
2. CUDA-compatible Python environment.
3. Install package dependencies:

```bash
pip install -r requirements.txt
```

4. TensorRT runtime must be installed on machine.
5. `onnxruntime-gpu` must expose `TensorrtExecutionProvider`.

Quick check:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You should see `TensorrtExecutionProvider` in the output.

## 2) Environment variables

Required/common:

- `MODEL_SCRIPT` chọn model script: `organ_aware_switch_vit` | `efficientnetv2-segformer` | `efficientnetv2-mask2former` | `mobilenetv3large-segformer` | `mobilenetv3large-deeplabv3`
- `MODEL_DIR` thư mục gốc model (mặc định `./model`)
- `MODEL` (tuỳ chọn) tên file `.pt` trong `./model/<MODEL_SCRIPT>/`
- `DEVICE` should be `cuda` or `cuda:0` for TensorRT
- `INFER_BACKEND=tensorrt`

TensorRT-specific:

- `ONNX_PATH` output ONNX path (default `./model/<MODEL_SCRIPT>/<MODEL_SCRIPT>.onnx`)
- `TRT_ENGINE_CACHE_DIR` engine cache folder (default `./model/<MODEL_SCRIPT>/<MODEL_SCRIPT>_trt_cache`)
- `TRT_DEVICE_ID` GPU index (default `0`)
- `TRT_FP16` enable fp16 (default `1`)
- `TRT_STRICT` strict TensorRT-only provider, no fallback (default `1`)
- `TRT_WORKSPACE_GB` TensorRT workspace GB (default `4`)

Example (Windows PowerShell):

```powershell
$env:MODEL_SCRIPT="organ_aware_switch_vit"
$env:MODEL_DIR="D:\Homework\BackEnd\ModelDetectApi\model"
$env:MODEL="best_model.pt"
$env:DEVICE="cuda:0"
$env:INFER_BACKEND="tensorrt"
$env:ONNX_PATH="D:\Homework\BackEnd\ModelDetectApi\model\organ_aware_switch_vit\organ_aware_switch_vit.onnx"
$env:TRT_ENGINE_CACHE_DIR="D:\Homework\BackEnd\ModelDetectApi\model\organ_aware_switch_vit\organ_aware_switch_vit_trt_cache"
$env:TRT_FP16="1"
$env:TRT_STRICT="1"
$env:TRT_WORKSPACE_GB="4"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 3) First startup behavior

On first run with TensorRT backend:

1. App loads `.pt` checkpoint.
2. If `ONNX_PATH` does not exist, app exports ONNX automatically.
3. ONNX Runtime TensorRT EP builds/loads TensorRT engine.
4. Engine is cached under `TRT_ENGINE_CACHE_DIR`.

First run is slower due to build/cache; next runs are faster.

## 4) Verify backend at runtime

Call:

```bash
curl http://127.0.0.1:8000/health
```

`model.backend` should be `tensorrt`.

## 5) Troubleshooting

### A. `TensorRT EP is not available`

- `onnxruntime-gpu` installed but does not include TensorRT EP build.
- Fix by installing matching ORT + TensorRT + CUDA versions.

### B. Export ONNX fails

- This model has custom routing logic and dynamic behavior.
- If export fails, keep backend `pytorch` temporarily and refactor unsupported ops.

### C. Shape mismatch

- This setup exports with input shape `(N,3,224,224)` and organ prior `(N,organ_dim)`.
- Keep preprocessing and model config aligned.

### D. No speedup

- Enable FP16 (`TRT_FP16=1`).
- Warm up model with a few requests before benchmarking.
- Ensure TensorRT provider is actually selected in `/health`.

## 6) Notes about compatibility

- API routes and request/response shape are kept unchanged.
- `routing` metrics are only fully available in PyTorch mode.
  In TensorRT mode, routing internals are not returned because exported graph only outputs
  classification logits and organ auxiliary logits.
- Với `efficientnetv2-mask2former`, `efficientnetv2-segformer`, `mobilenetv3large-segformer`, `mobilenetv3large-deeplabv3`, TensorRT chạy hybrid:
  segmentation branch vẫn chạy PyTorch, classifier backbone chạy TensorRT.
