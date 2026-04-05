# ModelDetectApi

API nhận diện ảnh thực vật với FastAPI, hỗ trợ 2 backend suy luận:

- `pytorch` (mặc định)
- `tensorrt` (qua ONNX Runtime TensorRT Execution Provider)

Giữ nguyên endpoint và cách chạy quen thuộc:

- `GET /health`
- `POST /predict/file`
- `POST /predict/base64`
- Run: `uvicorn api_server:app --host 0.0.0.0 --port 8000`

## 1) Yêu cầu hệ thống

- Python 3.10+
- GPU NVIDIA + driver hoạt động (`nvidia-smi`)
- (Khuyến nghị) Conda env riêng, ví dụ `pt_gpu`

## 2) Cài đặt

```powershell
conda activate pt_gpu
pip install -r requirements.txt
```

## 3) Cấu hình bằng `.env`

Project đã đọc `.env` tự động tại `app/core/config.py`.

Ví dụ `.env` hiện tại:

```env
CKPT=./model/best_model.pt
DEVICE=cuda:0
TOPK=5
MODEL_SCRIPT=organ_aware_switch_vit

INFER_BACKEND=tensorrt
ONNX_PATH=./model/best_model.onnx
TRT_ENGINE_CACHE_DIR=./model/trt_cache
TRT_DEVICE_ID=0
TRT_FP16=1
TRT_STRICT=1
TRT_WORKSPACE_GB=4
```

Giải thích nhanh biến quan trọng:

- `CKPT`: đường dẫn model `.pt`
- `DEVICE`: `auto|cpu|cuda|cuda:0`
- `MODEL_SCRIPT`: `organ_aware_switch_vit` hoặc `efficientnetv2-segformer`
- `INFER_BACKEND`: `pytorch` hoặc `tensorrt`
- `ONNX_PATH`: nơi lưu file ONNX export tự động
- `TRT_ENGINE_CACHE_DIR`: cache engine TensorRT
- `TRT_FP16`: bật FP16 để tăng tốc
- `TRT_STRICT=1`: chỉ dùng TensorRT (không fallback)

## 4) Chạy server

```powershell
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

D:\App\anaconda3\envs\pt_gpu\python.exe -m uvicorn api_server:app --host 0.0.0.0 --port 8000
D:\App\anaconda3\envs\pt_gpu\python.exe -c "import ctypes; ctypes.WinDLL('nvinfer_10.dll'); ctypes.WinDLL('nvinfer_plugin_10.dll'); ctypes.WinDLL('nvonnxparser_10.dll'); print('TRT DLL chain OK')"

Hoặc:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 5) Kiểm tra nhanh

### Health check

```powershell
curl http://127.0.0.1:8000/health
```

Khi chạy TensorRT, kiểm tra `model.backend` trong response là `tensorrt`.

### Predict từ file

```powershell
curl -X POST "http://127.0.0.1:8000/predict/file?topk=5&two_pass=true" -F "file=@src/input/Abrus longibracteatus_17.jpeg"
```

### Predict từ base64

```powershell
curl -X POST "http://127.0.0.1:8000/predict/base64" -H "Content-Type: application/json" -d "{\"image_b64\":\"<BASE64_IMAGE>\",\"topk\":5,\"two_pass\":true}"
```

## 6) TensorRT: lưu ý vận hành

- Lần chạy đầu có thể chậm do:
  1. Export ONNX (nếu chưa có `ONNX_PATH`)
  2. Build TensorRT engine
- Từ lần sau nhanh hơn nhờ cache ở `TRT_ENGINE_CACHE_DIR`
- Nếu lỗi provider TensorRT, chạy kiểm tra:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Phải có `TensorrtExecutionProvider`.

## 7) Chuyển nhanh giữa 2 backend

- Dùng PyTorch (ổn định hơn):

```env
INFER_BACKEND=pytorch
```

- Dùng TensorRT (tối ưu tốc độ, rủi ro tương thích cao hơn):

```env
INFER_BACKEND=tensorrt
```

## 8) Cấu trúc chính

- `app/main.py`: khởi tạo FastAPI, load model/runtime
- `app/api/routes.py`: định nghĩa endpoint
- `app/services/model_loader.py`: load checkpoint + chọn backend
- `app/services/onnx_export.py`: export ONNX tự động
- `app/services/runtime_engine.py`: runtime TensorRT (ORT EP)
- `app/services/predictor.py`: pipeline suy luận chung

## 9) Troubleshooting thường gặp

- `Checkpoint not found`: kiểm tra `CKPT` trong `.env`
- `TensorRT EP is not available`: version ORT/TensorRT/CUDA chưa tương thích
- Lỗi ONNX export: model có logic dynamic, có thể cần fallback `pytorch`
- `MODEL_SCRIPT=efficientnetv2-segformer` hiện mới khai báo selector, chưa implement pipeline inference

---

Nếu cần, mình có thể bổ sung thêm phần benchmark latency (PyTorch vs TensorRT) ngay trong README.
