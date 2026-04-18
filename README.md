# ModelDetectApi

API nhận diện ảnh thực vật với FastAPI, hỗ trợ 2 backend suy luận:

- `pytorch` (mặc định)
- `tensorrt` (qua ONNX Runtime TensorRT Execution Provider)

Hỗ trợ model script:

- `organ_aware_switch_vit`
- `efficientnetv2-segformer` (TensorRT chạy theo kiểu hybrid: SegFormer PyTorch + classifier TensorRT)
- `efficientnetv2-mask2former` (Mask2Former foreground-guided + EfficientNetV2-S classifier)
- `mobilenetv3large-segformer` (SegFormer-B4 foreground-guided + MobileNetV3-Large classifier)
- `mobilenetv3large-deeplabv3` (DeepLabV3 foreground-guided + MobileNetV3-Large classifier)

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

Luu y: `transformers` da pin `4.41.2` de tuong thich voi torch hien tai trong env.

## 3) Cấu hình bằng `.env`

Project đã đọc `.env` tự động tại `app/core/config.py`.

Ví dụ `.env` hiện tại:

```env
MODEL_DIR=./model
MODEL_SCRIPT=organ_aware_switch_vit
MODEL=
DEVICE=cuda:0
TOPK=5

INFER_BACKEND=tensorrt
TRT_DEVICE_ID=0
TRT_FP16=1
TRT_STRICT=1
TRT_WORKSPACE_GB=4
```

Giải thích nhanh biến quan trọng:

- `MODEL`: (tuỳ chọn) tên file model `.pt` trong `./model/<MODEL_SCRIPT>/` hoặc đường dẫn tuyệt đối
- `MODEL_DIR`: thư mục gốc chứa model theo từng script
- `DEVICE`: `auto|cpu|cuda|cuda:0`
- `MODEL_SCRIPT`: `organ_aware_switch_vit` hoặc `efficientnetv2-segformer` hoặc `efficientnetv2-mask2former` hoặc `mobilenetv3large-segformer` hoặc `mobilenetv3large-deeplabv3`
- `INFER_BACKEND`: `pytorch` hoặc `tensorrt`
- `ONNX_PATH`: mặc định tự suy ra `./model/<MODEL_SCRIPT>/<MODEL_SCRIPT>.onnx`
- `TRT_ENGINE_CACHE_DIR`: mặc định `./model/<MODEL_SCRIPT>/<MODEL_SCRIPT>_trt_cache`
- `TRT_FP16`: bật FP16 để tăng tốc
- `TRT_STRICT=1`: chỉ dùng TensorRT (không fallback)

## 4) Chạy server

```powershell
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

D:\App\anaconda3\envs\pt_gpu\python.exe -m uvicorn api_server:app --host 0.0.0.0 --port 8000

D:\App\anaconda3\envs\pt_gpu\python.exe -m uvicorn  D:\Homework\BackEnd\ModelDetectApi\api_server:app --host 0.0.0.0 --port 8000

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

- `Checkpoint not found`: kiểm tra `MODEL_SCRIPT`, `MODEL_DIR`, và `MODEL` (nếu có set)
- `TensorRT EP is not available`: version ORT/TensorRT/CUDA chưa tương thích
- Lỗi ONNX export: model có logic dynamic, có thể cần fallback `pytorch`

## 10) Dùng model `efficientnetv2-segformer`

Đã hỗ trợ nạp checkpoint `efficientnetv2s-segformerb4.pt` cho inference PyTorch.

```env
MODEL_SCRIPT=efficientnetv2-segformer
INFER_BACKEND=pytorch
MODEL_DIR=./model
MODEL=efficientnetv2s-segformerb4.pt
```

Nếu để trống `MODEL`, app tự chọn checkpoint theo thứ tự:

- ưu tiên `./model/<MODEL_SCRIPT>/best_model.pt` (nếu có)
- tiếp theo `./model/<MODEL_SCRIPT>/<MODEL_SCRIPT>.pt`

Gợi ý cấu trúc thư mục:

```text
model/
  organ_aware_switch_vit/
    best_model.pt
    organ_aware_switch_vit.onnx
    organ_aware_switch_vit_trt_cache/
  efficientnetv2-segformer/
    efficientnetv2s-segformerb4.pt
    efficientnetv2-segformer.onnx
    efficientnetv2-segformer_trt_cache/
```

Thông số train của checkpoint (tham chiếu bạn cung cấp):

- `epochs=50`
- `batch_size=32`
- `lr=1e-4`
- `weight_decay=1e-4`
- `mask_score_threshold=0.5`

Lưu ý: TensorRT cho `efficientnetv2-segformer` chạy theo hướng hybrid
(SegFormer vẫn chạy PyTorch, classifier EfficientNet chạy TensorRT).

## 11) Dùng model `efficientnetv2-mask2former`

Checkpoint bạn cung cấp `efficientnetv2s-mask2former.pt` đã được hỗ trợ cho inference.

```env
MODEL_SCRIPT=efficientnetv2-mask2former
INFER_BACKEND=pytorch
MODEL_DIR=./model
MODEL=efficientnetv2s-mask2former.pt
```

Gợi ý: có thể để file checkpoint ở:

- `./model/efficientnetv2-mask2former/efficientnetv2s-mask2former.pt`
- hoặc trực tiếp `./model/efficientnetv2s-mask2former.pt`

Model này dùng Mask2Former để tách foreground trước khi classify bằng EfficientNetV2-S.

## 12) Dùng model `mobilenetv3large-segformer`

Checkpoint bạn cung cấp `mobilenetv3large-segformerb4.pt` đã được hỗ trợ cho inference.

```env
MODEL_SCRIPT=mobilenetv3large-segformer
INFER_BACKEND=pytorch
MODEL_DIR=./model
MODEL=mobilenetv3large-segformerb4.pt
```

Vị trí file khuyến nghị:

- `./model/mobilenetv3large-segformer/mobilenetv3large-segformerb4.pt`

Model này dùng SegFormer-B4 tạo foreground mask và MobileNetV3-Large để classify.

---

Nếu cần, mình có thể bổ sung thêm phần benchmark latency (PyTorch vs TensorRT) ngay trong README.
