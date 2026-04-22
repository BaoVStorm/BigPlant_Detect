# API Contract (Stable)

## Endpoints

- `GET /health`
- `POST /predict/file`
- `POST /predict/base64`

## Hanh vi can giu on dinh

- Request/response schema cua 3 endpoint tren khong doi neu khong co yeu cau ro rang.
- `routes.py` phai tiep tuc su dung:
  - `preprocess_pil(...)`
  - `predict_one(...)`
- Thong tin timing trong response (`preprocess_ms`, `inference_ms`, `total_ms`) giu nguyen.

## Lenh chay quen thuoc

- `uvicorn api_server:app --host 0.0.0.0 --port 8000`
