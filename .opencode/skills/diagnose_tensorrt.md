# Skill: Diagnose TensorRT Runtime

## Muc tieu

Khoanh vung va xu ly loi khi backend TensorRT khoi dong/chay sai.

## Checklist chan doan

1. Kiem tra provider:
   - `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`
   - Phai co `TensorrtExecutionProvider`.
2. Kiem tra `DEVICE` la CUDA (`cuda` hoac `cuda:0`).
3. Kiem tra `ONNX_PATH`/`TRT_ENGINE_CACHE_DIR` co tach theo checkpoint.
4. Kiem tra quyen ghi vao cache dir.
5. Kiem tra Windows DLL TensorRT (`TRT_DLL_DIR`/`TRT_ROOT`/`PATH`) neu gap loi load provider.
6. Kiem tra `/health` de xac nhan backend thuc su la `tensorrt`.

## Dau hieu de nham lan cache

- Doi model 384/512 nhung startup bat thuong nhanh va ket qua nghi ngo.
- Cach xu ly:
  - Tach ONNX/cache theo checkpoint stem
  - Hoac xoa cache cu va build lai
