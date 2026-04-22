# Skill: Benchmark PyTorch vs TensorRT

## Muc tieu

So sanh accuracy va latency giua 2 backend tren nhieu checkpoint.

## Quy trinh

1. Mo `notebooks/benchmark_backends.ipynb`.
2. Set `MODELS_ROOT`, `DATA_ROOT`, `DEVICE_STR`, `MAX_IMAGES_PER_CLASS`, `WARMUP_IMAGES`.
3. Chay toan bo notebook.
4. Kiem tra `benchmark_summary.csv` va `benchmark_compare_trt_vs_torch.csv`.
5. Mo `notebooks/plot_benchmark_results.ipynb` de ve lai chart nhanh.

## Chu y

- Khong benchmark qua endpoint API; su dung ham service truc tiep.
- Tach startup latency (`startup_ms`) va steady latency (`infer_*`).
- Luon de cache TRT/ONNX rieng theo checkpoint.
