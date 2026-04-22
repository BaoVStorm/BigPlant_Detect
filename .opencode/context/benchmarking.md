# Benchmarking Context

## Notebooks

- `notebooks/benchmark_backends.ipynb`
- `notebooks/plot_benchmark_results.ipynb`

## Output artifacts

Duoc luu trong `benchmark_outputs/<timestamp>/`:

- `benchmark_summary.csv`
- `benchmark_raw_per_image.csv`
- `benchmark_summary.json`
- `benchmark_compare_trt_vs_torch.csv`
- `benchmark_charts.png`

## Metrics can theo doi

- `accuracy_top1`
- `startup_ms`
- `first_infer_ms`
- `infer_mean_ms`, `infer_p95_ms`

## Nguyen tac benchmark

- Tach startup va steady-state latency.
- Warmup truoc khi tinh latency.
- Giu cache TRT rieng theo checkpoint.
