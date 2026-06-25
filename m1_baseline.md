# M1 FP32 Baseline

## Hardware
- CPU: Intel Core Ultra 7 256V
- Framework: Ultralytics 8.4.26, PyTorch 2.11.0 (CPU)
- Test set: 55 images, 57 instances (fire-8 test split)

## Accuracy
| Metric | Value |
|--------|-------|
| mAP@0.5      | 0.9253 |
| mAP@0.5:0.95 | 0.5469 |
| Precision    | 0.9376 |
| Recall       | 0.8466 |

## Latency (per image)
| Metric | Value (ms) |
|--------|-----------|
| Mean   | 45.77 |
| Median | 42.75 |
| Std dev| 10.65 |
| Min    | 39.12 |
| Max    | 97.02 |
| P95    | 60.75 |
## Latency Comparison (FP32, laptop CPU)

| Runtime | Mean (ms) | Median (ms) | P95 (ms) |
|---------|-----------|-------------|----------|
| PyTorch (Ultralytics) | 45.77 | 42.75 | 60.75 |
| ONNX Runtime          | 47.74 | 42.80 | 80.43 |

Note: PyTorch and ONNX Runtime perform similarly on this CPU . The C++ engine will
target ONNX Runtime as the production reference point.