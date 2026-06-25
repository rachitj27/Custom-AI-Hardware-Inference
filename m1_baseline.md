# M1 FP32 Baseline

## Hardware
- CPU: Intel Core Ultra 7 256V (Lunar Lake)
- OS: Windows
- Python: 3.10.11
- PyTorch: 2.11.0 (CPU build)
- Test set: 55 images, 57 instances (fire-8 test split)

## Model
- Architecture: YOLOv8n (3,006,038 parameters, 8.1 GFLOPs)
- Trained: 50 epochs on T4 GPU via Google Colab
- Dataset: fire-8 (2 classes: fire, smoke)

## Accuracy
| Metric | Value |
|--------|-------|
| mAP@0.5      | 0.9253 |
| mAP@0.5:0.95 | 0.5469 |
| Precision    | 0.9376 |
| Recall       | 0.8466 |

## Latency Comparison (FP32, laptop CPU)

| Runtime | Mean (ms) | Median (ms) | P95 (ms) | Speedup vs PyTorch |
|---------|-----------|-------------|----------|--------------------|
| PyTorch (Ultralytics) | 45.77 | 42.75 | 60.75 | 1.00x |
| ONNX Runtime          | 40.10 | 38.98 | 51.90 | 1.14x |
| OpenVINO              | 30.57 | 29.93 | 35.04 | 1.50x |

## Observations
- OpenVINO is the fastest FP32 runtime on this Intel CPU, with the lowest variance (P95 only 5ms above median).
- ONNX Runtime provides modest improvement over PyTorch on Intel hardware because PyTorch already uses oneDNN under the hood for CPU inference.
- The C++ inference engine will use ONNX Runtime as the primary comparison target.

## Notes
- All latency numbers measured with 5 warm-up runs followed by 55 timed runs on the test set.
- PyTorch latency includes preprocessing and postprocessing (NMS). ONNX Runtime and OpenVINO numbers are inference only.