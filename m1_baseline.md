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

## Latency Comparison

| Runtime | Hardware | Precision | Mean (ms) | Median (ms) | P95 (ms) |
|---------|----------|-----------|-----------|-------------|----------|
| PyTorch (Ultralytics) | Intel Core Ultra 7 256V | FP32 | 45.77 | 42.75 | 60.75 |
| ONNX Runtime | Intel Core Ultra 7 256V | FP32 | 40.10 | 38.98 | 51.90 |
| OpenVINO | Intel Core Ultra 7 256V | FP32 | 30.57 | 29.93 | 35.04 |
| TensorRT | NVIDIA T4 (Colab) | FP16 | 12.39 | 12.05 | 13.94 |



## Observations
- OpenVINO is the fastest FP32 runtime on this Intel CPU, with the lowest variance (P95 only 5ms above median).
- ONNX Runtime provides modest improvement over PyTorch on Intel hardware because PyTorch already uses oneDNN under the hood for CPU inference.
- The C++ inference engine will use ONNX Runtime as the primary comparison target.

## Notes
- All latency numbers measured with 5 warm-up runs followed by 55 timed runs on the test set.
- PyTorch latency includes preprocessing and postprocessing (NMS). ONNX Runtime and OpenVINO numbers are inference only.
- Note: TensorRT measured on Google Colab T4 in default FP16 mode, the standard
production configuration for NVIDIA inference. CPU runtimes measured at FP32.