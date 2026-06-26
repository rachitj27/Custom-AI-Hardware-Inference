# Custom AI Hardware Inference

Building a custom inference pipeline for an edge wildfire detection model, from quantization through hardware-accelerated deployment.

## Status
- [x] M1: FP32 baseline established across PyTorch, ONNX Runtime, OpenVINO, and TensorRT
- [ ] M2: From-scratch INT8 quantization
- [ ] M3: Custom C++ inference engine
- [ ] M4: Hardware acceleration (FPGA tier planned)

## Baseline Results

| Runtime | Hardware | Precision | Mean Latency (ms) |
|---------|----------|-----------|-------------------|
| PyTorch | Intel Core Ultra 7 256V | FP32 | 45.77 |
| ONNX Runtime | Intel Core Ultra 7 256V | FP32 | 40.10 |
| OpenVINO | Intel Core Ultra 7 256V | FP32 | 30.57 |
| TensorRT | NVIDIA T4 (Colab) | FP16 | 12.39 |

Model: YOLOv8n trained on the fire-8 dataset (2 classes: fire, smoke), mAP@0.5 = 0.9253.
Full results in [m1_baseline.md](m1_baseline.md).

## Approach

This project takes a trained vision model and builds the inference stack around it from scratch:

1. **Baseline (M1)**: Measure FP32 accuracy and latency through four production runtimes
2. **Quantization (M2)**: Implement INT8 post-training static quantization from scratch in NumPy
3. **Custom Engine (M3)**: Write a C++ inference engine that loads the quantized model and benchmarks against ONNX Runtime
4. **Hardware (M4)**: Build a custom FPGA accelerator with aggressive quantization (TBD)

## Repo Structure
- `pt_benchmark.py` - PyTorch FP32 baseline
- `onnx_benchmark.py` - ONNX Runtime benchmark
- `openvino_benchmark` - OpenVINO benchmark
- `export_onnx.py` - Convert .pt to .onnx
- `fire_training.ipynb` - Colab training and TensorRT export
- `m1_baseline.md` - Detailed baseline results

## Model and Dataset

- Model: YOLOv8n fine-tuned for fire/smoke detection
- Dataset: [fire-8 from Abonia1](https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection)
- Training: 50 epochs on Tesla T4 (Google Colab)