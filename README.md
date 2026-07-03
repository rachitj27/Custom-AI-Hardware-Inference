# Custom AI Hardware Inference

A from-scratch inference and quantization pipeline for a YOLOv8n fire and smoke detection model. Instead of treating quantization and inference as black-box library calls, this project implements the underlying math from scratch, then benchmarks the result against production runtimes.

The goal is to understand what production inference systems do internally — and eventually match their functionality with hand-written code and custom hardware.

## Status

- [x] **M1: FP32 baseline** — trained YOLOv8n on the fire-8 dataset, benchmarked across four production runtimes
- [x] **M2: From-scratch INT8 quantization** — implemented affine quantization, quantized matmul, and activation calibration end-to-end. Preserved 91% of FP32 mAP@0.5
- [ ] **M3: Custom C++ inference engine** — load the quantized model, run INT8 inference in hand-written C++, benchmark against ONNX Runtime and OpenVINO
- [ ] **M4: Hardware accelerator** — Verilog RTL FPGA accelerator (stretch goal)

## Model

YOLOv8n fine-tuned on the fire-8 dataset (2 classes: fire, smoke). Trained for 50 epochs on a Tesla T4 in Google Colab.

- Parameters: 3.0M
- Model size (FP32): 6.0 MB
- Model size (INT8): 2.9 MB (~4× compression)

## FP32 Baseline

Benchmarked across four production runtimes on 55 test images.

| Runtime | Hardware | Precision | Mean Latency |
|---------|----------|-----------|--------------|
| PyTorch (Ultralytics) | Intel Core Ultra 7 256V | FP32 | 45.77 ms |
| ONNX Runtime | Intel Core Ultra 7 256V | FP32 | 40.10 ms |
| OpenVINO | Intel Core Ultra 7 256V | FP32 | 30.57 ms |
| TensorRT | NVIDIA T4 (Colab) | FP16 | 12.39 ms |

mAP@0.5 = 0.9253.

## Quantization Results

| Configuration | mAP@0.5 | Drop from FP32 |
|---------------|---------|----------------|
| FP32 baseline | 0.9253 | — |
| INT8 weights (symmetric per-tensor) | 0.8467 | -8.5% |
| INT8 weights + activations (asymmetric per-tensor) | 0.8445 | -8.7% |

From-scratch quantization preserves 91% of the FP32 mAP@0.5.

## Approach

M2 implements post-training static quantization from first principles rather than calling `torch.quantization`:

- **Weights**: symmetric per-tensor INT8, computed directly from weight min/max
- **Activations**: asymmetric per-tensor INT8, calibrated on 100 training images using PyTorch forward hooks
- **Matmul**: INT32 accumulation to prevent overflow, requantization scale `M = s_A × s_B / s_C` applied per output

M2 validates the quantization scheme with "fake quantization" (quantize-then-dequantize applied during PyTorch's forward pass), which isolates the math from the runtime. M3 will build the actual INT8 execution engine in C++.

## Repo Contents

- `quantization.py` — from-scratch quantize / dequantize / quantized_matmul
- `calibrate.py` — activation calibration via forward hooks
- `quantized_inference.py` — end-to-end fake-quantized inference
- `save_model.py` — serialize the quantized model to a custom binary + JSON format
- `pt_benchmark.py`, `onnx_benchmark.py`, `openvino_benchmark.py` — M1 runtime baselines
- `activation_scales.json` — calibrated per-layer activation scales
- `model_int8.json` — quantized model metadata (weight scales, layer shapes, byte offsets)
- `Results.md` — full benchmark results and notes

## Dataset

[fire-8 from Abonia1](https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection) — 2-class fire and smoke detection dataset.