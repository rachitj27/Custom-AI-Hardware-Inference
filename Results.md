## Custom C++ Engine Benchmark

Hardware: Intel Core Ultra 7 256V (laptop CPU)
Same test image, same 640x640 input, INT8 inference

| Metric | Value |
|--------|-------|
| Avg latency | 3145 ms |
| Min latency | 3069 ms |
| Max latency | 3270 ms |

## Comparison to Production Runtimes

| Runtime | Latency | Speedup vs custom engine |
|---------|---------|-------------------------|
| Custom C++ engine | 3145 ms | 1.0x (baseline) |
| PyTorch | 46 ms | 68x faster |
| ONNX Runtime | 40 ms | 79x faster |
| OpenVINO | 31 ms | 101x faster |
| TensorRT (T4 GPU) | 12 ms | 262x faster |

Production runtimes are significantly faster because they use hardware-specific optimizations that my hand-written naive C++ does not. Closing this gap would require implementing techniques like multi-threading and vectorization,