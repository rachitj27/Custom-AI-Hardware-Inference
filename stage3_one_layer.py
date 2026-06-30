import numpy as np
import torch
from ultralytics import YOLO


from quantization import compute_scale_zeropoint, quantize, dequantize


model = YOLO("best.pt")
pytorch_model = model.model
layer0 = pytorch_model.model[0]


weights_fp32 = layer0.conv.weight.detach().numpy()
print(f"Weight stats: min={weights_fp32.min():.4f}, max={weights_fp32.max():.4f}")

# Quantize weights
s, z = compute_scale_zeropoint(weights_fp32, symmetric=True)
print(f"Scale: {s:.6f}, Zero-point: {z}")

q_weights = quantize(weights_fp32, s, z)
weights_recovered = dequantize(q_weights, s, z)

# Compare original vs quantized
error = np.abs(weights_fp32 - weights_recovered)
print(f"Quantization error: mean={error.mean():.6f}, max={error.max():.6f}")
print(f"Relative to scale: max error / scale = {error.max() / s:.3f}")


#  Run conv layer with both weights
import torch.nn.functional as F

# Make a fake input image 
np.random.seed(42)
input_image = np.random.randn(1, 3, 640, 640).astype(np.float32)
input_tensor = torch.from_numpy(input_image)

# Get the original weights tensor 
weights_fp32_tensor = layer0.conv.weight.detach()

# FP32 forward pass through just the conv 
output_fp32 = F.conv2d(
    input_tensor, weights_fp32_tensor,
    stride=2, padding=1
)
print(f"\nFP32 output shape: {output_fp32.shape}")
print(f"FP32 output range: [{output_fp32.min():.4f}, {output_fp32.max():.4f}]")

# Fake-quantized forward pass (quantize then dequantize weights)
weights_fakeq_tensor = torch.from_numpy(weights_recovered).float()
output_fakeq = F.conv2d(
    input_tensor, weights_fakeq_tensor,
    stride=2, padding=1
)
print(f"Fake-q output range: [{output_fakeq.min():.4f}, {output_fakeq.max():.4f}]")

#  Compare outputs
output_error = np.abs(output_fp32.numpy() - output_fakeq.numpy())
print(f"\nOutput error: mean={output_error.mean():.6f}, max={output_error.max():.6f}")
print(f"Mean relative error: {(output_error.mean() / np.abs(output_fp32.numpy()).mean()):.3%}")