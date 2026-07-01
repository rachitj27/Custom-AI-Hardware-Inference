import numpy as np
import torch
import json
from ultralytics import YOLO
from quantization import compute_scale_zeropoint, quantize, dequantize

# Load the model
model = YOLO("best.pt")
pytorch_model = model.model
pytorch_model.eval()

# Find all conv layers and replace their weights 
conv_count = 0
for name, module in pytorch_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        
        w_fp32 = module.weight.detach().cpu().numpy()
        
        # Quantize 
        s, z = compute_scale_zeropoint(w_fp32, symmetric=True)
        w_int8 = quantize(w_fp32, s, z)
        w_recovered = dequantize(w_int8, s, z)
        
        # Replace the weight with the quantized-then-dequantized version
        module.weight.data = torch.from_numpy(w_recovered).float()
        
        conv_count += 1

print(f"Fake-quantized weights of {conv_count} conv layers")

# Run validation with weights-only fake quantization
metrics = model.val(
    data="YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8/data.yaml",
    split="test",
    imgsz=640,
    verbose=False,
)

print(f"\n=== Weights-only fake-quantization (INT8, symmetric per-tensor) ===")
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

# Load calibrated activation scales
with open("activation_scales.json") as f:
    activation_scales_raw = json.load(f)
activation_scales = {int(k): v for k, v in activation_scales_raw.items()}

def fake_quantize_tensor(x, scale, zero_point, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    q = torch.round(x / scale) + zero_point
    q = torch.clamp(q, qmin, qmax)
    return scale * (q - zero_point)

# Attach hooks that fake-quantize each top-level layer's output
activation_hooks = []
for i, layer in enumerate(pytorch_model.model):
    if i not in activation_scales:
        continue
    
    scale = activation_scales[i]["scale"]
    zero_point = activation_scales[i]["zero_point"]
    
    def make_act_hook(s, z):
        def hook(module, input, output):
            if isinstance(output, tuple):
                return tuple(fake_quantize_tensor(o, s, z) if torch.is_tensor(o) else o for o in output)
            if torch.is_tensor(output):
                return fake_quantize_tensor(output, s, z)
            return output
        return hook
    
    h = layer.register_forward_hook(make_act_hook(scale, zero_point))
    activation_hooks.append(h)

print(f"Attached fake-quantization hooks to {len(activation_hooks)} layers")

# Run validation again with both weights AND activations quantized
metrics = model.val(
    data="YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8/data.yaml",
    split="test",
    imgsz=640,
    verbose=False,
)

print(f"\n=== Full fake-quantization (INT8 weights + activations) ===")
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

# Clean up
for h in activation_hooks:
    h.remove()