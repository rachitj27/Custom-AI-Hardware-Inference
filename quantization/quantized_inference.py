import numpy as np
import torch
import json
from ultralytics import YOLO
from quantization import compute_scale_zeropoint, quantize, dequantize


model = YOLO("best.pt")
pytorch_model = model.model
pytorch_model.eval()

# Fake-quantize all conv weights
conv_count = 0
for name, module in pytorch_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        w_fp32 = module.weight.detach().cpu().numpy()
        s, z = compute_scale_zeropoint(w_fp32, symmetric=True)
        w_int8 = quantize(w_fp32, s, z)
        w_recovered = dequantize(w_int8, s, z)
        module.weight.data = torch.from_numpy(w_recovered).float()
        conv_count += 1

print(f"Fake-quantized weights of {conv_count} conv layers")

# Load activation scales and attach fake-quantization hooks
with open("activation_scales.json") as f:
    activation_scales_raw = json.load(f)
activation_scales = {int(k): v for k, v in activation_scales_raw.items()}

def fake_quantize_tensor(x, scale, zero_point, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    q = torch.round(x / scale) + zero_point
    q = torch.clamp(q, qmin, qmax)
    return scale * (q - zero_point)

SKIP_TYPES = ("Detect", "Concat", "Upsample")

activation_hooks = []
for i, layer in enumerate(pytorch_model.model):
    if i not in activation_scales:
        continue
    
    layer_type = type(layer).__name__
    if layer_type in SKIP_TYPES:
        print(f"Skipping layer {i} ({layer_type})")
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

#  Run validation 
metrics = model.val(
    data="YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8/data.yaml",
    split="test",
    imgsz=640,
    verbose=False,
)

print(f"\n=== Full fake-quantization (INT8 weights + activations) ===")
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")


for h in activation_hooks:
    h.remove()


