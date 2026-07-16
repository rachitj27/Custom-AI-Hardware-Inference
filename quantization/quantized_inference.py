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
with open("quantization/activation_scales.json") as f:
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


# ==================================================
# Dump reference data for C++ engine validation
# ==================================================

import torch
from PIL import Image
import torchvision.transforms as T

# Load one real test image
test_img_path = "YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8/test/images"  # adjust to your path
import os
first_image = sorted(os.listdir(test_img_path))[0]
img_path = os.path.join(test_img_path, first_image)

img = Image.open(img_path).convert("RGB")
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),  # gives us FP32 in [0, 1]
])
img_tensor = transform(img).unsqueeze(0)  # shape (1, 3, 640, 640)

# Quantize input to INT8 using input_scale
input_scale = 1.0 / 255.0  # standard for images
input_zero_point = -128

img_int8 = np.round(img_tensor.numpy() / input_scale + input_zero_point)
img_int8 = np.clip(img_int8, -128, 127).astype(np.int8)
img_int8 = img_int8[0]  # remove batch dim, now (3, 640, 640)

print(f"Input shape: {img_int8.shape}, dtype: {img_int8.dtype}")
print(f"Input first values: {img_int8.flatten()[:8]}")

# Save the quantized input as raw bytes
img_int8.tofile("test_input.bin")
print("Saved test_input.bin")

# Now run just layer 0 in FP32 to get the reference output
# (We'll do a proper fake-quantized version)

# Get layer 0 from the model
layer0_conv = None
for module in model.model.modules():
    if hasattr(module, "conv") and isinstance(module.conv, torch.nn.Conv2d):
        layer0_conv = module.conv
        break

if layer0_conv is None:
    print("Could not find layer 0 conv, adjust the search logic")
else:
    # Extract layer 0's weights (already quantized in fake_quantize style)
    # For validation, use the same weight quantization the C++ engine will use
    with open("quantization/model_int8.json") as f:
        import json
        metadata = json.load(f)
    
    layer0_info = metadata["conv_layers"][0]
    weight_scale = layer0_info["weight_scale"]
    
    # Load weights from bin
    with open("model_int8.bin", "rb") as f:
        f.seek(layer0_info["byte_offset"])
        w_bytes = f.read(layer0_info["byte_length"])
    
    w_int8 = np.frombuffer(w_bytes, dtype=np.int8).reshape(layer0_info["weight_shape"])
    print(f"Weight shape: {w_int8.shape}")
    
    # Convert input and weights back to FP32 for a "correct" reference calculation
    input_fp32 = (img_int8.astype(np.float32) - input_zero_point) * input_scale
    weight_fp32 = w_int8.astype(np.float32) * weight_scale
    
    # Run conv in FP32 using torch for the reference
    input_torch = torch.from_numpy(input_fp32).unsqueeze(0)
    weight_torch = torch.from_numpy(weight_fp32)
    output_fp32 = torch.nn.functional.conv2d(input_torch, weight_torch, stride=2, padding=1)
    output_fp32 = output_fp32.squeeze(0).numpy()  # shape (16, 320, 320)
    
    # Quantize output using the calibrated activation scale for layer 0
    with open("quantization/activation_scales.json") as f:
        act_scales = json.load(f)
    
    output_scale = act_scales["0"]["scale"]
    output_zp = act_scales["0"]["zero_point"]
    
    output_int8 = np.round(output_fp32 / output_scale + output_zp)
    output_int8 = np.clip(output_int8, -128, 127).astype(np.int8)
    
    print(f"Output shape: {output_int8.shape}")
    print(f"Output first values: {output_int8.flatten()[:8]}")
    print(f"Output scale: {output_scale}, zero_point: {output_zp}")
    
    output_int8.tofile("test_output.bin")
    print("Saved test_output.bin")