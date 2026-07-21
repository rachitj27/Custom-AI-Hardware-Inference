import numpy as np
import torch
import json
from ultralytics import YOLO
from quantization import compute_scale_zeropoint, quantize

# Load model fresh
model = YOLO("../best.pt")
pytorch_model = model.model
pytorch_model.eval()

# Load calibrated activation scales
with open("activation_scales.json") as f:
    activation_scales_raw = json.load(f)
activation_scales = {int(k): v for k, v in activation_scales_raw.items()}

# Collect quantized weights for every conv layer, keyed by module path
# so we can look them up later when building the architecture
conv_layers_by_path = {}
conv_layers = []
weight_blobs = []
byte_offset = 0
layer_id = 0

for name, module in pytorch_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        w_fp32 = module.weight.detach().cpu().numpy()
        s, z = compute_scale_zeropoint(w_fp32, symmetric=True)
        w_int8 = quantize(w_fp32, s, z)
        
        blob = w_int8.tobytes()
        byte_length = len(blob)
        
        entry = {
            "layer_id": layer_id,
            "path": name,
            "weight_shape": list(w_int8.shape),
            "byte_offset": byte_offset,
            "byte_length": byte_length,
            "weight_scale": float(s),
            "weight_zero_point": int(z),
            "quantization_scheme": "symmetric_per_tensor",
        }
        
        conv_layers.append(entry)
        conv_layers_by_path[name] = layer_id
        
        weight_blobs.append(blob)
        byte_offset += byte_length
        layer_id += 1

# Write binary blob
with open("../model_int8.bin", "wb") as f:
    for blob in weight_blobs:
        f.write(blob)

print(f"Wrote model_int8.bin ({byte_offset} bytes, {byte_offset/1024:.1f} KB)")

# Build architecture metadata: describe each of the 23 top-level modules
architecture = []

# YOLOv8n's skip connections and structure (hardcoded from the model spec)
# input_from: which earlier layer_id(s) feed into this module
# For most: previous layer. For Concat: multiple layers.
# For Detect: three feature maps.

def find_convs_in_module(module, base_path):
    """Find all conv layer paths inside a module (recursively)."""
    conv_paths = []
    for name, submod in module.named_modules():
        full_path = f"{base_path}.{name}" if name else base_path
        if isinstance(submod, torch.nn.Conv2d) and full_path in conv_layers_by_path:
            conv_paths.append(full_path)
    return conv_paths

# YOLOv8n architecture from Ultralytics' yolov8.yaml
# Format: (from, repeats, module_type, args)
# Layer index in pytorch_model.model corresponds to layer_id in architecture
yolov8n_arch = [
    # (from, module_type, extra_info)
    ("input", "Conv", {}),         # 0
    (0, "Conv", {}),               # 1
    (1, "C2f", {"n": 1, "shortcut": True}),  # 2
    (2, "Conv", {}),               # 3
    (3, "C2f", {"n": 2, "shortcut": True}),  # 4
    (4, "Conv", {}),               # 5
    (5, "C2f", {"n": 2, "shortcut": True}),  # 6
    (6, "Conv", {}),               # 7
    (7, "C2f", {"n": 1, "shortcut": True}),  # 8
    (8, "SPPF", {"k": 5}),         # 9
    (9, "Upsample", {}),            # 10
    ([10, 6], "Concat", {}),        # 11 - concat with layer 6
    (11, "C2f", {"n": 1, "shortcut": False}),  # 12
    (12, "Upsample", {}),           # 13
    ([13, 4], "Concat", {}),        # 14 - concat with layer 4
    (14, "C2f", {"n": 1, "shortcut": False}),  # 15
    (15, "Conv", {}),               # 16
    ([16, 12], "Concat", {}),       # 17 - concat with layer 12
    (17, "C2f", {"n": 1, "shortcut": False}),  # 18
    (18, "Conv", {}),               # 19
    ([19, 9], "Concat", {}),        # 20 - concat with layer 9
    (20, "C2f", {"n": 1, "shortcut": False}),  # 21
    ([15, 18, 21], "Detect", {}),   # 22 - detect from three scales
]

# For each top-level layer, find its internal conv layer IDs
for layer_idx, (input_from, module_type, extra) in enumerate(yolov8n_arch):
    module = pytorch_model.model[layer_idx]
    base_path = f"model.{layer_idx}"
    
    # Find all conv paths inside this module and get their layer_ids
    conv_ids = []
    for path, cid in conv_layers_by_path.items():
        if path.startswith(base_path + ".") or path == base_path:
            conv_ids.append(cid)
    conv_ids.sort()
    
    entry = {
        "layer_id": layer_idx,
        "type": module_type,
        "conv_ids": conv_ids,
        "input_from": input_from,
    }
    entry.update(extra)
    architecture.append(entry)

# Write metadata JSON
metadata = {
    "num_bits": 8,
    "conv_layers": conv_layers,
    "activation_scales": {str(k): v for k, v in activation_scales.items()},
    "architecture": architecture,
}

with open("model_int8.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Wrote model_int8.json ({len(conv_layers)} conv layers, {len(activation_scales)} activation scales, {len(architecture)} top-level modules)")

# Verification: print the architecture map
print("\nArchitecture map:")
for arch in architecture:
    print(f"  Layer {arch['layer_id']}: {arch['type']} (convs: {arch['conv_ids']}, from: {arch['input_from']})")