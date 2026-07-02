import numpy as np
import torch
import json
from ultralytics import YOLO
from quantization import compute_scale_zeropoint, quantize

# Load model fresh 
model = YOLO("best.pt")
pytorch_model = model.model
pytorch_model.eval()

# Load calibrated activation scales
with open("activation_scales.json") as f:
    activation_scales_raw = json.load(f)
activation_scales = {int(k): v for k, v in activation_scales_raw.items()}

# Collect quantized weights for every conv layer
conv_layers = []
weight_blobs = []
byte_offset = 0
layer_id = 0

for name, module in pytorch_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        w_fp32 = module.weight.detach().cpu().numpy()
        s, z = compute_scale_zeropoint(w_fp32, symmetric=True)
        w_int8 = quantize(w_fp32, s, z)
        
        # Serialize as raw bytes
        blob = w_int8.tobytes()
        byte_length = len(blob)
        
        conv_layers.append({
            "layer_id": layer_id,
            "path": name,
            "weight_shape": list(w_int8.shape),
            "byte_offset": byte_offset,
            "byte_length": byte_length,
            "weight_scale": float(s),
            "weight_zero_point": int(z),
            "quantization_scheme": "symmetric_per_tensor",
        })
        
        weight_blobs.append(blob)
        byte_offset += byte_length
        layer_id += 1

# Write binary blob
with open("model_int8.bin", "wb") as f:
    for blob in weight_blobs:
        f.write(blob)

print(f"Wrote model_int8.bin ({byte_offset} bytes, {byte_offset/1024:.1f} KB)")

# Write metadata JSON
metadata = {
    "num_bits": 8,
    "conv_layers": conv_layers,
    "activation_scales": {str(k): v for k, v in activation_scales.items()},
}

with open("model_int8.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Wrote model_int8.json ({len(conv_layers)} conv layers, {len(activation_scales)} activation scales)")

# reading back and reconstructing one layer
print("\nVerification: reading layer 0 back")
with open("model_int8.bin", "rb") as f:
    first_layer = conv_layers[0]
    f.seek(first_layer["byte_offset"])
    blob = f.read(first_layer["byte_length"])
    weights_back = np.frombuffer(blob, dtype=np.int8).reshape(first_layer["weight_shape"])
    print(f"Layer 0 shape: {weights_back.shape}")
    print(f"Layer 0 first 5 values: {weights_back.flatten()[:5]}")
    print(f"Scale: {first_layer['weight_scale']:.6f}, Zero-point: {first_layer['weight_zero_point']}")