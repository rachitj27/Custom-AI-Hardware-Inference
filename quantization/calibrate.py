import numpy as np
import torch
import glob
from ultralytics import YOLO
from PIL import Image
import cv2


model = YOLO("best.pt")
pytorch_model = model.model
pytorch_model.eval()  # inference mode, no gradients


calib_paths = sorted(glob.glob("YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8/train/images/*.jpg"))[:100]
print(f"Using {len(calib_paths)} calibration images")

# Storage for activation ranges
activation_ranges = {}

def make_hook(layer_idx):
    """Create a hook that records min/max of this layer's output."""
    def hook(module, input, output):
        
        if isinstance(output, tuple):
            output = output[0]
        # Handle non-tensor outputs
        if not torch.is_tensor(output):
            return
        
        out_min = output.detach().min().item()
        out_max = output.detach().max().item()
        
        if layer_idx not in activation_ranges:
            activation_ranges[layer_idx] = {"min": out_min, "max": out_max}
        else:
            activation_ranges[layer_idx]["min"] = min(activation_ranges[layer_idx]["min"], out_min)
            activation_ranges[layer_idx]["max"] = max(activation_ranges[layer_idx]["max"], out_max)
    return hook

# Attach hooks to every layer
hooks = []
for i, layer in enumerate(pytorch_model.model):
    h = layer.register_forward_hook(make_hook(i))
    hooks.append(h)

print(f"Attached hooks to {len(hooks)} layers")


#  converts image file to model input tensor
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  
    img = np.expand_dims(img, 0)  
    return torch.from_numpy(img)

# Run all calibration images through the model
print("Running calibration...")
with torch.no_grad():
    for i, img_path in enumerate(calib_paths):
        img_tensor = preprocess_image(img_path)
        pytorch_model(img_tensor)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(calib_paths)}")

print(f"\nCalibration complete. Captured ranges for {len(activation_ranges)} layers.")


print("\nSample activation ranges:")
for idx in [0, 5, 10, 15, 20]:
    if idx in activation_ranges:
        r = activation_ranges[idx]
        print(f"  Layer {idx}: [{r['min']:.3f}, {r['max']:.3f}]")

# Compute per-layer scale and zero-point from observed ranges
from quantization import compute_scale_zeropoint

activation_scales = {}
for layer_idx, r in activation_ranges.items():
    range_tensor = np.array([r["min"], r["max"]], dtype=np.float32)
    scale, zero_point = compute_scale_zeropoint(range_tensor, symmetric=False)
    activation_scales[layer_idx] = {
        "scale": float(scale),
        "zero_point": int(zero_point),
        "min": r["min"],
        "max": r["max"],
    }


print("\nActivation scales computed:")
for idx in sorted(activation_scales.keys())[:5]:
    s = activation_scales[idx]
    print(f"  Layer {idx}: scale={s['scale']:.5f}, zp={s['zero_point']}, range=[{s['min']:.3f}, {s['max']:.3f}]")

# Save to disk
import json
with open("activation_scales.json", "w") as f:
    json.dump({str(k): v for k, v in activation_scales.items()}, f, indent=2)

print(f"\nSaved {len(activation_scales)} activation scales to activation_scales.json")


for h in hooks:
    h.remove()
print("Hooks removed.")