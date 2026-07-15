from ultralytics import YOLO
import time
import glob
import numpy as np


MODEL_PATH = "best.pt"
DATA_YAML = "YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8/data.yaml"
TEST_IMAGES = "YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8/test/images/*.jpg"

# === Accuracy benchmark (mAP) ===
print("=== Running accuracy benchmark ===")
model = YOLO(MODEL_PATH)
metrics = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=640,
    verbose=False,
)

# === Latency benchmark ===
print("\n=== Running latency benchmark ===")
image_paths = sorted(glob.glob(TEST_IMAGES))
print(f"Found {len(image_paths)} test images")

print("Warming up...")
for _ in range(5):
    model.predict(image_paths[0], verbose=False)

print("Timing...")
latencies = []
for img_path in image_paths:
    start = time.perf_counter()
    model.predict(img_path, verbose=False)
    elapsed = (time.perf_counter() - start) * 1000
    latencies.append(elapsed)

latencies = np.array(latencies)

print("\n" + "="*40)
print("M1 FP32 BASELINE (Laptop CPU)")
print("="*40)
print(f"Model:        {MODEL_PATH}")
print(f"Images:       {len(image_paths)}")
print(f"\n--- Accuracy ---")
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")
print(f"\n--- Latency (per image) ---")
print(f"Mean:         {latencies.mean():.2f} ms")
print(f"Median:       {np.median(latencies):.2f} ms")
print(f"Std dev:      {latencies.std():.2f} ms")
print(f"Min:          {latencies.min():.2f} ms")
print(f"Max:          {latencies.max():.2f} ms")
print(f"P95:          {np.percentile(latencies, 95):.2f} ms")