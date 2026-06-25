import openvino as ov
import numpy as np
import cv2
import glob
import time

core = ov.Core()
model = core.read_model("best.onnx")
compiled = core.compile_model(model, "CPU")
output_layer = compiled.output(0)

image_paths = sorted(glob.glob("YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8/test/images/*.jpg"))

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img

# Warm-up
for _ in range(5):
    compiled([preprocess(image_paths[0])])

# Timing
latencies = []
for path in image_paths:
    img = preprocess(path)
    start = time.perf_counter()
    compiled([img])
    latencies.append((time.perf_counter() - start) * 1000)

latencies = np.array(latencies)
print(f"OpenVINO FP32 Latency")
print(f"Mean:   {latencies.mean():.2f} ms")
print(f"Median: {np.median(latencies):.2f} ms")
print(f"P95:    {np.percentile(latencies, 95):.2f} ms")