import onnxruntime as ort
import numpy as np
import cv2
import glob
import time

session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

image_paths = sorted(glob.glob("YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8/test/images/*.jpg"))

# Preprocess one image to get the right shape
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, 0)  # add batch dim
    return img

# Warm-up
for _ in range(5):
    session.run(None, {input_name: preprocess(image_paths[0])})

# Timing
latencies = []
for path in image_paths:
    img = preprocess(path)
    start = time.perf_counter()
    session.run(None, {input_name: img})
    latencies.append((time.perf_counter() - start) * 1000)

latencies = np.array(latencies)
print(f"ONNX Runtime FP32 Latency")
print(f"Mean:   {latencies.mean():.2f} ms")
print(f"Median: {np.median(latencies):.2f} ms")
print(f"P95:    {np.percentile(latencies, 95):.2f} ms")