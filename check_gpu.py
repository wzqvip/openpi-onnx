
import onnxruntime as ort
import torch

print("Torch CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Torch Device Name:", torch.cuda.get_device_name(0))

print("ORT Available Providers:", ort.get_available_providers())

try:
    sess = ort.InferenceSession("./dist/final_w8a16_new/model.w8a16.onnx", providers=["CUDAExecutionProvider"])
    print("ORT Session Provider:", sess.get_providers())
except Exception as e:
    print("ORT Session Creation Failed:", e)
