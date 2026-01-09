import subprocess
import sys

cmd = [
    "/usr/src/tensorrt/bin/trtexec",
    "--onnx=checkpoints/pi05_libero_pytorch/model.fp32.onnx",
    "--duration=1",
    "--noDataTransfers"
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(result.stdout)
