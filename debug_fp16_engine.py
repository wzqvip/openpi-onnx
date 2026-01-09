import subprocess

cmd = [
    "/usr/src/tensorrt/bin/trtexec",
    "--loadEngine=model.trt",
    "--duration=1",
    "--noDataTransfers"
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(result.stdout)
