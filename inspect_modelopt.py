
import os
import modelopt
import modelopt.torch.quantization
try:
    import modelopt.deploy.tensorrt
    print("modelopt.deploy.tensorrt imported")
except ImportError:
    print("modelopt.deploy.tensorrt NOT found")

print(f"ModelOpt path: {os.path.dirname(modelopt.__file__)}")

# Walk through modelopt dir to find .so files
for root, dirs, files in os.walk(os.path.dirname(modelopt.__file__)):
    for file in files:
        if file.endswith(".so"):
            print(f"Found lib: {os.path.join(root, file)}")
