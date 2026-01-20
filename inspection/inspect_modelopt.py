
import os
import modelopt
import modelopt.torch.quantization as mtq

print("--- ModelOpt Quantization Configs ---")
for x in dir(mtq):
    if "CFG" in x:
        print(x)
