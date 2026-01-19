
import modelopt.torch.quantization as mtq
print("Available MTQ Configs:")
for x in dir(mtq):
    if "CFG" in x:
        print(x)
