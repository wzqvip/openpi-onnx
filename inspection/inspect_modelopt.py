import modelopt.torch.quantization as mtq
print("Attributes of mtq:")
for x in dir(mtq):
    print(x)
print("\nChecking for export:")
if hasattr(mtq, 'export_model'):
    print("Found export_model")
if hasattr(mtq, 'convert'):
    print("Found convert")
