import torchao
try:
    import torchao.float8
    print("torchao.float8 members:")
    print(dir(torchao.float8))
except ImportError:
    print("torchao.float8 failed to import")

# Also check torchao.quantization.quant_api if it exists
try:
    import torchao.quantization.quant_api
    print("torchao.quantization.quant_api members:")
    print(dir(torchao.quantization.quant_api))
except ImportError:
    print("torchao.quantization.quant_api failed to import")


