import onnx

model = onnx.load("checkpoints/pi05_libero_pytorch/model.fp32.onnx")
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")

print("\nOutputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
