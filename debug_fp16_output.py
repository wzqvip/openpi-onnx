#!/usr/bin/env python3
"""对比INT8和FP16引擎的输出"""
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    """加载TensorRT引擎"""
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def run_inference(engine, inputs):
    """运行推理"""
    context = engine.create_execution_context()
    
    # 分配GPU内存
    d_inputs = []
    d_outputs = []
    bindings = []
    
    # 处理输入
    for i, inp in enumerate(inputs):
        d_input = cuda.mem_alloc(inp.nbytes)
        cuda.memcpy_htod(d_input, inp)
        d_inputs.append(d_input)
        bindings.append(int(d_input))
    
    # 分配输出
    output_shape = (1, 10, 32)
    output = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)
    d_outputs.append(d_output)
    bindings.append(int(d_output))
    
    # 执行推理
    context.execute_v2(bindings)
    
    # 复制输出
    cuda.memcpy_dtoh(output, d_output)
    
    # 清理
    for d_input in d_inputs:
        d_input.free()
    for d_output in d_outputs:
        d_output.free()
    
    return output

def main():
    # 创建相同的随机输入
    np.random.seed(42)
    
    base_rgb = np.random.randn(1, 3, 224, 224).astype(np.float32)
    left_rgb = np.random.randn(1, 3, 224, 224).astype(np.float32)
    right_rgb = np.ones((1, 3, 224, 224), dtype=np.float32) * -1.0
    prompt = np.random.randint(0, 50000, (1, 200), dtype=np.int32)
    prompt_mask = np.zeros((1, 200), dtype=bool)
    prompt_mask[0, :20] = True
    noise = np.random.randn(1, 10, 32).astype(np.float32)
    
    inputs = [base_rgb, left_rgb, right_rgb, prompt, prompt_mask, noise]
    
    print("🔍 对比 INT8 vs FP16 Fixed 输出")
    print("=" * 60)
    
    # INT8
    print("\n📦 加载 INT8 引擎...")
    int8_engine = load_engine("checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine")
    int8_output = run_inference(int8_engine, inputs)
    
    print(f"INT8 输出:")
    print(f"  shape: {int8_output.shape}")
    print(f"  mean: {int8_output.mean():.6f}")
    print(f"  std: {int8_output.std():.6f}")
    print(f"  min: {int8_output.min():.6f}")
    print(f"  max: {int8_output.max():.6f}")
    print(f"  前10个值: {int8_output[0, 0, :10]}")
    
    # FP16
    print("\n📦 加载 FP16 Fixed 引擎...")
    fp16_engine = load_engine("checkpoints/pi05_libero_onnx_compat/engine_fp16_fixed.trt")
    fp16_output = run_inference(fp16_engine, inputs)
    
    print(f"\nFP16 Fixed 输出:")
    print(f"  shape: {fp16_output.shape}")
    print(f"  mean: {fp16_output.mean():.6f}")
    print(f"  std: {fp16_output.std():.6f}")
    print(f"  min: {fp16_output.min():.6f}")
    print(f"  max: {fp16_output.max():.6f}")
    print(f"  前10个值: {fp16_output[0, 0, :10]}")
    
    # 对比
    print("\n📊 差异分析:")
    diff = np.abs(int8_output - fp16_output)
    print(f"  绝对误差 mean: {diff.mean():.6f}")
    print(f"  绝对误差 max: {diff.max():.6f}")
    print(f"  相对误差 mean: {(diff / (np.abs(int8_output) + 1e-8)).mean():.6f}")
    
    # 检查是否有异常值
    if np.isnan(fp16_output).any():
        print(f"  ⚠️ FP16输出包含 NaN！")
    if np.isinf(fp16_output).any():
        print(f"  ⚠️ FP16输出包含 Inf！")
    
    # 检查值域
    if np.abs(fp16_output).max() > 100:
        print(f"  ⚠️ FP16输出值域异常大: {fp16_output.max()}")
    
    if diff.max() > 10:
        print(f"  ⚠️ 输出差异过大！可能存在数值问题")
    elif diff.max() > 1:
        print(f"  ⚠️ 输出差异较大，可能影响准确率")
    else:
        print(f"  ✅ 输出差异在合理范围内")

if __name__ == "__main__":
    main()
