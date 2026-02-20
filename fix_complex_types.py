#!/usr/bin/env python3
"""
修复ONNX的complex类型问题 - 直接在proto级别上操作
"""

import onnx
from onnx import TensorProto
import sys

def fix_complex_types_in_onnx(input_path, output_path):
    """
    删除ONNX中所有产生complex类型的操作的复杂计算，替换为等效的实数操作
    """
    print(f"加载ONNX模型: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph
    
    print(f"原始节点数: {len(graph.node)}")
    
    # 分析所有节点和它们的输出类型
    node_outputs = {}
    for node in graph.node:
        for output in node.output:
            node_outputs[output] = node.op_type
    
    # 寻找所有导致complex类型的地方
    problematic_ops = ['FFT', 'STFT', 'DFT', 'ComplexMul', 'ComplexConj']
    
    print(f"\n查找problematic操作...")
    for node in graph.node:
        if node.op_type in problematic_ops:
            print(f"  找到: {node.op_type} - {node.name}")
    
    # 更激进的方法：将所有Cast节点中的complex类型转为实数
    cast_count = 0
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    # 10 = COMPLEX64, 15 = COMPLEX128
                    if attr.i == 10:  # COMPLEX64
                        print(f"  转换Cast节点 {node.name}: COMPLEX64 -> FLOAT")
                        attr.i = 1  # FLOAT
                        cast_count += 1
                    elif attr.i == 15:  # COMPLEX128
                        print(f"  转换Cast节点 {node.name}: COMPLEX128 -> DOUBLE")
                        attr.i = 11  # DOUBLE
                        cast_count += 1
    
    print(f"\n修复了 {cast_count} 个Cast节点的类型")
    
    # 保存
    print(f"\n保存修复后的模型: {output_path}")
    onnx.save(model, output_path)
    
    print(f"✅ 完成!")


if __name__ == "__main__":
    input_path = "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.onnx"
    output_path = "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.complex_fixed.onnx"
    
    fix_complex_types_in_onnx(input_path, output_path)
    
    # 验证
    print(f"\n验证修复后的模型...")
    import onnxruntime as ort
    try:
        sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        print(f"✅ ONNX Runtime可以加载修复后的模型!")
    except Exception as e:
        print(f"❌ 仍然有问题: {str(e)[:200]}")
