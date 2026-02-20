#!/usr/bin/env python3
"""
修复CumSum类型错误 - 使用ONNX优化器
"""

import onnx
from onnxruntime.transformers import optimizer
import sys

def fix_cumsum_type_error():
    """
    修复CumSum的bool类型错误
    """
    onnx_path = "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.onnx"
    output_path = "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.fixed.onnx"
    
    print(f"加载ONNX模型: {onnx_path}")
    model = onnx.load(onnx_path)
    graph = model.graph
    
    print(f"原始节点数: {len(graph.node)}")
    
    # 查找所有CumSum节点及其输入源
    cumsum_nodes = []
    for node in graph.node:
        if node.op_type == "CumSum":
            cumsum_nodes.append(node)
            print(f"\n找到CumSum节点: {node.name}")
            print(f"  输入: {node.input}")
            print(f"  输出: {node.output}")
    
    print(f"\n总共找到 {len(cumsum_nodes)} 个CumSum节点")
    
    # 对每个CumSum节点，检查其轴输入(axis)的类型
    fixed_count = 0
    for cumsum_node in cumsum_nodes:
        if len(cumsum_node.input) >= 2:
            axis_input_name = cumsum_node.input[1]
            print(f"\nCumSum {cumsum_node.name}的轴输入: {axis_input_name}")
            
            # 找到产生这个轴的节点
            for producer_node in graph.node:
                if axis_input_name in producer_node.output:
                    print(f"  来自节点: {producer_node.name} ({producer_node.op_type})")
                    
                    # 如果是Constant节点，可以修改其类型
                    if producer_node.op_type == "Constant":
                        for attr in producer_node.attribute:
                            if attr.name == "value":
                                # 检查tensor_value的类型
                                tensor_value = attr.t
                                print(f"    当前dtype: {tensor_value.data_type}")
                                # int64 = 7
                                if tensor_value.data_type == 9:  # bool = 9
                                    tensor_value.data_type = 7  # 改为int64
                                    print(f"    修改为: int64 (7)")
                                    fixed_count += 1
    
    print(f"\n修复了 {fixed_count} 个Constant节点的类型")
    
    # 尝试修复 - 将所有bool类型的Constant转为int64
    bool_to_int_count = 0
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    tensor_value = attr.t
                    if tensor_value.data_type == 9:  # bool
                        print(f"\nConvert bool->int64 in {node.name}")
                        tensor_value.data_type = 7  # int64
                        bool_to_int_count += 1
    
    print(f"\n总共转换了 {bool_to_int_count} 个bool常数为int64")
    
    # 保存修复后的模型
    print(f"\n保存到: {output_path}")
    onnx.save(model, output_path)
    
    print(f"✅ 修复完成!")
    
    return output_path


if __name__ == "__main__":
    fix_cumsum_type_error()
