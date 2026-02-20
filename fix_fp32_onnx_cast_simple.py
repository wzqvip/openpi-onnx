#!/usr/bin/env python3
"""
简单方法：移除ONNX中无用的Cast操作
"""

import onnx
import sys

def simplify_cast_nodes(onnx_path, output_path):
    """
    移除或简化Cast节点
    """
    print(f"加载ONNX模型: {onnx_path}")
    model = onnx.load(onnx_path)
    graph = model.graph
    
    print(f"原始节点数: {len(graph.node)}")
    
    # 找到所有Cast节点
    cast_nodes = []
    for node in graph.node:
        if node.op_type == "Cast":
            cast_nodes.append(node)
    
    print(f"找到 {len(cast_nodes)} 个Cast节点")
    
    # 分析并标记冗余节点
    nodes_to_remove_names = set()
    
    for node in cast_nodes:
        if len(node.attribute) > 0:
            for attr in node.attribute:
                if attr.name == "to":
                    to_type = attr.i
                    # 1 = FLOAT, 10 = FLOAT16
                    # 对于FLOAT->FLOAT的Cast，检查输入类型
                    input_name = node.input[0]
                    output_name = node.output[0]
                    
                    # 检查输入节点的输出类型
                    input_producer = None
                    for n in graph.node:
                        if output_name in [o for o in n.output]:
                            input_producer = n
                            break
                    
                    # 简单策略：Float32->Float32的Cast都可以移除
                    if to_type == 1:  # FLOAT = 1
                        print(f"  移除冗余Cast: {node.name} ({input_name} -> {output_name})")
                        nodes_to_remove_names.add(node.name)
                        
                        # 重连：用input_name替换output_name在所有后续节点中的引用
                        for consumer_node in graph.node:
                            for i, inp in enumerate(consumer_node.input):
                                if inp == output_name:
                                    print(f"    重连: {consumer_node.name} 的输入 {i}")
                                    consumer_node.input[i] = input_name
    
    # 移除标记的节点
    removed = 0
    new_nodes = []
    for node in graph.node:
        if node.name not in nodes_to_remove_names:
            new_nodes.append(node)
        else:
            removed += 1
    
    del graph.node[:]
    for node in new_nodes:
        graph.node.append(node)
    
    print(f"\n移除节点数: {removed}")
    print(f"新节点数: {len(graph.node)}")
    
    # 保存
    print(f"保存到: {output_path}")
    onnx.save(model, output_path)
    
    print(f"✅ 完成!")
    return output_path


if __name__ == "__main__":
    onnx_path = "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.onnx"
    output_path = "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.fixed.onnx"
    
    simplify_cast_nodes(onnx_path, output_path)
