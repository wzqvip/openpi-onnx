#!/usr/bin/env python3
"""修补 FP16 ONNX 模型的 CumSum 节点类型"""

import onnx
from onnx import helper, TensorProto
import sys
from pathlib import Path

def patch_cumsum_for_fp16(input_path, output_path):
    """修补 FP16 ONNX 中的 CumSum 节点"""
    print(f"加载 ONNX 模型: {input_path}")
    model = onnx.load(input_path)
    
    # 查找所有 CumSum 节点
    cumsum_nodes = [node for node in model.graph.node if node.op_type == "CumSum"]
    print(f"找到 {len(cumsum_nodes)} 个 CumSum 节点")
    
    if len(cumsum_nodes) == 0:
        print("⚠️ 没有找到 CumSum 节点，模型可能已经修补过")
        return
    
    # 为每个 CumSum 节点添加类型转换
    nodes_to_add = []
    nodes_to_remove = []
    
    for i, node in enumerate(cumsum_nodes):
        print(f"  修补节点: {node.name}")
        
        # 创建新的节点名称
        cast_in_name = f"{node.name}_cast_in"
        cast_out_name = f"{node.name}_cast_out"
        cast_in = f"{node.input[0]}_int32"
        cumsum_out = f"{node.output[0]}_int32"
        original_output = node.output[0]
        
        # Cast: FP16 → INT32
        cast_in_node = helper.make_node(
            "Cast",
            inputs=[node.input[0]],
            outputs=[cast_in],
            to=TensorProto.INT32,
            name=cast_in_name
        )
        
        # 更新 CumSum 节点
        node.input[0] = cast_in
        node.output[0] = cumsum_out
        
        # Cast: INT32 → INT64 (TensorRT 要求)
        cast_out_node = helper.make_node(
            "Cast",
            inputs=[cumsum_out],
            outputs=[original_output],
            to=TensorProto.INT64,
            name=cast_out_name
        )
        
        nodes_to_add.extend([cast_in_node, cast_out_node])
    
    # 将新节点插入到图中
    print(f"\n✅ 成功修补 {len(cumsum_nodes)} 个 CumSum 节点")
    
    # 找到第一个 CumSum 节点的位置
    insert_pos = 0
    for i, node in enumerate(model.graph.node):
        if node.op_type == "CumSum":
            insert_pos = i
            break
    
    # 在 CumSum 节点之前插入 Cast 节点
    for node in reversed(nodes_to_add):
        model.graph.node.insert(insert_pos, node)
    
    # 保存修补后的模型（使用外部数据）
    print(f"保存修补后的模型: {output_path}")
    onnx.save_model(
        model, 
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=Path(output_path).stem + ".data",
        size_threshold=1024,
    )
    print("✅ 模型修补完成（使用外部数据格式）")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python patch_fp16_cumsum.py <input_onnx> [output_onnx]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.onnx', '.patched.onnx')
    
    patch_cumsum_for_fp16(input_path, output_path)
    print(f"\n完成！修补后的模型: {output_path}")
