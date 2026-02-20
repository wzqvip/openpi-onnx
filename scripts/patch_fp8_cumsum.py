#!/usr/bin/env python3
"""
修复 FP8 ONNX 模型的 CumSum 节点类型问题
将 FP8 类型的 CumSum 输入转换为 INT32，输出转换为 INT64
"""

import onnx
from onnx import helper, TensorProto
import sys

def patch_cumsum_for_fp8(input_path, output_path):
    """为 FP8 模型修补 CumSum 节点"""
    
    print(f"加载 ONNX 模型: {input_path}")
    model = onnx.load(input_path)
    
    cumsum_nodes = [n for n in model.graph.node if n.op_type == "CumSum"]
    print(f"找到 {len(cumsum_nodes)} 个 CumSum 节点")
    
    if not cumsum_nodes:
        print("没有 CumSum 节点需要修补")
        return
    
    patched_count = 0
    new_nodes = []
    
    for node in model.graph.node:
        if node.op_type == "CumSum":
            # 为每个 CumSum 节点添加类型转换
            original_output = node.output[0]
            cumsum_out = node.name + "_cumsum_int32_output"
            cast_in = node.name + "_cast_in_output"
            
            # Cast input to INT32 (from FP8 or bool)
            cast_in_node = helper.make_node(
                "Cast",
                inputs=node.input[:1],  # 第一个输入
                outputs=[cast_in],
                to=TensorProto.INT32,
                name=node.name + "_cast_in_patch"
            )
            
            # Modify CumSum to use INT32 input/output
            node.input[0] = cast_in
            node.output[0] = cumsum_out
            
            # Cast output to INT64 (CumSum 预期输出)
            cast_out_node = helper.make_node(
                "Cast",
                inputs=[cumsum_out],
                outputs=[original_output],
                to=TensorProto.INT64,
                name=node.name + "_cast_out_patch"
            )
            
            new_nodes.append(cast_in_node)
            new_nodes.append(node)
            new_nodes.append(cast_out_node)
            patched_count += 1
            
            print(f"  修补节点: {node.name}")
        else:
            new_nodes.append(node)
    
    if patched_count > 0:
        # Replace nodes
        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)
        print(f"\n✅ 成功修补 {patched_count} 个 CumSum 节点")
        
        # Save with external data format
        print(f"保存修补后的模型: {output_path}")
        from pathlib import Path
        onnx.save_model(
            model, 
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=Path(output_path).stem + ".data",
            size_threshold=1024,
        )
        print("✅ 模型修补完成（使用外部数据格式）")
    else:
        print("没有节点被修补")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python patch_fp8_cumsum.py <input_onnx> [output_onnx]")
        print("示例: python patch_fp8_cumsum.py model.nvfp8.modelopt.gs_clean.onnx model.nvfp8.patched.onnx")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".onnx", "_patched.onnx")
    
    patch_cumsum_for_fp8(input_path, output_path)
    
    print(f"\n完成！修补后的模型: {output_path}")
