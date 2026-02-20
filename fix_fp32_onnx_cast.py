#!/usr/bin/env python3
"""
修复FP32 ONNX模型中的Cast操作问题
去除或替换不兼容的Cast节点以使其能被TensorRT处理
"""

import onnx
import onnx_graphsurgeon as gs
import numpy as np

def fix_cast_operations(onnx_path, output_path):
    """
    修复ONNX中不兼容的Cast操作
    主要问题：Float32到Float32的Cast操作在vision tower中无法被TensorRT处理
    """
    print(f"加载ONNX模型: {onnx_path}")
    graph = gs.import_onnx(onnx.load(onnx_path))
    
    print(f"模型节点数: {len(graph.nodes)}")
    
    # 收集所有Cast节点
    cast_nodes = [node for node in graph.nodes if node.op == "Cast"]
    print(f"找到 {len(cast_nodes)} 个Cast节点")
    
    # 分析Cast节点
    cast_float32_count = 0
    removed_count = 0
    
    nodes_to_remove = []
    
    for i, node in enumerate(cast_nodes):
        # 检查Cast的源和目标类型
        if len(node.attrs) > 0 and 'to' in node.attrs:
            target_type = node.attrs['to']
            # Float32 = 1, Float16 = 10, BFloat16 = 16
            if target_type == 1:  # Cast to Float32
                # 检查输入是否已经是Float32
                input_node = node.inputs[0]
                if hasattr(input_node, 'dtype') and input_node.dtype == np.float32:
                    print(f"  节点{node.name}: Float32->Float32 Cast (冗余), 移除")
                    nodes_to_remove.append(node)
                    cast_float32_count += 1
                    removed_count += 1
    
    # 移除冗余的Cast节点
    if nodes_to_remove:
        print(f"\n移除{len(nodes_to_remove)}个冗余Cast节点")
        for node in nodes_to_remove:
            # 重连：输入直接连到输出用户
            if len(node.inputs) > 0 and len(node.outputs) > 0:
                input_tensor = node.inputs[0]
                output_tensor = node.outputs[0]
                
                # 找所有使用output的节点，替换为input
                for consumer_node in graph.nodes:
                    for i, inp in enumerate(consumer_node.inputs):
                        if inp == output_tensor:
                            consumer_node.inputs[i] = input_tensor
                
                # 移除节点
                graph.nodes.remove(node)
    
    print(f"\n清理图...")
    graph.cleanup().toposort()
    
    # 保存修复后的ONNX
    print(f"保存修复后的ONNX: {output_path}")
    onnx.save(
        gs.export_onnx(graph),
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.fp32.fixed.data",
        size_threshold=1024,
        convert_attribute=False
    )
    
    print(f"✅ 修复完成!")
    print(f"  移除节点数: {removed_count}")
    print(f"  输出文件大小: {np.format_float_positional(np.float32(1.0) * 1e-6)}")


if __name__ == "__main__":
    import sys
    
    onnx_path = "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.onnx"
    output_path = "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.fixed.onnx"
    
    fix_cast_operations(onnx_path, output_path)
    
    print("\n下一步: 用修复后的ONNX重新构建TensorRT引擎")
    print(f"python3 scripts/build_trt_engine.py {output_path} --output checkpoints/pi05_libero_onnx_compat/engine_fp32_unrolled.trt --workspace 8")
