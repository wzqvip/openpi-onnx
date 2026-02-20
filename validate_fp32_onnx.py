#!/usr/bin/env python3
"""
验证FP32 ONNX模型是否正常工作
"""

import onnx
import onnxruntime as ort
import numpy as np
import os

def validate_fp32_onnx():
    """验证FP32 ONNX模型"""
    
    # 检查两个版本的ONNX文件
    onnx_files = {
        "原始": "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.onnx",
        "修复后": "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.fixed.onnx"
    }
    
    for label, onnx_path in onnx_files.items():
        print(f"\n{'='*80}")
        print(f"验证 {label} ONNX: {onnx_path}")
        print(f"{'='*80}")
        
        if not os.path.exists(onnx_path):
            print(f"❌ 文件不存在")
            continue
        
        try:
            # 1. 加载ONNX模型
            print(f"\n1️⃣ 加载ONNX模型...")
            model = onnx.load(onnx_path)
            print(f"✅ 模型加载成功")
            
            # 2. 检查模型结构
            print(f"\n2️⃣ 模型信息...")
            graph = model.graph
            print(f"  节点数: {len(graph.node)}")
            print(f"  输入数: {len(graph.input)}")
            print(f"  输出数: {len(graph.output)}")
            
            # 输入信息
            print(f"\n  输入:")
            for inp in graph.input:
                print(f"    - {inp.name}")
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        print(f"      dim: {dim.dim_value}", end=" ")
                    else:
                        print(f"      dim: ?", end=" ")
                print()
            
            # 输出信息
            print(f"\n  输出:")
            for out in graph.output:
                print(f"    - {out.name}")
            
            # 3. 验证模型
            print(f"\n3️⃣ 验证ONNX模型...")
            try:
                onnx.checker.check_model(model)
                print(f"✅ 模型验证通过")
            except Exception as e:
                print(f"⚠️  模型验证警告: {str(e)[:200]}")
            
            # 4. 尝试用ONNX Runtime加载
            print(f"\n4️⃣ 用ONNX Runtime加载...")
            try:
                sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                print(f"✅ ONNX Runtime加载成功")
                
                # 5. 测试推理
                print(f"\n5️⃣ 测试推理...")
                
                # 创建虚拟输入
                input_names = sess.get_inputs()
                output_names = sess.get_outputs()
                
                print(f"  输入数量: {len(input_names)}")
                for inp in input_names:
                    print(f"    - {inp.name}: shape={inp.shape}, type={inp.type}")
                
                # 构建虚拟输入字典
                dummy_inputs = {}
                for inp in input_names:
                    shape = list(inp.shape)
                    # 替换动态维度
                    shape = [s if isinstance(s, int) else 1 for s in shape]
                    dummy_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
                
                # 运行推理
                outputs = sess.run(None, dummy_inputs)
                print(f"✅ 推理成功!")
                print(f"  输出数量: {len(outputs)}")
                for i, out in enumerate(outputs):
                    print(f"    输出{i}: shape={out.shape}, dtype={out.dtype}")
                    print(f"      范围: [{out.min():.4f}, {out.max():.4f}]")
                    print(f"      是否有NaN: {np.isnan(out).any()}")
                    print(f"      是否有Inf: {np.isinf(out).any()}")
                
            except Exception as e:
                print(f"❌ ONNX Runtime错误: {e}")
                import traceback
                traceback.print_exc()
            
            # 6. 文件大小
            file_size = os.path.getsize(onnx_path) / (1024**2)
            print(f"\n6️⃣ 文件信息...")
            print(f"  大小: {file_size:.1f} MB")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("✅ 验证完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    validate_fp32_onnx()
