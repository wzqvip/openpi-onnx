#!/usr/bin/env python3
"""
PyTorch 升级和 FP8 支持验证脚本

功能:
  1. 检查当前 PyTorch 版本
  2. 显示 PyTorch 2.9.1 升级指令
  3. 验证 FP8 支持
  4. 推荐后续步骤

使用: python3 upgrade_pytorch_and_test_fp8.py
"""

import torch
import subprocess
import sys


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_section(text):
    print(f"\n📌 {text}")
    print("-" * 60)


def check_pytorch_version():
    """检查当前 PyTorch 版本"""
    print_section("当前 PyTorch 版本")
    print(f"✓ PyTorch 版本: {torch.__version__}")
    print(f"✓ CUDA 可用: {torch.cuda.is_available()}")
    print(f"✓ Python 版本: {sys.version.split()[0]}")
    
    # 解析版本号
    version_str = torch.__version__.split('+')[0]
    major, minor, patch = map(int, version_str.split('.')[:3])
    
    needs_upgrade = (major, minor, patch) < (2, 9, 1)
    return needs_upgrade, (major, minor, patch)


def show_upgrade_instructions():
    """显示 PyTorch 2.9.1 升级指令"""
    print_section("升级到 PyTorch 2.9.1")
    
    print("📥 选项 1: 使用 Jetson AI Lab (推荐 for Jetson Orin)")
    print("""
pip install --upgrade 'torch>=2.9.1' \\
  -i https://pypi.jetson-ai-lab.io/sbsa/cu130/
    """.strip())
    
    print("\n📥 选项 2: 使用官方 PyPI")
    print("""
pip install --upgrade 'torch>=2.9.1'
    """.strip())
    
    print("\n⚙️ 升级后验证:")
    print("""
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
    """.strip())


def test_fp8_operations():
    """测试 FP8 基础操作支持"""
    print_section("FP8 操作支持检查")
    
    # 创建 FP8 张量
    try:
        w_fp8 = torch.randn(4, 4).to(torch.float8_e4m3fn)
        print("✓ FP8 张量创建成功")
    except Exception as e:
        print(f"✗ FP8 张量创建失败: {e}")
        return 0
    
    # 测试关键操作
    test_cases = [
        ("sum", lambda x: x.sum()),
        ("mean", lambda x: x.mean()),
        ("add", lambda x: x + x),
        ("matmul", lambda x: x @ torch.randn(4, 4)),
    ]
    
    supported_count = 0
    results = {}
    
    for op_name, op_fn in test_cases:
        try:
            result = op_fn(w_fp8)
            print(f"✅ {op_name:10} - 支持")
            results[op_name] = True
            supported_count += 1
        except RuntimeError as e:
            error_msg = str(e).split('\n')[0][:40]
            print(f"❌ {op_name:10} - {error_msg}...")
            results[op_name] = False
        except Exception as e:
            print(f"❌ {op_name:10} - 异常: {type(e).__name__}")
            results[op_name] = False
    
    return supported_count, results


def analyze_fp8_support(supported_count):
    """分析 FP8 支持等级"""
    print_section("FP8 支持等级分析")
    
    support_level = supported_count / 4
    
    if support_level >= 0.75:
        level = "🟢 高 (推荐使用 FP8)"
        action = "✅ 重新运行: python3 verify_fp8_libero.py"
    elif support_level >= 0.5:
        level = "🟡 中等 (可尝试使用 FP8)"
        action = "⚠️ 可尝试运行 FP8 评估，但可能部分失败"
    else:
        level = "🔴 低 (推荐使用 INT8)"
        action = "❌ 降级到 INT8: python3 scripts/eval_libero_trt_v1.py"
    
    print(f"支持率: {supported_count}/4 = {support_level*100:.0f}%")
    print(f"等级: {level}")
    print(f"\n推荐行动:")
    print(f"  {action}")
    
    return support_level >= 0.75


def recommend_next_steps(needs_upgrade, support_level_ok):
    """推荐后续步骤"""
    print_section("后续步骤")
    
    if needs_upgrade:
        print("🔴 版本过旧: 需要升级 PyTorch")
        print("\n推荐步骤:")
        print("1️⃣  执行上方的升级指令")
        print("2️⃣  升级完成后，重新运行此脚本验证")
        print("3️⃣  根据 FP8 支持等级选择方案")
        
    elif support_level_ok:
        print("🟢 PyTorch 版本合适，FP8 支持良好")
        print("\n推荐步骤:")
        print("1️⃣  运行 FP8 LIBERO 评估:")
        print("     python3 verify_fp8_libero.py")
        print("\n2️⃣  如果成功，评估 LIBERO Spatial 任务:")
        print("     TRIALS_PER_TASK=20 python3 scripts/eval_libero_trt_v1.py \\")
        print("       --checkpoint_dir checkpoints/pi05_libero_pytorch_fp8")
        
    else:
        print("🟡 PyTorch 版本合适，但 FP8 支持不足")
        print("\n推荐步骤:")
        print("1️⃣  降级到 INT8 方案:")
        print("     python3 scripts/eval_libero_trt_v1.py \\")
        print("       --checkpoint_dir checkpoints/pi05_libero_pytorch_int8_dynamic")
        print("\n2️⃣  如果 INT8 成功率 ≥95%，构建 TensorRT:")
        print("     python3 scripts/build_trt_engine.py \\")
        print("       --checkpoint_dir checkpoints/pi05_libero_pytorch_int8_dynamic \\")
        print("       --quantization_type int8")


def main():
    print_header("PyTorch 升级和 FP8 支持验证工具")
    
    # 1. 检查版本
    needs_upgrade, current_version = check_pytorch_version()
    
    if needs_upgrade:
        print(f"\n⚠️  当前版本 {'.'.join(map(str, current_version))} 过旧")
        print("   建议升级到 2.9.1 以获得更好的 FP8 支持")
        show_upgrade_instructions()
        
        print_section("升级后的下一步")
        print("1. 完成上方的升级命令")
        print("2. 重新运行此脚本验证:")
        print("   python3 upgrade_pytorch_and_test_fp8.py")
        
    else:
        print(f"\n✅ 版本 {'.'.join(map(str, current_version))} 满足要求")
        
        # 2. 测试 FP8 支持
        result = test_fp8_operations()
        
        if isinstance(result, tuple):
            supported_count, results = result
        else:
            supported_count = result
        
        # 3. 分析支持等级
        support_level_ok = analyze_fp8_support(supported_count)
        
        # 4. 推荐后续步骤
        recommend_next_steps(needs_upgrade=False, support_level_ok=support_level_ok)
    
    # 显示诊断报告
    print_section("详细诊断报告")
    print("📄 完整的诊断和分析已保存至:")
    print("   /home/taco/openpi-onnx/FP8_DIAGNOSTIC_REPORT.md")
    print("\n可使用以下命令查看:")
    print("   cat FP8_DIAGNOSTIC_REPORT.md | less")
    print("   # 或在编辑器中打开")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
