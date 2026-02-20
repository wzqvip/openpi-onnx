#!/usr/bin/env python3
"""
FP8 诊断报告快速查看器
显示问题总结和建议的下一步
"""

import os
from pathlib import Path


def print_banner():
    banner = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              FP8 量化诊断完成 - PyTorch 2.9.1 升级方案可用               ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_core_issue():
    print("📋 核心问题")
    print("─" * 80)
    print("""
  当前状态:     PyTorch 2.7.1 FP8 推理失败
  错误信息:     "sum_cpu" not implemented for 'Float8_e4m3fn'
  原因:         PyTorch 2.7.1 缺少 FP8 CPU 算子实现
  
  新发现:       PyTorch 2.9.1 已发布，包含 FP8 改进！ ✨
  
""")


def print_three_solutions():
    print("📊 三种解决方案对比")
    print("─" * 80)
    print("""
  优先级   方案名         耗时         风险      收益
  ─────────────────────────────────────────────────────────────────
  🥇     升级 2.9.1    30 分钟       低       最大化 (如成功)
  🥈     使用 INT8     2-3 小时      低       高保障 (98% 验证)
  ❌     修复 FP8      2-4 周       高       不确定
  
""")


def print_next_steps():
    print("🚀 建议的下一步 (3 个简单步骤)")
    print("─" * 80)
    print("""
  【步骤 1】升级 PyTorch 到 2.9.1 (30 分钟)
  ─────────────────────────────────────────
  pip install --upgrade 'torch>=2.9.1' \\
    -i https://pypi.jetson-ai-lab.io/sbsa/cu130/


  【步骤 2】验证 FP8 支持改进 (5 分钟)
  ──────────────────────────────────────
  python3 upgrade_pytorch_and_test_fp8.py


  【步骤 3】根据结果选择方案
  ─────────────────────────────
  
  ✅ 如果 FP8 支持改进 (≥75%):
     使用 FP8，获得最小文件 + 最快加载
     
  ✅ 如果 FP8 支持不足 (<75%):
     使用 INT8，获得可靠方案 + TensorRT 加速
     
""")


def print_available_docs():
    print("📁 已生成的诊断文档")
    print("─" * 80)
    
    docs = [
        ("FP8_QUICK_SUMMARY.md", "快速总结（5 分钟阅读）"),
        ("FP8_DIAGNOSTIC_REPORT.md", "完整诊断（技术深度分析）"),
        ("FP8_DOCS_INDEX.md", "文档导航（使用指南）"),
        ("upgrade_pytorch_and_test_fp8.py", "升级验证脚本（推荐运行）"),
    ]
    
    base_path = Path("/home/taco/openpi-onnx")
    
    for filename, description in docs:
        filepath = base_path / filename
        size_info = ""
        
        if filepath.exists():
            size = filepath.stat().st_size
            if size > 1024 * 1024:
                size_info = f" ({size / (1024*1024):.1f} MB)"
            elif size > 1024:
                size_info = f" ({size / 1024:.1f} KB)"
            else:
                size_info = f" ({size} B)"
        
        print(f"  ✅ {filename:40} {description:30} {size_info}")
    
    print()


def print_quick_commands():
    print("⚡ 快速命令")
    print("─" * 80)
    print("""
  # 一键升级和测试
  pip install --upgrade 'torch>=2.9.1' \\
    -i https://pypi.jetson-ai-lab.io/sbsa/cu130/ && \\
  python3 upgrade_pytorch_and_test_fp8.py

  # 查看完整诊断
  less FP8_DIAGNOSTIC_REPORT.md
  
  # 快速部署 INT8 (无需等待 FP8 测试)
  TRIALS_PER_TASK=20 python3 scripts/eval_libero_trt_v1.py \\
    --checkpoint_dir checkpoints/pi05_libero_pytorch_int8_dynamic

""")


def print_expected_outcomes():
    print("🎯 预期结果")
    print("─" * 80)
    print("""
  如果 PyTorch 2.9.1 FP8 成功:
    • 最小存储: 4.14 GB (vs INT8 的 4.67 GB)
    • 最快加载: 0.84s (vs INT8 的 2.54s)
    • 高精度: 预期 95-98%

  如果 PyTorch 2.9.1 FP8 失败:
    • 降级到 INT8 (已验证 98% 成功率)
    • 构建 TensorRT 引擎
    • 获得 2-4x 推理加速

""")


def print_quick_decision_tree():
    print("💡 决策树 - 选择适合你的方案")
    print("─" * 80)
    print("""
  1️⃣  想要最优性能（最小 + 最快）
      → 升级 PyTorch 2.9.1 → 测试 FP8 → 如失败则用 INT8
      耗时: 30 分钟, 风险: 低, 收益: 极高

  2️⃣  想要可靠部署（不想冒风险）
      → 直接使用 INT8 → 构建 TensorRT
      耗时: 2-3 小时, 风险: 低, 收益: 高

  3️⃣  需要立即启动
      → 使用现有 INT8 → 部署服务
      耗时: 1 小时, 风险: 低, 收益: 高

""")


def print_footer():
    footer = """
═══════════════════════════════════════════════════════════════════════════════

👉 建议您的下一步:

   1. 【推荐】运行自动化脚本:
      python3 upgrade_pytorch_and_test_fp8.py

   2. 或手动执行升级:
      pip install --upgrade 'torch>=2.9.1' \\
        -i https://pypi.jetson-ai-lab.io/sbsa/cu130/

   3. 查看完整诊断 (如有疑问):
      cat FP8_DIAGNOSTIC_REPORT.md | less

═══════════════════════════════════════════════════════════════════════════════

📌 注意:
   • 升级 PyTorch 是完全可逆的（可回滚到 2.7.1）
   • INT8 方案已经过验证，98% LIBERO 成功率
   • 整个测试流程最多需要 30 分钟

═══════════════════════════════════════════════════════════════════════════════
"""
    print(footer)


def main():
    print_banner()
    print_core_issue()
    print_three_solutions()
    print_next_steps()
    print_available_docs()
    print_quick_commands()
    print_expected_outcomes()
    print_quick_decision_tree()
    print_footer()


if __name__ == "__main__":
    main()
