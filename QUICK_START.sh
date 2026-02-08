#!/bin/bash
# OpenPI TensorRT 快速启动指南

cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║         OpenPI TensorRT 推理 - 快速启动指南                  ║
╚═══════════════════════════════════════════════════════════════╝

📌 查看基准测试结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ./show_benchmark_results.sh

📌 启动 FP16 推理服务器（推荐 - 1.75x 加速）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  source /home/taco/.venv/bin/activate
  python3 scripts/serve_trt.py \
    --engine checkpoints/pi05_libero_onnx_compat/model.fp16.trt.engine \
    --port 8000

📌 启动 FP32 推理服务器（baseline）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  source /home/taco/.venv/bin/activate
  python3 scripts/serve_trt.py \
    --engine checkpoints/pi05_libero_onnx_compat/engine_fp32_cumsum_cast.trt \
    --port 8000

📌 运行单次推理测试
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 修改 test_trt_inference.py 中的端口号为 8000
  python3 test_trt_inference.py

📌 运行完整 LIBERO 基准测试
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # FP16（推荐）
  ./run_benchmark_fp16.sh
  
  # FP32
  ./run_benchmark_fp32.sh

📌 关键文件位置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ├─ 基准测试结果:  docs/benchmarks/BENCHMARK_RESULTS.md
  ├─ 完成总结:       COMPLETION_SUMMARY.md
  ├─ 待办事项:       todo.md
  ├─ FP16 引擎:      checkpoints/pi05_libero_onnx_compat/model.fp16.trt.engine
  └─ FP32 引擎:      checkpoints/pi05_libero_onnx_compat/engine_fp32_cumsum_cast.trt

📊 性能对比
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FP32:  313ms (baseline, 13GB)
  FP16:  179ms (1.75x faster, 6.1GB) ⚡ 推荐
  INT8:  编译失败（CumSum 问题）

🔧 调试工具
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 查看服务器日志
  tail -f /tmp/serve_trt_*.log
  
  # 查看基准测试日志
  tail -f /tmp/benchmark_*.txt
  
  # 检查进程
  ps aux | grep serve_trt
  
  # 清理进程
  pkill -f serve_trt

📖 更多信息
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  查看完整文档: cat COMPLETION_SUMMARY.md

╔═══════════════════════════════════════════════════════════════╗
║  提示: FP16 提供最佳性能，建议生产环境使用                   ║
╚═══════════════════════════════════════════════════════════════╝
EOF
