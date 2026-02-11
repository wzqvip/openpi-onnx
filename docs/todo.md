# TODO

## ✅ 已完成
- [x] Setup venv and install PyTorch
- [x] Clone OpenPI and install dependencies
- [x] Download Libero dataset
- [x] Download OpenPI Checkpoint
- [x] Run conversion script `convert_jax_model_to_pytorch.py`
- [x] **TensorRT 基准测试完成**
    - [x] 编译 FP32 TRT 引擎（13GB，修复 CumSum Cast 问题）
    - [x] 测试 FP16 TRT 引擎（6.1GB，已存在）
    - [x] 运行 LIBERO Spatial 评测（10 tasks）
    - [x] 修复 20+ 个推理服务器问题：
        - [x] ctypes CUDA 返回类型
        - [x] cudaStreamSynchronize 缺少 return
        - [x] TRT 输出 buffer 分配错误
        - [x] 动态形状处理
        - [x] WebSocket 稳定性
    - [x] 文档化基准测试结果

## 📊 基准测试结果摘要
- **FP16 vs FP32**: 1.75x 加速（179ms vs 313ms）
- **内存节省**: 47%（6.1GB vs 13GB）
- **P99 延迟**: 38% 改善（203ms vs 329ms）

## ⏳ 待解决问题

### 高优先级
1. **INT8 编译失败** ❌
   - 问题：CumSum 算子 dtype 不兼容
   - 尝试：创建 patch_cumsum_cast.py 插入 Cast 节点
   - 状态：TRT parser 仍拒绝接受
   - 下一步：
     - [ ] 检查 ModelOpt 是否有其他 INT8 导出选项
     - [ ] 尝试直接从 PyTorch 量化（跳过 ONNX）
     - [ ] 联系 NVIDIA 支持获取 TRT ONNX parser 详细日志

2. **任务成功率 0%** ⚠️
   - 现象：FP32 和 FP16 都是 0% 成功率
   - 分析：不是精度问题（两个精度一致）
   - 可能原因：
     - [ ] 模型 checkpoint 未针对 LIBERO 训练
     - [ ] 推理超参数需调整（replan_steps, temperature）
     - [ ] 环境配置与训练不匹配
   - 下一步：
     - [ ] 检查模型训练数据集
     - [ ] 调整推理参数
     - [ ] 对比原始 PyTorch 推理结果

### 中优先级
3. **FP4 量化探索** 🔍
   - [ ] 检查 NVIDIA ModelOpt 是否支持 FP4
   - [ ] 如果支持，导出 FP4 ONNX 模型
   - [ ] 编译 FP4 TRT 引擎
   - [ ] 基准测试对比

4. **性能分析** 📈
   - [ ] 使用 nsys 分析 GPU 利用率
   - [ ] 识别瓶颈（计算 vs 内存带宽）
   - [ ] 测试 batch_size > 1 的性能

### 低优先级
5. **代码清理**
   - [ ] 移除临时调试日志
   - [ ] 整理 serve_trt.py（20+ 次迭代修改）
   - [ ] 添加单元测试
   - [ ] 创建完整的 CI/CD pipeline

## 📁 重要文件
- 基准测试结果: `docs/benchmarks/BENCHMARK_RESULTS.md`
- 结果查看: `./show_benchmark_results.sh`
- FP32 引擎: `checkpoints/pi05_libero_onnx_compat/engine_fp32_cumsum_cast.trt`
- FP16 引擎: `checkpoints/pi05_libero_onnx_compat/model.fp16.trt.engine`
- 推理服务器: `scripts/serve_trt.py`
- CumSum 修复: `patch_cumsum_cast.py`

## 🎯 近期目标
1. 修复 INT8 编译（阻塞项）
2. 调查任务成功率问题（非阻塞，但需要理解）
3. 探索 FP4 可行性（优化目标）
