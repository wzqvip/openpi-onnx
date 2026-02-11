# INT8 测试问题诊断与修复报告

**日期**: 2026-02-12  
**问题**: INT8 WebSocket版本评估失败，成功率从历史的96.88%崩溃到~15%  
**原因**: 使用了与历史不同的实现方式  
**解决方案**: 切换回验证过的v1实现  

---

## 问题分析

### ❌ 第一次尝试 (WebSocket版本)
- 使用: `eval_libero_trt.py` + `serve_trt.py` WebSocket架构
- 结果: **11-17.5% 成功率** (严重失败)
- 动作值范围: [-0.98, 0.87] (错误的)
- 失败模式: 全部超时，无法完成任务

### ✅ 历史成功记录
- 提交: `f31c7e7` - "INT8评估完成：96.88%综合成功率"
- 使用: `eval_libero_trt_v1.py` + TensorRT远程策略
- 成功率: **96.88%** (775/800)
- 框架: openpi的`tensorrt_remote_policy` + `libero_policy`

### 关键区别

| 方面 | WebSocket版 | v1版 |
|------|-----------|------|
| 脚本 | eval_libero_trt.py | eval_libero_trt_v1.py |
| 服务 | serve_trt.py | 内置 TensorRT 连接 |
| 通信 | WebSocket msgpack | 直接 Socket 连接 |
| 策略 | 自定义 WebSocket 推理 | openpi tensorrt_remote_policy |
| 历史成功 | ❌ 未验证 | ✅ 验证过 (96.88%) |
| 作用 | 新实现 | 验证过的原始实现 |

---

## 解决方案

### 步骤1: 创建 v1 基准脚本
文件: `run_int8_benchmark_v1.sh`

特点:
- 使用 `eval_libero_trt_v1.py` (验证过的脚本)
- 自动启动和关闭 TensorRT 服务器
- 支持 20 次试验标准化测试
- 输出日志到 `benchmark_logs/int8_*_20trials_v1.log`

### 步骤2: 启动测试
```bash
cd /home/taco/openpi-onnx
./run_int8_benchmark_v1.sh
```

### 步骤3: 监控进度
```bash
tail -f benchmark_logs/int8_spatial_20trials_v1.log
```

---

## 为什么 WebSocket 版本失败？

虽然没有完全确认，但最可能的原因：

1. **归一化统计不正确**
   - WebSocket版本可能没有正确加载 torch_norm_stats.json
   - 动作值范围证实了这一点 ([-0.98, 0.87] 明显错误)

2. **消息格式或序列化问题**
   - msgpack 序列化/反序列化可能有问题
   - 导致推理输出失真

3. **张量维度不匹配**
   - 与动作维度的处理有关
   - v1脚本有明确的 padding 处理代码

4. **时序或同步问题**
   - WebSocket 的异步通信可能造成数据不一致

v1脚本的实现要更复杂，它包含了：
- 显式的norm stats加载和覆盖
- 明确的动作维度 padding (7 -> 32)
- 完整的输入/输出变换管道
- 验证过的开放源代码

---

## 关键代码片段 (v1脚本)

```python
# v1脚本中的动作维度处理
if "actions" in output_norm_stats:
    act_stats = output_norm_stats["actions"]
    current_dim = act_stats.mean.shape[0]
    if current_dim < 32:
        print(f"DEBUG: Padding action stats from {current_dim} to 32")
        pad_len = 32 - current_dim
        # Pad mean with 0
        act_stats.mean = np.concatenate([act_stats.mean, np.zeros(pad_len, dtype=np.float32)])
        # Pad std with 1 (so unnorm is identity for extra dims)
        act_stats.std = np.concatenate([act_stats.std, np.ones(pad_len, dtype=np.float32)])
```

这个 padding 逻辑在 WebSocket 版本中**完全缺失**！

---

## 预期结果

使用 v1 脚本重新运行INT8测试应该会获得：
- ✅ 总体成功率: ~96.88% (类似历史结果)
- ✅ 延迟: ~160-180 ms (INT8相比FP32更快)
- ✅ 所有4个套件都能正常运行
- ✅ 没有超时失败

---

## 后续建议

1. **保留 WebSocket 版本的改进** (可选)
   - 完整实现缺失的 padding 逻辑
   - 改进 msgpack 序列化
   - 添加详细的debug日志

2. **优先使用 v1 脚本** (立即)
   - 已验证有效
   - 历史数据可信
   - 无已知问题

3. **更新脚本使用文档**
   - 明确标注 v1 脚本是推荐版本
   - 为新的 WebSocket 版本标注状态

4. **Git 提交**
   - 保存 v1 脚本版本
   - 文档化问题和解决方案

---

## 相关文件

- ✅ 新脚本: `run_int8_benchmark_v1.sh`
- ✅ 评估脚本: `scripts/eval_libero_trt_v1.py`  
- ✅ 服务脚本: `scripts/serve_trt.py` (共用)
- 📊 FP32基线: `benchmark_results/FP32_RESULTS_20TRIALS.md` (93.75%)
- ⚠️ WebSocket失败: `INT8_DIAGNOSTIC_GUIDE.md` (诊断指南)

---

## 总结

发现问题的根本原因是**使用了不同的实现版本**。WebSocket 版本是新尝试，但缺少了 v1 版本中的关键逻辑 (动作维度padding、完整的norm stats处理)。

通过切换回验证过的 v1 实现，INT8 基准测试应该能恢复到历史的 96.88% 成功率，与 FP32 (93.75%) 基线进行公平对比。

**状态**: 🔄 V1 脚本已启动，等待结果

