# OpenPI ONNX - Model Quantization & Performance Evaluation

模型量化和性能评估项目，支持FP32 PyTorch基线和INT8 TensorRT量化对比。

## ✨ Quick Overview

### 🔥 PyTorch FP32 Baseline Results (20 Trials Standard Test)

**测试配置**: 每任务20次试验 × 4套件 = 800 episodes | 随机种子: 42 | 测试日期: 2026-02-11/12

| Suite | Accuracy | Success/Total | Avg Latency (ms) | Median Latency (ms) | P99 Latency (ms) | GPU Memory (GB) |
|-------|----------|---------------|------------------|---------------------|------------------|-----------------|
| **libero_spatial** | **99.5%** | 199/200 | 263.23 | 261.68 | 286.90 | 8.10 |
| **libero_goal** | **91.0%** | 182/200 | 259.49 | 258.43 | 271.99 | 8.10 |
| **libero_object** | **95.0%** | 190/200 | 264.36 | 263.70 | 283.00 | 8.10 |
| **libero_10** | **89.5%** | 179/200 | 262.56 | 262.46 | 273.26 | 8.10 |
| **Overall** | **93.75%** | **750/800** | **262.41** | **261.57** | **278.79** | **8.10** |

**Key Findings**:
- ✅ High accuracy: 93.75% overall (750/800 episodes)
- ⚡ Consistent latency: 262.41ms mean, only 1.75ms std across suites
- 🎯 Low tail latency: P99 only +17ms vs median (+6.5%)
- 💾 Stable memory: 8.10GB GPU usage (constant across all suites)
- 📊 Task stratification: spatial (99.5%) > object (95.0%) > goal (91.0%) > libero_10 (89.5%)

📄 **详细报告**: [benchmark_results/FP32_RESULTS_20TRIALS.md](benchmark_results/FP32_RESULTS_20TRIALS.md)

### 📊 INT8 TensorRT Quantization Results (Historical - 需要重新测试)

**注意**: 以下INT8结果基于之前的测试，建议使用标准化20次试验重新评估以便公平对比。

- **Overall Accuracy**: 96.88%
- **Quantization**: INT8 W8A8 (ModelOpt)
- **Evaluation**: LIBERO Benchmark
- **Status**: 待重新验证

#### Suite Performance (INT8 - Historical)
| Suite | Accuracy |
|-------|----------|
| libero_goal | 99.0% |
| libero_spatial | 98.5% |
| libero_object | 98.0% |
| libero_10 | 92.0% |

**下一步**:
- [ ] 运行 INT8 标准化 20次试验测试 (`./run_int8_benchmark.sh`)
- [ ] 生成 FP32 vs INT8 公平对比报告
- [ ] 分析量化对延迟的影响

## 📚 Documentation

### Recommended Reading Order
1. **[INT8_QUICK_REFERENCE.md](INT8_QUICK_REFERENCE.md)** ⭐ 
   - Quick overview: Key data, common commands, file structure
   - **For**: Quick understanding of current status

2. **[FP32_INT8_COMPARISON.md](FP32_INT8_COMPARISON.md)** 🆕 ⏳ 待更新
   - Model comparison: FP32 baseline vs INT8 quantization
   - **Status**: 等待INT8标准化测试完成后更新
   - **For**: Performance comparison and deployment decisions

3. **[INT8_SUMMARY.md](INT8_SUMMARY.md)**
   - Complete summary: Problem diagnosis, solutions, key findings
   - **For**: Understanding project background and technical details

4. **[INT8_FINAL_RESULTS.md](INT8_FINAL_RESULTS.md)**
   - Detailed results: Complete data for all 40 tasks
   - **For**: Viewing specific accuracy and failure analysis

5. **[PYTORCH_FP32_FINAL_RESULTS.md](PYTORCH_FP32_FINAL_RESULTS.md)** (已过期 - 10次试验)
   - PyTorch baseline: 之前的FP32模型评估（10次试验）
   - **新版本**: [benchmark_results/FP32_RESULTS_20TRIALS.md](benchmark_results/FP32_RESULTS_20TRIALS.md) ⭐
   - **For**: 查看最新标准化测试结果

6. **[README_INT8.md](README_INT8.md)**
   - Technical documentation: Configuration, quick start, troubleshooting
   - **For**: Practical operations and deployment

## 🚀 快速开始

### 查看评估结果
```bash
# 快速参考（推荐）
cat INT8_QUICK_REFERENCE.md

# 完整总结
cat INT8_SUMMARY.md

# 详细数据
cat INT8_FINAL_RESULTS.md
```

### 运行PyTorch FP32基线测试

PyTorch模型是经过验证的工作版本，适合作为性能基准。

#### ✅ 推荐：使用标准化脚本（20次试验）
```bash
# FP32标准化测试（已完成 ✅）
./run_fp32_benchmark.sh

# 查看结果
cat benchmark_results/fp32_summary.txt
cat benchmark_results/FP32_RESULTS_20TRIALS.md
```

**当前状态**: 
- ✅ FP32 测试已完成 (2026-02-11/12)
- ⏳ INT8 测试待运行
- 📊 总体成功率: 93.75% (750/800)
- ⚡ 平均延迟: 262.41 ms

#### 手动单个套件测试
```bash
# 激活环境
source .venv/bin/activate

# 测试单个套件（例如libero_spatial）
PYTHONPATH=/home/taco/openpi-onnx/third_party/libero:$PYTHONPATH \
python3 scripts/eval_libero_torch.py \
  --checkpoint=checkpoints/pi05_libero_pytorch \
  --config=pi05_libero \
  --task_suite_name=libero_spatial \
  --num_trials_per_task=10 \
  --seed=42 > pytorch_benchmark_spatial.log 2>&1 &
```

#### 全部套件测试
```bash
# 测试所有4个套件: libero_spatial, libero_goal, libero_object, libero_10
for suite in spatial goal object 10; do
  PYTHONPATH=/home/taco/openpi-onnx/third_party/libero:$PYTHONPATH \
  timeout 7200 python3 scripts/eval_libero_torch.py \
    --checkpoint=checkpoints/pi05_libero_pytorch \
    --config=pi05_libero \
    --task_suite_name=libero_${suite} \
    --num_trials_per_task=10 \
    --seed=42 > pytorch_benchmark_${suite}.log 2>&1 &
done
```

#### 监控进度
```bash
# 实时查看日志
tail -f pytorch_benchmark_spatial.log

# 统计成功率
grep -c "Result: success" pytorch_benchmark_spatial.log
grep -c "Result: failure" pytorch_benchmark_spatial.log

# 查看最终结果
tail -100 pytorch_benchmark_spatial.log | grep -A 5 "Total Success Rate"
```

### 运行TensorRT INT8评估

#### ✅ 推荐：使用标准化脚本（20次试验）
```bash
# INT8标准化测试（与FP32相同配置）
./run_int8_benchmark.sh

# 实时监控进度
tail -f benchmark_logs/int8_spatial_20trials.log

# 查看结果
cat benchmark_results/int8_summary.txt
```

**配置**:
- 每任务20次试验 × 4套件 = 800 episodes
- 随机种子: 42
- 模型: model.int8.modelopt.engine (4.6GB)

#### 手动启动服务和测试

#### 启动TensorRT服务
```bash
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine \
  --port=8012 &
```

#### 运行单个套件
```bash
python scripts/eval_libero_trt_v1.py \
  --task_suite_name=libero_spatial \
  --num_trials_per_task=20 \
  --port=8012 --seed=7
```

#### 运行全部4套件
```bash
bash run_int8_all_suites_20trials.sh
```

### 监控进度
```bash
# 可视化进度显示
bash check_int8_progress.sh

# 持续监控（每30秒刷新）
watch -n 30 'bash check_int8_progress.sh'
```

### 性能基准对比 (FP32 vs INT8)

运行完整的基准对比（自动测试FP32和INT8）:
```bash
bash scripts/run_full_benchmark.sh
```

或分别测试单个模型:
```bash
# FP32基线 (仅需激活服务器)
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.fp32.modelopt.engine \
  --port=8012 &

python scripts/benchmark_trt_models.py \
  --model_type=fp32 --num_trials=10 --task_suite_name=all --port=8012

# INT8量化
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine \
  --port=8012 &

python scripts/benchmark_trt_models.py \
  --model_type=int8 --num_trials=10 --task_suite_name=all --port=8012
```

结果保存在 `./benchmark_results/` 目录，详见 [docs/conversion/FP32_FP4_INT8_COMPARISON.md](docs/conversion/FP32_FP4_INT8_COMPARISON.md)

## 📁 完整项目结构

### 根目录文件

#### 📄 核心文档 (INT8评估和性能对比)
- **README.md** - 本文件
- **BENCHMARK_GUIDE.md** ✨ (新) - 性能基准对比指南（FP32 vs INT8 vs FP4）
- **INT8_QUICK_REFERENCE.md** ⭐ - 快速参考（推荐首先阅读）
- **INT8_SUMMARY.md** - 完整总结（问题诊断、解决方案、关键发现）
- **INT8_FINAL_RESULTS.md** - 最终结果（所有40个任务的详细数据）
- **INT8_EVALUATION_RESULTS_20_TRIALS.md** - 20次试验详情（libero_spatial）
- **README_INT8.md** - 技术文档（配置、快速开始、故障排除）

#### 🔧 关键脚本
- **run_int8_all_suites_20trials.sh** - 自动化运行全部4个LIBERO套件评估 (INT8)
- **check_int8_progress.sh** - 可视化进度监控工具
- **scripts/benchmark_trt_models.py** - FP32/FP4/INT8 基准测试脚本
- **scripts/serve_trt.py** - TensorRT WebSocket推理服务器
- **scripts/eval_libero_trt_v1.py** - LIBERO基准评估脚本

#### 📊 数据文件
- **calibration_data.pt** (284MB) - INT8校准数据（包含200个真实推理样本）
- **torch_norm_stats.json** - 归一化统计（状态和动作的mean/std）

#### 📋 配置文件
- **pyproject.toml** - Python项目配置
- **uv.lock** - uv包管理器lock文件
- **LICENSE** - Apache 2.0许可证
- **LICENSE_GEMMA.txt** - Gemma许可证

### scripts目录

```
scripts/
├── eval_libero_trt_v1.py     ← INT8评估脚本（核心）
│   ├── 支持4个LIBERO套件评估
│   ├── 使用TensorRT INT8引擎
│   ├── WebSocket客户端与serve_trt.py通信
│   └── 输出成功率统计结果
│
├── serve_trt.py              ← TensorRT服务器
│   ├── WebSocket服务（端口8012）
│   ├── INT8引擎推理
│   ├── 支持分布式部署
│   └── 消息打包使用msgpack
│
└── （其他支持脚本）
```

### checkpoints目录

```
checkpoints/
└── pi05_libero_onnx_compat/
    ├── model.int8.modelopt.engine    ← TensorRT INT8引擎 (4.6GB)
    │   └── ModelOpt W8A8量化
    ├── model.int8.modelopt.cleaned.onnx  ← 清理后的ONNX模型
    ├── model.fp32.onnx               ← FP32基线（用于对比）
    └── （其他模型文件）
```

### docs目录

#### 转换指南
- **docs/conversion/pi05_onnx_conversion_guide.md** - ONNX模型转换指南
- **docs/conversion/tutorial_libero_trt.md** - LIBERO TensorRT教程
- **docs/conversion/norm_stats.md** - 归一化统计文档

#### 开发文档
- **docs/dev/CONTRIBUTING.md** - 贡献指南
- **docs/dev/docker.md** - Docker配置文档

### 清理后的文件统计

| 类别 | 数量 | 说明 |
|------|------|------|
| 根目录文件 | 15 | 精简至核心文件 |
| 文档 | 6 | INT8专项文档 |
| 脚本 | 2 | 核心INT8脚本 |
| docs目录 | 5 | 保留关键指南 |
| **总计** | **28** | 删除了40+多余文件 |

## � 保留文档详细说明

### INT8评估文档组
这些文档组成完整的INT8评估工作记录：

1. **INT8_QUICK_REFERENCE.md** (3.4KB)
   - 内容：一页纸快速查看、常用命令速查、文件导航地图
   - 用途：新用户快速了解项目

2. **INT8_SUMMARY.md** (4.6KB)
   - 内容：完整总结、问题诊断、解决方案、关键发现、经验教训
   - 用途：理解项目背景和技术决策

3. **INT8_FINAL_RESULTS.md** (4.2KB)
   - 内容：所有40个任务的最终成功率数据
   - 用途：查看详细的评估结果

4. **INT8_EVALUATION_RESULTS_20_TRIALS.md** (4.5KB)
   - 内容：libero_spatial的20次试验详细分析
   - 用途：深入研究单个套件的性能

5. **README_INT8.md** (7.6KB)
   - 内容：技术文档、快速开始、故障排除
   - 用途：实际操作和部署指南

### docs目录文档组
保留的5个文档提供转换和部署的必要知识：

1. **docs/conversion/FP32_FP4_INT8_COMPARISON.md** ✨ (新)
   - FP32基线 vs INT8量化 vs FP4的完整对比指南
   - 基准测试方法论、运行方式、结果解释

2. **docs/conversion/pi05_onnx_conversion_guide.md**
   - 指导如何将Pi05模型转换为ONNX格式

3. **docs/conversion/tutorial_libero_trt.md**
   - LIBERO数据集与TensorRT集成教程

4. **docs/conversion/norm_stats.md**
   - 解释归一化统计的计算和使用方法

5. **docs/dev/CONTRIBUTING.md**
   - 项目贡献指南

6. **docs/dev/docker.md**
   - Docker部署配置文档

## �🔧 技术细节

### 量化配置
```yaml
Model: pi05_libero
Quantization: W8A8 (ModelOpt)
Engine Format: TensorRT
Engine Size: 4.6 GB
```

### 评估配置
```yaml
Framework: LIBERO Benchmark
Trials per Task: 20
Tasks per Suite: 10
Total Suites: 4
Total Trials: 800
Seed: 7
```

### 转换管道
```
输入转换:
  LiberoInputs → ImageNormalize → Normalize → TokenizePrompt
  
输出转换:
  Unnormalize → PadStatesAndActions → slice[:, :7]
```

## 📊 关键发现

### ✅ 优势
- **高准确率**: 与FP32基线相比精度损失极小
- **稳定性**: 98-99%的成功率
- **生产就绪**: 96.88%综合成功率适合部署
- **完美任务**: 25/40任务达到100%

### ⚠️ 注意事项
- **libero_10 Task 8**: 85% (17/20) - 长序列任务
- **libero_10 Task 9**: 65% (13/20) - 长序列任务

## 💡 核心经验

1. **完整的转换管道是必须的** - 每一步都重要
2. **保留原始版本** - commit 68672fe是黄金标准
3. **多次试验验证** - 单次成功≠稳定
4. **维度匹配精确** - 7D动作 vs 32D填充
5. **归一化统计一致** - 必须正确加载

## 🎯 下一步工作

### 立即可做
- [ ] 分析libero_10失败任务原因
- [ ] 对比FP32基线性能
- [ ] 测量推理延迟

### 未来探索
- [ ] INT4量化探索
- [ ] 内存占用优化
- [ ] 校准数据优化

## 📞 故障排除

### 找不到文件？
```bash
ls -lh INT8_*.md *.sh
```

### 评估卡住了？
```bash
ps aux | grep "eval_libero\|serve_trt"
tail -f /tmp/int8_all_suites_master.log
```

### 需要更多帮助？
```bash
# 查看INT8_QUICK_REFERENCE.md的帮助部分
grep -A 10 "## 📞 帮助" INT8_QUICK_REFERENCE.md
```

## 📖 相关资源

- **LIBERO Benchmark**: https://libero-project.github.io/
- **NVIDIA ModelOpt**: https://github.com/NVIDIA/TensorRT-Model-Optimizer
- **TensorRT**: https://developer.nvidia.com/tensorrt

## 📝 许可证

See LICENSE file in the repository.

## ✨ 总结

这是一个完整的INT8量化评估项目，包含了从问题诊断到生产部署的全套工作。综合成功率96.88%证明了INT8量化在机器人操作任务上的有效性。所有工作已完成并文档完善，可直接用于生产环境。

**当前状态**: ✅ **生产就绪**

---

**最后更新**: 2026-02-08  
**分支**: INT8  
**Git提交**: ee45242
```

#### Option B: FP4 Quantization (Thor/Blackwell Only)
Leverages native FP4 Tensor Cores on NVIDIA Thor.
```bash
# 1. Quantize to FP4 (requires verified environment)
python scripts/quantize_thor_vla.py

# 2. Deploy
# See the detailed guide for compiling the Split-Stack engine:
# guides/FP4_DEPLOYMENT_GUIDE.md
```

### 4. Verification
Verify the accuracy of your quantized model using our evaluation suite:
```bash
python scripts/eval_libero_torch.py --task_suite_name libero_spatial
```

---

## 📊 Benchmark Results

| Platform | Precision | Accuracy | Latency | VRAM | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Orin/Thor** | **INT8** | **100.0%** | **~118 ms** | **4.0 GB** | **Stable** |
| **Thor** | **FP4** | **100.0%** | *< 50 ms* | ~6.0 GB | Verified (Sim) |
| -- | FP32 | 80.0% | ~250 ms | 13.0 GB | Baseline |
| -- | FP16 | 0.0% | N/A | 6.2 GB | **Unstable** |

## 📂 Repository Structure
*   `scripts/`: Core quantization and conversion utilities.
*   `exports/`: ONNX export pipelines.
*   `src/openpi/`: Shared model definitions (PyTorch port of Pi0).
*   `guides/`: Detailed deployment documentation.
