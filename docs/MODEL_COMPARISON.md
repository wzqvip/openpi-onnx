# INT8 vs Fake FP4 模型对比与评估

**最后更新**: 2026年2月5日  
**评估目标**: 完整对比 INT8 和 Fake FP4 量化模型的准确率、延迟、内存使用

---

## 📊 模型清单

### INT8 模型 (ModelOpt 量化)
- **状态**: ✅ 完成导出与编译
- **位置**: `/home/taco/openpi-onnx/checkpoints/pi05_libero_onnx_compat/`
- **文件**:
  - ONNX: `model.int8.modelopt.cleaned.onnx` (43 MB)
  - 数据: `model.int8.modelopt.cleaned.data` (13 GB)
  - 引擎: `model.int8.modelopt.engine` (4.6 GB)
  - 校准数据: `../calibration_data.pt` (297 MB)
- **推理方式**: TensorRT (WebSocket 服务)
- **精度**: W8A8 (权重INT8, 激活INT8)
- **导出脚本**: `exports/export_modelopt_int8.py`
- **编译日志**: `build_int8.log`

### Fake FP4 模型 (Thor FP4 量化)
- **状态**: ✅ 完成量化
- **位置**: `/home/taco/openpi-onnx/checkpoints/pi05_libero_onnx_compat/thor_fp4_ckpt/`
- **文件**:
  - 检查点: `quantized_model.safetensors` (8.0 GB)
  - NVFP4 ONNX: `model.nvfp4.modelopt.gs_clean.onnx` (如存在)
- **推理方式**: PyTorch (CPU/GPU 直接推理)
- **精度**: FP4 (Fake Quantization - 模拟量化，实际仍为FP16/FP32)
- **量化范围**: 使用 `modelopt` 的 FP4 量化器

---

## 🧪 评估配置

### 任务套件
1. **libero_spatial** - 空间操作 (10任务)
2. **libero_object** - 物体操作 (10任务)
3. **libero_goal** - 目标导向 (10任务)
4. **libero_10** - 混合任务 (10任务)

### 评估参数
- **试验次数**: 3 trials per task (总 120 个测试)
- **超时**: 10 秒/推理 (WebSocket)
- **评估指标**:
  - 成功率 (Success Rate)
  - 平均延迟 (Mean Latency)
  - P99 延迟 (P99 Latency)
  - 内存使用 (GPU/CPU Memory)
- **环境**: Jetson Thor (Blackwell GPU, 128GB 统一内存)

---

## 📈 预期结果

| 指标 | INT8 (TensorRT) | FP4 (PyTorch) | 基准 (FP32) |
|------|-----------------|---------------|-----------|
| **准确率** | ~100% | ~100% | 100% |
| **平均延迟 (ms)** | 118-150 | 150-200 | 300-350 |
| **P99 延迟 (ms)** | 150-180 | 180-250 | 330-380 |
| **引擎大小** | 4.6 GB | 8.0 GB | 13.0 GB |
| **推理服务** | TRT WebSocket | PyTorch Direct | PyTorch Direct |

---

## 🚀 运行评估

### 第一步: 准备环境

```bash
cd /home/taco/openpi-onnx

# 确保虚拟环境激活
source .venv/bin/activate  # 或 .venv312/bin/activate
```

### 第二步: INT8 评估

#### 启动 TensorRT 推理服务

```bash
# 终端1: 启动服务 (WebSocket on port 8012)
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine \
  --port=8012 \
  --max_batch_size=1
```

**预期输出**:
```
Loading TensorRT engine: model.int8.modelopt.engine
Engine loaded successfully
Listening on ws://localhost:8012
```

#### 运行完整评估

```bash
# 终端2: 运行评估脚本
python scripts/eval_libero_trt.py \
  --task_suite_name=libero_spatial \
  --ws_url=ws://localhost:8012 \
  --num_trials=3 \
  --output_dir=logs/eval_int8_spatial_$(date +%Y%m%d_%H%M%S)

python scripts/eval_libero_trt.py \
  --task_suite_name=libero_object \
  --ws_url=ws://localhost:8012 \
  --num_trials=3 \
  --output_dir=logs/eval_int8_object_$(date +%Y%m%d_%H%M%S)

python scripts/eval_libero_trt.py \
  --task_suite_name=libero_goal \
  --ws_url=ws://localhost:8012 \
  --num_trials=3 \
  --output_dir=logs/eval_int8_goal_$(date +%Y%m%d_%H%M%S)

python scripts/eval_libero_trt.py \
  --task_suite_name=libero_10 \
  --ws_url=ws://localhost:8012 \
  --num_trials=3 \
  --output_dir=logs/eval_int8_10_$(date +%Y%m%d_%H%M%S)
```

**预期运行时间**: ~2 小时 (120 个测试 × 40-60秒/测试)

### 第三步: Fake FP4 评估

```bash
# PyTorch 直接推理，无需启动服务
python scripts/eval_fp4_torch.py \
  --checkpoint_path=checkpoints/pi05_libero_onnx_compat/thor_fp4_ckpt/quantized_model.safetensors \
  --task_suite_name=libero_spatial \
  --num_trials=3 \
  --output_dir=logs/eval_fp4_spatial_$(date +%Y%m%d_%H%M%S)

python scripts/eval_fp4_torch.py \
  --checkpoint_path=checkpoints/pi05_libero_onnx_compat/thor_fp4_ckpt/quantized_model.safetensors \
  --task_suite_name=libero_object \
  --num_trials=3 \
  --output_dir=logs/eval_fp4_object_$(date +%Y%m%d_%H%M%S)

python scripts/eval_fp4_torch.py \
  --checkpoint_path=checkpoints/pi05_libero_onnx_compat/thor_fp4_ckpt/quantized_model.safetensors \
  --task_suite_name=libero_goal \
  --num_trials=3 \
  --output_dir=logs/eval_fp4_goal_$(date +%Y%m%d_%H%M%S)

python scripts/eval_fp4_torch.py \
  --checkpoint_path=checkpoints/pi05_libero_onnx_compat/thor_fp4_ckpt/quantized_model.safetensors \
  --task_suite_name=libero_10 \
  --num_trials=3 \
  --output_dir=logs/eval_fp4_10_$(date +%Y%m%d_%H%M%S)
```

**预期运行时间**: ~2.5 小时

---

## 📊 结果收集与分析

### INT8 结果位置
```
logs/eval_int8_spatial_*/summary.txt
logs/eval_int8_object_*/summary.txt
logs/eval_int8_goal_*/summary.txt
logs/eval_int8_10_*/summary.txt
```

### FP4 结果位置
```
logs/eval_fp4_spatial_*/summary.txt
logs/eval_fp4_object_*/summary.txt
logs/eval_fp4_goal_*/summary.txt
logs/eval_fp4_10_*/summary.txt
```

### 生成对比报告

```bash
python tools/generate_comparison_report.py \
  --int8_results=logs/eval_int8_*/summary.txt \
  --fp4_results=logs/eval_fp4_*/summary.txt \
  --output=comparison_report_$(date +%Y%m%d_%H%M%S).md
```

---

## 🔧 故障排查

### INT8 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 准确率 0% | 状态未归一化 | 检查 `eval_libero_trt.py` 行 235-240 有无状态归一化 |
| WebSocket 超时 | 推理挂起 | 检查 `serve_trt.py` 是否正在运行，查看服务日志 |
| OOM 错误 | 内存不足 | 减少 batch_size 或关闭其他进程 |
| 引擎文件为 0 字节 | 编译失败 | 重新编译: `trtexec --onnx=model.int8.modelopt.cleaned.onnx --saveEngine=model.int8.modelopt.engine --int8` |

### FP4 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 找不到模型文件 | 路径错误 | 检查 `thor_fp4_ckpt/quantized_model.safetensors` 是否存在 |
| 量化器属性错误 | ONNX 导出问题 | 重新运行 FP4 量化导出脚本 |
| 内存溢出 | 批处理太大 | 降低 batch_size 到 1 |

---

## 📝 评估脚本模板

### INT8 评估脚本 (`eval_libero_trt.py` 关键部分)

```python
# 1. 加载归一化统计
norm_stats = json.load(open('checkpoints/pi05_libero_pytorch/assets/.../norm_stats.json'))
state_mean = np.array(norm_stats['state']['mean'])
state_std = np.array(norm_stats['state']['std'])
action_mean = np.array(norm_stats['actions']['mean'])
action_std = np.array(norm_stats['actions']['std'])

# 2. 归一化状态输入
state = np.concatenate([eef_pos, eef_angle, gripper_state])
state = (state - state_mean) / (state_std + 1e-6)

# 3. WebSocket 推理调用
async def call_trt_inference(state, image_tokens, prompt_tokens):
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({
            'state': state.tolist(),
            'image_tokens': image_tokens,
            'prompt_tokens': prompt_tokens
        }))
        result = await asyncio.wait_for(ws.recv(), timeout=10.0)
        return json.loads(result)

# 4. 反归一化动作输出
actions = result['actions'][:7]  # 取前7维
actions = actions * action_std + action_mean
```

### FP4 评估脚本 (`eval_fp4_torch.py` 关键部分)

```python
# 1. 加载 FP4 检查点
model = load_model_with_fp4_weights('thor_fp4_ckpt/quantized_model.safetensors')
model.eval()

# 2. 前向推理 (无额外的量化处理)
with torch.no_grad():
    actions = model(images, text_tokens, state)

# 3. 提取和反归一化
actions_7d = actions[:, :7]
actions_7d = actions_7d * torch.tensor(action_std) + torch.tensor(action_mean)
```

---

## 📋 评估检查清单

- [ ] INT8 TensorRT 引擎文件大小 > 1GB (正常)
- [ ] FP4 safetensors 文件大小 ~8GB (正常)
- [ ] `serve_trt.py` 成功加载引擎
- [ ] WebSocket 连接正常 (ws://localhost:8012)
- [ ] 状态归一化代码存在 (eval_libero_trt.py)
- [ ] 动作反归一化代码存在 (eval_libero_trt.py)
- [ ] 评估日志生成成功
- [ ] 四个任务套件都运行了
- [ ] 每个套件 3 trials × 10 tasks = 30 个测试
- [ ] 总体成功率 ≥ 50% (最少目标)

---

## 📚 参考文件

- INT8 导出: [exports/export_modelopt_int8.py](../exports/export_modelopt_int8.py)
- INT8 评估: [scripts/eval_libero_trt.py](../scripts/eval_libero_trt.py)
- TRT 服务: [scripts/serve_trt.py](../scripts/serve_trt.py)
- FP4 导出: [exports/export_fp4_torch.py](../exports/export_fp4_torch.py) (如存在)
- 规范化统计: `checkpoints/pi05_libero_pytorch/assets/norm_stats.json`
- 校准数据: `calibration_data.pt` (INT8 使用)

---

## 🎯 成功标志

✅ **INT8 模型**:
- 启动服务无错误
- 四个任务套件准确率 ≥ 50%
- 平均延迟 < 200ms
- 引擎加载和推理稳定

✅ **FP4 模型**:
- 模型加载无错误
- 四个任务套件准确率 ≥ 50%
- 平均延迟 < 250ms
- 推理稳定无 OOM

✅ **对比分析**:
- INT8 和 FP4 准确率差异 < 5%
- INT8 比 FP4 快 20-30%
- 两者都比 FP32 快显著
