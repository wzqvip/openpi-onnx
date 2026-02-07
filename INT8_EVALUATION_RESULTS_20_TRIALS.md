# INT8 模型评估结果 (20次试验)

**评估日期**: 2026年2月7日  
**评估脚本**: commit 68672fe 原版脚本 (eval_libero_trt_v1.py)  
**任务套件**: libero_spatial  
**试验次数**: 每任务20次  
**总试验数**: 200次 (10任务 × 20次)

---

## 📊 详细结果

| Task ID | 任务描述 | 成功次数 | 成功率 |
|---------|---------|---------|--------|
| Task 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 20/20 | **100.00%** ✅ |
| Task 1 | pick up the black bowl next to the ramekin and place it on the plate | 20/20 | **100.00%** ✅ |
| Task 2 | pick up the black bowl from table center and place it on the plate | 20/20 | **100.00%** ✅ |
| Task 3 | pick up the black bowl on the cookie box and place it on the plate | 20/20 | **100.00%** ✅ |
| Task 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 18/20 | **90.00%** ⚠️ |
| Task 5 | pick up the black bowl on the ramekin and place it on the plate | 20/20 | **100.00%** ✅ |
| Task 6 | pick up the black bowl next to the cookie box and place it on the plate | 20/20 | **100.00%** ✅ |
| Task 7 | pick up the black bowl on the stove and place it on the plate | 20/20 | **100.00%** ✅ |
| Task 8 | pick up the black bowl next to the plate and place it on the plate | 20/20 | **100.00%** ✅ |
| Task 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 19/20 | **95.00%** ✅ |

---

## 🎯 总体统计

- **总成功次数**: 197/200
- **平均成功率**: **98.50%** ✅
- **完美任务数**: 8/10 (80%)
- **高成功率任务**: 10/10 (90%以上)

---

## 🔍 分析

### 成功率分布
- **100%成功**: 8个任务
- **95-99%成功**: 1个任务 (Task 9)
- **90-94%成功**: 1个任务 (Task 4)

### 失败案例
- **Task 4**: 2次失败 (可能因为抽屉开合动作复杂)
- **Task 9**: 1次失败 (放置在柜子上的碗，可能位置较高)

### 关键成功因素
✅ 使用原版 eval_libero_trt_v1.py (commit 68672fe)  
✅ 完整的 OpenPI transform pipeline  
✅ 正确的状态归一化和动作反归一化  
✅ INT8 ModelOpt 引擎 (4.6GB)  
✅ 真实校准数据 (calibration_data.pt, 284MB)

---

## 🚀 模型配置

- **引擎文件**: `/home/taco/checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine`
- **引擎大小**: 4.6 GB
- **量化方式**: ModelOpt W8A8 (权重INT8, 激活INT8)
- **校准数据**: 真实推理数据采集
- **TensorRT版本**: 10.x
- **CUDA版本**: 12.x

---

## ⏱️ 性能指标

- **评估总耗时**: ~2小时 (200次试验)
- **平均单次推理**: ~20-30秒/episode
- **TRT服务延迟**: ~150-200ms/inference
- **GPU利用率**: 中等 (推理瓶颈在环境交互)

---

## 📈 与单次试验对比

| 指标 | 单次试验 (1 trial) | 20次试验平均 |
|------|-------------------|-------------|
| **成功率** | 100% (10/10) | 98.50% (197/200) |
| **完美任务** | 10/10 | 8/10 |
| **稳定性** | 未知 | 优秀 |

**结论**: 20次试验显示INT8模型非常稳定，平均成功率高达98.5%，仅有3次失败案例。

---

## 🔧 技术细节

### Transform Pipeline
```python
input_transforms = [
    LiberoInputs(model_type=ModelType.PI05),
    ImageNormalize(),  # uint8[0,255] → float32[-1,1]
    Normalize(norm_stats, use_quantiles=True),  # 状态归一化
    InjectDefaultPrompt(),
    TokenizePrompt(),
]

output_transforms = [
    Unnormalize(action_stats),  # 动作反归一化
    PadStatesAndActions(action_dim=32),
]
```

### 归一化统计
- **State**: 8维 (eef_pos:3, eef_angle:3, gripper:2)
- **Actions**: 7维 (delta_pos:3, delta_angle:3, gripper:1)
- **Padding**: 模型输出32维，使用前7维

---

## 📝 下一步

1. ✅ **libero_spatial 完成** (98.50%)
2. ⏳ **libero_object 待评估** (20次试验)
3. ⏳ **libero_goal 待评估** (20次试验)
4. ⏳ **libero_10 待评估** (20次试验)
5. ⏳ **FP4 模型对比评估**

---

## 🎓 经验总结

### 成功要素
1. **完整的数据预处理流程** - 不能简化transform pipeline
2. **真实的校准数据** - 从实际推理中收集
3. **正确的归一化/反归一化** - 状态输入和动作输出都要处理
4. **原版评估脚本** - 保持与训练时的一致性

### 问题排查
- ❌ 简化版脚本缺少状态归一化 → 0-23%成功率
- ✅ 恢复原版脚本 → 98.5%成功率
- ✅ 修复torch.load weights_only问题

---

**日志文件**: `/tmp/eval_int8_original_libero_spatial_20260207_010808.log`  
**Git分支**: `INT8`  
**Commit**: fff9647
