# INT8 Quick Reference 快速参考

## ✨ 一句话总结
**INT8量化模型在LIBERO benchmark上实现96.88%综合成功率，生产就绪。**

---

## 📊 关键数据（一目了然）

```
总体成功率: 96.88% ████████████████████░
成功试验:   775/800
完美任务:   25/40 (62.5%)
评估时间:   ~3小时
```

### 各套件成绩单
```
🥇 libero_goal    : 99.00% █████████████████████
🥈 libero_spatial : 98.50% ████████████████████░
🥉 libero_object  : 98.00% ████████████████████░
✅ libero_10      : 92.00% ██████████████████░░░
```

---

## 🚀 快速命令

### 查看结果
```bash
# 详细结果
cat INT8_FINAL_RESULTS.md

# 完整总结
cat INT8_SUMMARY.md

# 快速查看（本文件）
cat INT8_QUICK_REFERENCE.md
```

### 运行评估
```bash
# 1. 启动TRT服务器
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine \
  --port=8012 &

# 2. 运行单个套件
python scripts/eval_libero_trt_v1.py \
  --task_suite_name=libero_spatial \
  --num_trials_per_task=20

# 3. 运行全部4套件
bash run_int8_all_suites_20trials.sh
```

### 监控进度
```bash
# 可视化进度
bash check_int8_progress.sh

# 持续监控（每30秒刷新）
watch -n 30 'bash check_int8_progress.sh'
```

---

## 📁 文件地图

```
openpi-onnx/
├── INT8_QUICK_REFERENCE.md  ← 你在这里 📍
├── INT8_SUMMARY.md           ← 完整总结
├── INT8_FINAL_RESULTS.md     ← 详细结果（所有任务）
├── README_INT8.md            ← 完整文档
├── check_int8_progress.sh    ← 进度监控工具
├── run_int8_all_suites_20trials.sh  ← 自动化运行器
└── scripts/
    ├── eval_libero_trt_v1.py  ← 评估脚本
    └── serve_trt.py           ← TRT服务器
```

---

## 🔍 需要注意的

### ⚠️ 问题点
- **libero_10 Task 8**: 85% (17/20) - 需要review
- **libero_10 Task 9**: 65% (13/20) - 需要review

### ✅ 优势点
- **25个任务100%成功** - 超过半数完美
- **转换管道完整** - 状态归一化+动作反归一化
- **稳定性验证** - 800次试验证明稳定

---

## 💡 核心经验

1. **不要简化评估脚本** - 转换管道每一步都重要
2. **保留原始版本** - commit 68672fe是黄金标准
3. **多次试验验证** - 20次才能看出稳定性
4. **维度要精确** - 7D vs 32D必须明确
5. **归一化统计一致** - norm_stats必须正确

---

## 🎯 下一步

### 立即可做
- [ ] 分析libero_10 Task 8-9失败原因
- [ ] 对比FP32基线性能
- [ ] 测量推理延迟

### 未来探索
- [ ] INT4量化尝试
- [ ] 内存占用优化
- [ ] 校准数据优化

---

## 📞 帮助

**找不到文件？**
```bash
ls -lh INT8_*.md check_int8_progress.sh
```

**评估卡住了？**
```bash
ps aux | grep "eval_libero\|serve_trt"
tail -f /tmp/int8_all_suites_master.log
```

**结果看不懂？**
```bash
# 阅读顺序：
# 1. INT8_QUICK_REFERENCE.md (本文件)
# 2. INT8_SUMMARY.md         (完整总结)
# 3. INT8_FINAL_RESULTS.md   (所有数据)
# 4. README_INT8.md          (技术细节)
```

---

**Git分支**: INT8  
**最后更新**: 2026-02-07  
**状态**: ✅ 生产就绪  

