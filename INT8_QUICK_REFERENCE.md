# INT8 Quick Reference

## ✅ One-line summary
**INT8 TensorRT achieved 98.25% overall success on LIBERO (786/800 episodes).**

## 📊 Key numbers (20 trials standard test)

- **Overall accuracy**: 98.25% (786/800)
- **Suites**:
  - libero_spatial: 99.00% (198/200)
  - libero_goal: 98.50% (197/200)
  - libero_object: 99.50% (199/200)
  - libero_10: 96.00% (192/200)
- **Episode wall time** (end-to-end): mean 10.43s, median 9.21s, p99 21.21s
- **GPU memory (INT8)**: 4954 MiB (~4.95 GB) with engine loaded

> Episode wall time is derived from tqdm “20/20” lines in the logs. Inference-only latency is not logged by `eval_libero_trt_v1.py`.

### Monitoring progress
```bash
# One-shot progress view
bash check_int8_progress.sh

# Refresh every 30 seconds
watch -n 30 'bash check_int8_progress.sh'
```

---

## 📁 File map

```
openpi-onnx/
├── INT8_QUICK_REFERENCE.md
├── INT8_SUMMARY.md
├── INT8_FINAL_RESULTS.md
├── README_INT8.md
├── check_int8_progress.sh
├── run_int8_all_suites_20trials.sh
└── scripts/
  ├── eval_libero_trt_v1.py
  └── serve_trt.py
```
**Last updated**: 2026-02-12
```bash

ps aux | grep "eval_libero\|serve_trt"
