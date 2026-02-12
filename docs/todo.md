# TODO

## ✅ Completed
- [x] Setup venv and install PyTorch
- [x] Clone OpenPI and install dependencies
- [x] Download Libero dataset
- [x] Download OpenPI Checkpoint
- [x] Run conversion script `convert_jax_model_to_pytorch.py`
-- [x] **TensorRT benchmark completed**
   - [x] Build FP32 TRT engine (13GB, fixed CumSum Cast issue)
   - [x] Test FP16 TRT engine (6.1GB, prebuilt)
   - [x] Run LIBERO Spatial eval (10 tasks)
   - [x] Fix 20+ inference server issues:
      - [x] ctypes CUDA return types
      - [x] missing return in cudaStreamSynchronize
      - [x] TRT output buffer allocation errors
      - [x] dynamic shape handling
      - [x] WebSocket stability
   - [x] Document benchmark results

## 📊 Benchmark Results Summary
- **FP16 vs FP32**: 1.75x speedup (179ms vs 313ms)
- **Memory savings**: 47% (6.1GB vs 13GB)
- **P99 latency**: 38% improvement (203ms vs 329ms)

## ⏳ Open Issues

### High Priority
1. **INT8 build failed** ❌
    - Issue: CumSum operator dtype mismatch
    - Attempt: create patch_cumsum_cast.py to insert Cast node
    - Status: TRT parser still rejects
    - Next steps:
       - [ ] Check whether ModelOpt has alternative INT8 export options
       - [ ] Try quantizing directly from PyTorch (skip ONNX)
       - [ ] Contact NVIDIA support for TRT ONNX parser logs

2. **Task success rate 0%** ⚠️
    - Symptom: FP32 and FP16 both at 0% success
    - Analysis: not a precision issue (both match)
    - Possible causes:
       - [ ] Checkpoint not trained for LIBERO
       - [ ] Inference hyperparameters need tuning (`replan_steps`, `temperature`)
       - [ ] Environment config mismatch with training
    - Next steps:
       - [ ] Inspect training datasets
       - [ ] Adjust inference parameters
       - [ ] Compare against original PyTorch inference

### Medium Priority
3. **FP4 quantization exploration** 🔍
   - [ ] Check whether NVIDIA ModelOpt supports FP4
   - [ ] If supported, export FP4 ONNX
   - [ ] Build FP4 TRT engine
   - [ ] Run benchmark comparisons

4. **Performance analysis** 📈
   - [ ] Use nsys to profile GPU utilization
   - [ ] Identify bottlenecks (compute vs memory bandwidth)
   - [ ] Test batch_size > 1 performance

### Low Priority
5. **Code cleanup**
   - [ ] Remove temporary debug logs
   - [ ] Refactor serve_trt.py (20+ iterative changes)
   - [ ] Add unit tests
   - [ ] Create a full CI/CD pipeline

## 📁 Key Files
- Benchmark results: `docs/benchmarks/BENCHMARK_RESULTS.md`
- View results: `./show_benchmark_results.sh`
- FP32 engine: `checkpoints/pi05_libero_onnx_compat/engine_fp32_cumsum_cast.trt`
- FP16 engine: `checkpoints/pi05_libero_onnx_compat/model.fp16.trt.engine`
- Inference server: `scripts/serve_trt.py`
- CumSum fix: `patch_cumsum_cast.py`

## 🎯 Near-Term Goals
1. Fix INT8 build (blocking)
2. Investigate task success rate issues (non-blocking, needs understanding)
3. Explore FP4 feasibility (optimization goal)
