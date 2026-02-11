# Quantization Calibration Explained

## What is Quantization?

**Quantization** is the process of converting a neural network's weights and activations from high-precision formats (like FP32 or FP16) to lower-precision formats (like INT8). This reduces:
- **Model size**: INT8 weights use 4x less memory than FP32
- **Memory bandwidth**: Smaller data transfers between memory and compute units
- **Inference latency**: INT8 operations can be faster on modern hardware

### Quantization Formats

| Format | Weights | Activations | Size Reduction | Speed Gain | Accuracy Impact |
|--------|---------|-------------|----------------|------------|-----------------|
| **FP32** | 32-bit float | 32-bit float | baseline | baseline | baseline |
| **FP16** | 16-bit float | 16-bit float | 50% | 1.2-1.5x | minimal |
| **W8A16** | 8-bit int | 16-bit float | 50% | 1.2-1.3x | low |
| **W8A8** | 8-bit int | 8-bit int | 75% | 1.5-2.0x | moderate |

## Why Calibration is Needed

When converting from floating-point to integer representation, we need to determine **scaling factors** that map the continuous range of floats to the discrete range of integers.

### The Problem

Consider converting a weight tensor with values in range `[-2.5, 3.7]` to INT8 (range `[-128, 127]`):

```
FP32 value: 2.5
INT8 range: -128 to 127

Without calibration: How do we map 2.5 to an integer?
```

### The Solution: Scaling Factors

We compute a **scale** and **zero-point** that define the mapping:

```python
# Quantization formula
quantized_value = round(float_value / scale) + zero_point

# Dequantization formula (for inference)
float_value = (quantized_value - zero_point) * scale
```

**Example:**
```
If scale = 0.02 and zero_point = 0:
  FP32: 2.5  →  INT8: round(2.5 / 0.02) = 125
  FP32: -2.5 →  INT8: round(-2.5 / 0.02) = -125
```

## How Calibration Works

### 1. **Collect Representative Data**

We run the model on **real input samples** from the target domain (Libero environment in our case):

```python
calibration_data = [
    {
        "image": <224x224x3 RGB image>,
        "wrist_image": <224x224x3 RGB image>,
        "state": <8-dim robot state>,
        "prompt": "pick up the black bowl..."
    },
    # ... 50 samples total
]
```

**Why real data?**
- The model sees the actual distribution of inputs it will encounter in production
- Ensures scaling factors are optimized for the real use case
- Random/synthetic data would lead to suboptimal quantization

### 2. **Forward Pass with Quantizers**

We insert **quantizer modules** into the model that:
- Monitor the range of values flowing through each layer
- Track min/max values for activations
- Compute optimal scaling factors

```python
# ModelOpt inserts 1377 quantizers in our model
mtq.quantize(wrapper, quant_cfg, forward_loop=calibrate_model)

# During calibration, each quantizer observes:
for sample in calibration_samples:
    output = model(sample)  # Quantizers record min/max values
```

### 3. **Compute Optimal Scales**

After seeing all calibration samples, each quantizer computes:

```python
# For each layer/tensor
observed_min = min(all_values_seen)
observed_max = max(all_values_seen)

# Compute scale to map [observed_min, observed_max] → [-128, 127]
scale = (observed_max - observed_min) / 255
zero_point = -round(observed_min / scale) - 128
```

### 4. **Apply Quantization**

The computed scales are embedded into the ONNX model:

```
Original:     weight_fp32 = [0.5, -1.2, 2.3, ...]
Quantized:    weight_int8 = [25, -60, 115, ...]  (using scale=0.02)
              scale = 0.02 (stored as metadata)
```

## Our Implementation

### Calibration Data Source

We use **64 real samples** collected from the Libero environment:

```python
# From collect_calibration_data.py
env = get_libero_env(task_suite="libero_spatial", task_id=0)
obs = env.reset()

calibration_data.append({
    "image": obs["agentview_rgb"],           # Base camera
    "wrist_image": obs["robot0_eye_in_hand_rgb"],  # Wrist camera
    "state": obs["robot0_eef_pos"],          # End-effector position
    "prompt": task.language                   # Task description
})
```

### Calibration Process

```python
def calibrate_model(wrapper, calibration_samples):
    """Run calibration forward loop"""
    wrapper.eval()
    with torch.no_grad():
        for sample in tqdm(calibration_samples, desc="Calibrating"):
            # Each forward pass updates quantizer statistics
            _ = wrapper(*sample)
    
    # After loop completes, quantizers compute final scales
```

**Performance:**
- 50 samples used (subset of 64 collected)
- ~16 seconds per sample
- Total calibration time: ~13 minutes

### Quantizer Configuration

**W8A16 (8-bit weights, 16-bit activations):**
```python
quant_cfg = {
    "*weight_quantizer": {"num_bits": 8, "axis": 0},    # Per-channel quantization
    "*input_quantizer": {"num_bits": 16, "axis": None},  # Per-tensor quantization
    "*output_quantizer": {"enable": False}               # Don't quantize outputs
}
```

**W8A8 (8-bit weights, 8-bit activations):**
```python
quant_cfg = {
    "*weight_quantizer": {"num_bits": 8, "axis": 0},
    "*input_quantizer": {"num_bits": 8, "axis": None},   # More aggressive!
    "*output_quantizer": {"enable": False}
}
```

## Why This Matters for OpenPI

### Without Proper Calibration

Previous attempts failed because:
```python
# Old approach: No calibration
mtq.quantize(wrapper, quant_cfg, forward_loop=None)
# Result: "Quantizer has not been calibrated" error
```

The quantizers had no data to compute scales from, so they couldn't determine how to map FP32 → INT8.

### With Proper Calibration

Our new approach:
```python
# Load real Libero data
calib_data = torch.load("calibration_data.pt")
calibration_samples = prepare_calibration_samples(calib_data, config, model)

# Run calibration
def forward_loop(model):
    calibrate_model(model, calibration_samples)

mtq.quantize(wrapper, quant_cfg, forward_loop=forward_loop)
# Result: 1377 quantizers calibrated with optimal scales
```

## Expected Benefits

### W8A16 Model
- **Size**: ~6 GB (vs 12.4 GB FP32) = **50% reduction**
- **Speed**: 20-30% faster inference
- **Accuracy**: 60-70% expected (vs 80% FP32)
- **Use case**: Good balance of compression and accuracy

### W8A8 Model
- **Size**: ~3 GB (vs 12.4 GB FP32) = **75% reduction**
- **Speed**: 30-40% faster inference
- **Accuracy**: 50-60% expected (vs 80% FP32)
- **Use case**: Maximum compression, acceptable accuracy loss

## Technical Deep Dive

### Per-Channel vs Per-Tensor Quantization

**Per-Tensor** (activations):
```python
# Single scale for entire tensor
scale = (max(tensor) - min(tensor)) / 255
quantized = round(tensor / scale)
```

**Per-Channel** (weights):
```python
# Different scale for each output channel
for channel in range(num_channels):
    scale[channel] = (max(weights[channel]) - min(weights[channel])) / 255
    quantized[channel] = round(weights[channel] / scale[channel])
```

Per-channel quantization is more accurate for weights because different channels may have different value distributions.

### Symmetric vs Asymmetric Quantization

**Symmetric** (zero_point = 0):
```python
scale = max(abs(min_val), abs(max_val)) / 127
quantized = round(value / scale)
# Range: [-127, 127] (not using -128 for symmetry)
```

**Asymmetric** (zero_point ≠ 0):
```python
scale = (max_val - min_val) / 255
zero_point = -round(min_val / scale) - 128
quantized = round(value / scale) + zero_point
# Range: [-128, 127] (full INT8 range)
```

ModelOpt uses **symmetric quantization** by default for better hardware compatibility.

## Comparison with Other Approaches

### Post-Training Quantization (PTQ) - What We're Doing
- ✅ No retraining required
- ✅ Fast (minutes to hours)
- ✅ Works with pre-trained checkpoints
- ❌ Some accuracy loss (5-30%)

### Quantization-Aware Training (QAT)
- ✅ Minimal accuracy loss (0-5%)
- ❌ Requires full retraining
- ❌ Slow (days to weeks)
- ❌ Needs training data and compute

### Dynamic Quantization
- ✅ No calibration needed
- ✅ Easy to apply
- ❌ Only quantizes weights (not activations)
- ❌ Less speedup (no activation quantization)

For our use case (deploying a pre-trained model), **PTQ with calibration** is the optimal choice.

## Summary

**Calibration is essential** because:
1. It determines how to map continuous floats to discrete integers
2. It ensures the quantization is optimized for the actual data distribution
3. It prevents accuracy degradation from poor scaling choices

**Our approach:**
1. Collect 64 real samples from Libero environment
2. Run 50 samples through the quantized model
3. Let quantizers observe value ranges and compute optimal scales
4. Export to ONNX with embedded quantization parameters
5. Build TensorRT engine that uses INT8 operations

This gives us **50-75% size reduction** with **acceptable accuracy loss** for deployment on resource-constrained devices like Jetson Thor.
