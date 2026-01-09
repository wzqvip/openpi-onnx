import os
import subprocess
import re
import sys
from pathlib import Path

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
BASE_DIR = Path("checkpoints/pi05_libero_pytorch")

# Define variants and their specific TRT flags
MODELS = {
    "FP32":  {"path": "fp32/model.onnx",  "flags": []},
    "FP16":  {"path": "fp16/model.onnx",  "flags": ["--fp16"]},
    "INT8":  {"path": "int8/model.onnx",  "flags": ["--int8", "--fp16"]},
    "INT4":  {"path": "int4/model.onnx",  "flags": ["--int8", "--fp16"]}, # Assuming INT4 uses INT8 kernels or is decompressed
    "NVFP8": {"path": "nvfp8/model.onnx", "flags": ["--fp8", "--fp16"]},
    "NVFP4": {"path": "nvfp4/model.onnx", "flags": ["--best"]}, # Hope --best picks up FP4 if available
}

def run_benchmark(name, config):
    model_path = BASE_DIR / config["path"]
    if not model_path.exists():
        print(f"[{name}] Skipping - File not found: {model_path}")
        return None

    cmd = [
        TRTEXEC,
        f"--onnx={model_path}",
        "--duration=10",  # Run for 10 seconds
        "--avgRuns=100",  # Average over 100 runs
        "--noDataTransfers", # Measure compute mostly, but we can enable if we want end-to-end
        # "--useCudaGraph", # Optional: might help small models
    ] + config["flags"]

    print(f"[{name}] Running: {' '.join(cmd)}")
    
    try:
        # Run trtexec and capture output
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            check=False 
        )
        
        output = result.stdout
        
        # Parse output
        metrics = {}
        
        # Parse Latency
        # "Mean Latency: 12.34 ms"
        lat_match = re.search(r"Mean Latency:\s+([\d.]+)\s+ms", output)
        if lat_match:
            metrics["latency_ms"] = float(lat_match.group(1))
            
        # Parse Throughput
        # "Throughput: 123.45 qps"
        thruh_match = re.search(r"Throughput:\s+([\d.]+)\s+qps", output)
        if thruh_match:
            metrics["throughput_qps"] = float(thruh_match.group(1))
            
        # Parse Memory
        # "Total Host Memory: 123.4 MiB"
        # "Total Device Memory: 456.7 MiB"
        mem_match = re.search(r"Total Device Memory:\s+([\d.]+)\s+MiB", output)
        if mem_match:
            metrics["gpu_mem_mib"] = float(mem_match.group(1))
        else:
             # Fallback to simple "GPU memory usage: ... MiB" if reported differently
            mem_match = re.search(r"GPU memory usage:\s+([\d.]+)\s+MiB", output)
            if mem_match:
                metrics["gpu_mem_mib"] = float(mem_match.group(1))

        if result.returncode != 0:
            print(f"[{name}] Failed with return code {result.returncode}")
            # print(output[-1000:]) # Print last 1000 chars of error
            return None
            
        return metrics

    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

def main():
    print("| Precision | Latency (ms) | Throughput (QPS) | GPU Mem (MiB) |")
    print("|-----------|--------------|------------------|---------------|")
    
    results = []
    
    for name, config in MODELS.items():
        metrics = run_benchmark(name, config)
        if metrics:
            row = f"| {name} | {metrics.get('latency_ms', 'N/A')} | {metrics.get('throughput_qps', 'N/A')} | {metrics.get('gpu_mem_mib', 'N/A')} |"
            print(row)
            results.append(row)
        else:
            print(f"| {name} | Failed | Failed | Failed |")
            results.append(f"| {name} | Failed | Failed | Failed |")
            
    # Save to file
    with open("BENCHMARK_RESULTS.md", "w") as f:
        f.write("# Model Benchmark Results\n\n")
        f.write("| Precision | Latency (ms) | Throughput (QPS) | GPU Mem (MiB) |\n")
        f.write("|-----------|--------------|------------------|---------------|\n")
        for line in results:
            f.write(line + "\n")

if __name__ == "__main__":
    main()
