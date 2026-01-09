import subprocess
import re

ENGINES = {
    "FP32": "model.fp32.engine",
    "FP16": "model.trt"
}

def run_bench(name, engine_path):
    print(f"\n--- Benchmarking {name} ({engine_path}) ---")
    cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--loadEngine={engine_path}",
        "--duration=10",
        "--avgRuns=100",
        "--noDataTransfers"
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = result.stdout
        
        # Parse
        lat = re.search(r"Mean Latency:\s+([\d.]+)\s+ms", output)
        thruh = re.search(r"Throughput:\s+([\d.]+)\s+qps", output)
        mem = re.search(r"Total Device Memory:\s+([\d.]+)\s+MiB", output)
        if not mem:
             mem = re.search(r"GPU memory usage:\s+([\d.]+)\s+MiB", output)
        
        l_val = lat.group(1) if lat else "N/A"
        t_val = thruh.group(1) if thruh else "N/A"
        m_val = mem.group(1) if mem else "N/A"
        
        print(f"| {name} | {l_val} | {t_val} | {m_val} |")
        
        if result.returncode != 0:
            print("Failed.")
            # print(output[-500:])

    except Exception as e:
        print(f"Error: {e}")

print("| Precision | Latency (ms) | Throughput (QPS) | GPU Mem (MiB) |")
print("|-----------|--------------|------------------|---------------|")

for name, path in ENGINES.items():
    run_bench(name, path)
