#!/bin/bash
set -e

mkdir -p data/libero/logs

PYTHON=/home/taco/miniconda3/envs/torch-py311/bin/python
export MUJOCO_GL=egl

tegrastats --interval 1000 --logfile data/libero/logs/tegrastats.log &
TEGRA_PID=$!
trap "kill $TEGRA_PID" EXIT


echo "Running Libero Spatial..."
PYTHONPATH=src $PYTHON scripts/eval_libero_torch.py --task_suite_name libero_spatial --video_out_path data/libero/videos_torch/spatial 2>&1 | tee data/libero/logs/spatial.log || true

echo "Running Libero Goal..."
PYTHONPATH=src $PYTHON scripts/eval_libero_torch.py --task_suite_name libero_goal --video_out_path data/libero/videos_torch/goal 2>&1 | tee data/libero/logs/goal.log || true

echo "Running Libero 10..."
PYTHONPATH=src $PYTHON scripts/eval_libero_torch.py --task_suite_name libero_10 --video_out_path data/libero/videos_torch/10 2>&1 | tee data/libero/logs/10.log || true

echo "All benchmarks completed."
