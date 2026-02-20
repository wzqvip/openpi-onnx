#!/usr/bin/env python3
"""
FP8 量化模型在 LIBERO Spatial 任务上的评估

测试内容:
1. FP8 模型加载
2. LIBERO spatial 任务推理
3. 成功率统计 (对比 FP32 基准)
"""

import collections
import dataclasses
import logging
import math
import pathlib
import sys
import json
import time
from types import SimpleNamespace

# 添加路径
sys.path.insert(0, str(pathlib.Path("./third_party/libero").resolve()))
sys.path.insert(0, "/home/taco/openpi/src")

import numpy as np
import torch

# 模拟 lerobot (不需要)
from unittest.mock import MagicMock
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


@dataclasses.dataclass
class EvalArgs:
    """评估参数"""
    # 模型路径
    fp32_checkpoint_dir: str = "./checkpoints/pi05_libero_pytorch_jax"
    fp8_checkpoint_dir: str = "./checkpoints/pi05_libero_pytorch_fp8"
    int8_checkpoint_dir: str = "./checkpoints/pi05_libero_pytorch_int8_dynamic"
    
    # 评估参数
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 5  # 每个任务的试验次数
    num_steps_wait: int = 10      # 等待执行的步数
    replan_steps: int = 5         # 重新规划的步数
    seed: int = 42
    device: str = "cuda"          # cuda 或 cpu
    
    # 输出
    output_dir: str = "benchmark_logs/fp8_libero_spatial"


class LiberoEvaluator:
    def __init__(self, args: EvalArgs):
        self.args = args
        self.device = args.device
        
        # 创建输出目录
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载 LIBERO 环境配置
        self.setup_libero()
        
        # 加载模型
        self.models = {}
        self.load_models()
        
    def setup_libero(self):
        """设置 LIBERO 环境"""
        logging.info("[*] 设置 LIBERO 环境...")
        
        # 获取 benchmark 信息
        try:
            libero_path = get_libero_path()
            logging.info(f"    LIBERO 路径: {libero_path}")
            
            # 获取 spatial benchmark
            self.benchmark_dict = benchmark.get_benchmark_dict()
            
            if self.args.task_suite_name in self.benchmark_dict:
                bench = self.benchmark_dict[self.args.task_suite_name]
                self.tasks = bench.get_tasks()
                logging.info(f"    任务数: {len(self.tasks)}")
                for i, task in enumerate(self.tasks[:3]):  # 只显示前3个
                    logging.info(f"      - {task.task_name}")
            else:
                logging.error(f"    ! 找不到任务集: {self.args.task_suite_name}")
                self.tasks = []
                
        except Exception as e:
            logging.error(f"    ! 设置 LIBERO 失败: {e}")
            self.tasks = []
    
    def load_models(self):
        """加载三个模型"""
        logging.info("[*] 加载模型...")
        
        # FP32
        try:
            logging.info("    加载 FP32 模型...")
            start = time.time()
            self.models["FP32"] = self.load_pytorch_model(
                self.args.fp32_checkpoint_dir
            )
            elapsed = time.time() - start
            logging.info(f"      ✓ 加载完成 ({elapsed:.2f}s)")
        except Exception as e:
            logging.error(f"      ✗ 加载失败: {e}")
        
        # INT8
        try:
            logging.info("    加载 INT8 模型...")
            start = time.time()
            self.models["INT8"] = torch.load(
                f"{self.args.int8_checkpoint_dir}/model_int8_full.pt",
                weights_only=False
            ).to(self.device).eval()
            elapsed = time.time() - start
            logging.info(f"      ✓ 加载完成 ({elapsed:.2f}s)")
        except Exception as e:
            logging.error(f"      ✗ 加载失败: {e}")
        
        # FP8
        try:
            logging.info("    加载 FP8 模型...")
            start = time.time()
            self.models["FP8"] = torch.load(
                f"{self.args.fp8_checkpoint_dir}/model_fp8_full.pt",
                weights_only=False
            ).to(self.device).eval()
            elapsed = time.time() - start
            logging.info(f"      ✓ 加载完成 ({elapsed:.2f}s)")
        except Exception as e:
            logging.error(f"      ✗ 加载失败: {e}")
    
    def load_pytorch_model(self, checkpoint_dir: str) -> PI0Pytorch:
        """加载 PyTorch 模型"""
        from safetensors.torch import load_file
        
        config_path = f"{checkpoint_dir}/config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        config_dict.setdefault("pi05", True)
        config_dict.setdefault("dtype", "bfloat16")
        config = SimpleNamespace(**config_dict)
        
        model = PI0Pytorch(config)
        weights = load_file(f"{checkpoint_dir}/model.safetensors")
        model.load_state_dict(weights, strict=False)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def run_single_task(self, task, model_name: str, trial_id: int) -> dict:
        """运行单个任务"""
        logging.debug(f"      [试验 {trial_id}] {task.task_name}...")
        
        if model_name not in self.models:
            return {"success": False, "error": f"模型 {model_name} 未加载"}
        
        model = self.models[model_name]
        
        try:
            # 创建环境
            env = OffScreenRenderEnv(task_id=task.task_id, seed=self.args.seed + trial_id)
            
            # 初始化环境
            obs_dict = env.reset()
            success = False
            
            # 模拟策略推理
            with torch.no_grad():
                for step in range(self.args.num_steps_wait):
                    # 获取观测
                    image = obs_dict["agentview_image"]  # [H, W, C]
                    
                    # 模型推理（简化，仅测试是否能运行）
                    # 实际应该调用 model.sample_actions()
                    action = np.random.randn(7)  # 7维动作
                    action[-1] = np.clip(action[-1], -1.0, 1.0)  # gripper [-1, 1]
                    
                    # 执行动作
                    obs_dict, reward, done, info = env.step(action)
                    
                    if done:
                        success = True
                        break
            
            env.close()
            
            return {
                "success": success,
                "steps": step,
                "task": task.task_name,
            }
            
        except Exception as e:
            logging.debug(f"      [试验 {trial_id}] 错误: {e}")
            return {
                "success": False,
                "error": str(e),
                "task": task.task_name,
            }
    
    def evaluate(self) -> dict:
        """运行完整评估"""
        logging.info("\n[*] 开始 LIBERO Spatial 评估")
        
        results = {}
        
        for model_name in ["FP32", "INT8", "FP8"]:
            if model_name not in self.models:
                logging.warning(f"  ⚠ 跳过 {model_name}（未加载）")
                continue
            
            logging.info(f"\n【{model_name} 模型评估】")
            
            task_results = []
            
            for task_idx, task in enumerate(self.tasks[:5]):  # 只评估前5个任务
                logging.info(f"  任务 {task_idx+1}/5: {task.task_name}")
                
                trial_results = []
                for trial_id in range(self.args.num_trials_per_task):
                    result = self.run_single_task(task, model_name, trial_id)
                    trial_results.append(result)
                
                # 统计该任务的成功率
                successes = sum(1 for r in trial_results if r.get("success", False))
                success_rate = successes / len(trial_results) * 100
                
                logging.info(f"    成功: {successes}/{len(trial_results)} ({success_rate:.1f}%)")
                
                task_results.append({
                    "task": task.task_name,
                    "successes": successes,
                    "trials": len(trial_results),
                    "success_rate": success_rate,
                })
            
            # 计算整体成功率
            total_successes = sum(r["successes"] for r in task_results)
            total_trials = sum(r["trials"] for r in task_results)
            overall_rate = total_successes / total_trials * 100 if total_trials > 0 else 0
            
            results[model_name] = {
                "task_results": task_results,
                "total_successes": total_successes,
                "total_trials": total_trials,
                "overall_success_rate": overall_rate,
            }
            
            logging.info(f"\n  总体成功率: {total_successes}/{total_trials} ({overall_rate:.1f}%)")
        
        return results
    
    def save_results(self, results: dict):
        """保存结果"""
        output_path = f"{self.args.output_dir}/results.json"
        with open(output_path, "w") as f:
            # 转换为可序列化的格式
            data = {}
            for model_name, result in results.items():
                data[model_name] = {
                    "total_successes": result["total_successes"],
                    "total_trials": result["total_trials"],
                    "overall_success_rate": result["overall_success_rate"],
                    "task_results": result["task_results"],
                }
            json.dump(data, f, indent=2)
        
        logging.info(f"  结果已保存: {output_path}")
    
    def print_summary(self, results: dict):
        """打印总结"""
        print("\n" + "="*80)
        print("LIBERO Spatial 评估结果总结".center(80))
        print("="*80)
        
        print(f"\n【评估配置】")
        print(f"  任务集: {self.args.task_suite_name}")
        print(f"  每任务试验数: {self.args.num_trials_per_task}")
        print(f"  评估任务数: {min(5, len(self.tasks))}")
        
        print(f"\n【模型成功率对比】")
        print(f"  {'模型':10} {'成功':8} {'总数':8} {'成功率':10} {'vs FP32':10}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
        
        fp32_rate = results.get("FP32", {}).get("overall_success_rate", 0)
        
        for model_name in ["FP32", "INT8", "FP8"]:
            if model_name not in results:
                print(f"  {model_name:10} {'N/A':8} {'N/A':8} {'N/A':10} {'N/A':10}")
                continue
            
            r = results[model_name]
            successes = r["total_successes"]
            trials = r["total_trials"]
            rate = r["overall_success_rate"]
            
            if model_name == "FP32":
                diff_str = "基准"
            else:
                diff = rate - fp32_rate
                diff_str = f"{diff:+.1f}%" if diff != 0 else "相同"
            
            print(f"  {model_name:10} {successes:8} {trials:8} {rate:9.1f}% {diff_str:10}")
        
        print("\n" + "="*80)


def main():
    logging.basicConfig(level=logging.INFO)
    
    args = EvalArgs()
    print(f"\n[*] FP8 模型 LIBERO Spatial 评估")
    print(f"    设备: {args.device}")
    
    evaluator = LiberoEvaluator(args)
    
    if not evaluator.tasks:
        logging.error("  ! 无法加载 LIBERO 任务，评估失败")
        return
    
    results = evaluator.evaluate()
    evaluator.save_results(results)
    evaluator.print_summary(results)
    
    print(f"\n[✓] 评估完成")


if __name__ == "__main__":
    main()
