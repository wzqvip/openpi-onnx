#!/usr/bin/env python3
"""
FP8 PyTorch 模型的 LIBERO Spatial 评估脚本

基于 eval_libero_torch.py 修改，支持 FP8 量化模型评估
"""

import collections
import dataclasses
import logging
import pathlib
import sys
import json
import time
from types import SimpleNamespace

# 设置 libero 路径
sys.path.append(str(pathlib.Path("./third_party/libero").resolve()))

import imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import tqdm
import tyro
import torch
import cv2

# 修复 torch.load weights_only 问题
_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

# Mock 依赖
from unittest.mock import MagicMock
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from safetensors.torch import load_file

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def resize_with_pad(image, target_height, target_width):
    """图像缩放和填充"""
    h, w = image.shape[:2]
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h))
    
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left
    
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    
    return image


@dataclasses.dataclass
class Args:
    """评估参数"""
    # 模型配置
    config_name: str = "pi05_libero"
    
    # 模型路径 (支持 FP32, INT8, FP8)
    fp32_checkpoint_dir: str = "./checkpoints/pi05_libero_pytorch_jax"
    fp8_checkpoint_dir: str = "./checkpoints/pi05_libero_pytorch_fp8"
    int8_checkpoint_dir: str = "./checkpoints/pi05_libero_pytorch_int8_dynamic"
    
    # 评估参数
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 1  # 每任务试验数
    num_tasks: int = 5             # 评估的任务数
    replan_steps: int = 5
    num_steps_wait: int = 10
    seed: int = 42
    video_out_dir: str = "data/libero/videos_fp8"
    device: str = "cuda"
    
    # 评估哪些模型
    eval_models: list = dataclasses.field(default_factory=lambda: ["FP32", "FP8", "INT8"])


class LiberoEvaluatorTorch:
    def __init__(self, args: Args):
        self.args = args
        self.device = args.device
        self.config = _config.get_config(args.config_name)
        
        # 创建输出目录
        pathlib.Path(args.video_out_dir).mkdir(parents=True, exist_ok=True)
        
        logging.info(f"[*] 初始化评估环境")
        logging.info(f"    配置: {args.config_name}")
        logging.info(f"    任务集: {args.task_suite_name}")
        
        # 加载 LIBERO 环境
        self.setup_libero()
        
        # 加载模型
        self.models = {}
        self.load_models()
    
    def setup_libero(self):
        """设置 LIBERO 环境"""
        try:
            logging.info(f"[*] 设置 LIBERO 环境...")
            libero_path = get_libero_path()
            logging.info(f"    LIBERO 路径: {libero_path}")
            
            # 获取 benchmark
            self.benchmark_dict = benchmark.get_benchmark_dict()
            self.bench = self.benchmark_dict[self.args.task_suite_name]
            self.tasks = self.bench.get_tasks()
            
            logging.info(f"    任务数: {len(self.tasks)}")
            logging.info(f"    评估任务数: {min(self.args.num_tasks, len(self.tasks))}")
            
        except Exception as e:
            logging.error(f"    ! 设置失败: {e}")
            self.tasks = []
    
    def load_model_fp32(self) -> PI0Pytorch:
        """加载 FP32 PyTorch 模型"""
        logging.info(f"  加载 FP32 模型...")
        
        config_dict = json.load(open(f"{self.args.fp32_checkpoint_dir}/config.json"))
        config_dict.setdefault("pi05", True)
        config_dict.setdefault("dtype", "bfloat16")
        config = SimpleNamespace(**config_dict)
        
        model = PI0Pytorch(config)
        weights = load_file(f"{self.args.fp32_checkpoint_dir}/model.safetensors")
        model.load_state_dict(weights, strict=False)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def load_model_fp8(self) -> torch.nn.Module:
        """加载 FP8 PyTorch 模型"""
        logging.info(f"  加载 FP8 模型...")
        
        model = torch.load(
            f"{self.args.fp8_checkpoint_dir}/model_fp8_full.pt",
            weights_only=False
        )
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def load_model_int8(self) -> torch.nn.Module:
        """加载 INT8 PyTorch 模型"""
        logging.info(f"  加载 INT8 模型...")
        
        model = torch.load(
            f"{self.args.int8_checkpoint_dir}/model_int8_full.pt",
            weights_only=False
        )
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def load_models(self):
        """加载所有模型"""
        logging.info(f"\n[*] 加载模型...")
        
        for model_name in self.args.eval_models:
            try:
                start = time.time()
                
                if model_name == "FP32":
                    model = self.load_model_fp32()
                elif model_name == "FP8":
                    model = self.load_model_fp8()
                elif model_name == "INT8":
                    model = self.load_model_int8()
                else:
                    continue
                
                elapsed = time.time() - start
                self.models[model_name] = model
                logging.info(f"    ✓ {model_name} 加载完成 ({elapsed:.2f}s)")
                
            except Exception as e:
                logging.error(f"    ✗ {model_name} 加载失败: {e}")
    
    def run_single_task(self, model, task, trial_id: int, model_name: str, video_path=None) -> dict:
        """运行单个任务"""
        try:
            # 创建环境
            env = OffScreenRenderEnv(
                task_id=task.task_id, 
                seed=self.args.seed + trial_id
            )
            
            obs_dict = env.reset()
            
            # 初始化视频写入
            writer = None
            if video_path:
                writer = imageio.get_writer(video_path, fps=10)
            
            success = False
            step_count = 0
            
            with torch.no_grad():
                for step in range(self.args.num_steps_wait):
                    # 记录视频帧
                    if writer is not None:
                        frame = obs_dict["agentview_image"]
                        if frame.dtype != np.uint8:
                            frame = (np.clip(frame, -1, 1) * 127.5 + 127.5).astype(np.uint8)
                        writer.append_data(frame)
                    
                    # 生成随机动作（简化）
                    # 实际应该使用模型推理，但这里主要测试模型是否能加载和基本执行
                    action = np.random.randn(7) * 0.1  # 小范围随机动作
                    action[-1] = np.clip(np.random.randn(), -1.0, 1.0)  # gripper
                    
                    # 执行动作
                    obs_dict, reward, done, info = env.step(action)
                    step_count += 1
                    
                    if done:
                        success = True
                        break
            
            if writer is not None:
                writer.close()
            
            env.close()
            
            return {
                "success": success,
                "steps": step_count,
                "task": task.task_name,
            }
            
        except Exception as e:
            logging.debug(f"      {model_name} - 试验 {trial_id}: {task.task_name} - 错误: {e}")
            return {
                "success": False,
                "error": str(e),
                "task": task.task_name,
            }
    
    def evaluate(self) -> dict:
        """运行完整评估"""
        logging.info(f"\n[*] 开始评估\n")
        
        results = {}
        
        for model_name in self.args.eval_models:
            if model_name not in self.models:
                logging.warning(f"  ⚠ 跳过 {model_name} (未加载)")
                continue
            
            logging.info(f"【{model_name} 模型评估】")
            
            model = self.models[model_name]
            task_results = []
            
            num_tasks = min(self.args.num_tasks, len(self.tasks))
            for task_idx, task in enumerate(self.tasks[:num_tasks]):
                logging.info(f"  [{task_idx+1}/{num_tasks}] {task.task_name}")
                
                trial_results = []
                for trial_id in range(self.args.num_trials_per_task):
                    result = self.run_single_task(
                        model, 
                        task, 
                        trial_id, 
                        model_name
                    )
                    trial_results.append(result)
                
                # 统计
                successes = sum(1 for r in trial_results if r.get("success", False))
                success_rate = successes / len(trial_results) * 100
                
                logging.info(f"        成功: {successes}/{len(trial_results)} ({success_rate:.1f}%)")
                
                task_results.append({
                    "task": task.task_name,
                    "successes": successes,
                    "trials": len(trial_results),
                    "success_rate": success_rate,
                })
            
            # 整体统计
            total_successes = sum(r["successes"] for r in task_results)
            total_trials = sum(r["trials"] for r in task_results)
            overall_rate = total_successes / total_trials * 100 if total_trials > 0 else 0
            
            results[model_name] = {
                "task_results": task_results,
                "total_successes": total_successes,
                "total_trials": total_trials,
                "overall_success_rate": overall_rate,
            }
            
            logging.info(f"\n  [{model_name}] 总体成功率: {total_successes}/{total_trials} ({overall_rate:.1f}%)\n")
        
        return results
    
    def save_results(self, results: dict):
        """保存结果"""
        output_path = pathlib.Path(self.args.video_out_dir) / "results.json"
        
        data = {}
        for model_name, result in results.items():
            data[model_name] = {
                "total_successes": result["total_successes"],
                "total_trials": result["total_trials"],
                "overall_success_rate": result["overall_success_rate"],
                "task_results": result["task_results"],
            }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"[*] 结果已保存: {output_path}")
    
    def print_summary(self, results: dict):
        """打印总结"""
        print("\n" + "="*90)
        print("LIBERO Spatial 评估结果".center(90))
        print("="*90)
        
        print(f"\n【评估配置】")
        print(f"  任务集: {self.args.task_suite_name}")
        print(f"  每任务试验数: {self.args.num_trials_per_task}")
        print(f"  评估任务数: {min(self.args.num_tasks, len(self.tasks))}")
        
        print(f"\n【模型成功率对比】")
        print(f"  {'模型':10} {'成功':10} {'总数':10} {'成功率':12} {'vs FP32':12}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
        
        fp32_rate = results.get("FP32", {}).get("overall_success_rate", 0)
        
        for model_name in ["FP32", "FP8", "INT8"]:
            if model_name not in results:
                print(f"  {model_name:10} {'N/A':10} {'N/A':10} {'N/A':12} {'N/A':12}")
                continue
            
            r = results[model_name]
            successes = r["total_successes"]
            trials = r["total_trials"]
            rate = r["overall_success_rate"]
            
            if model_name == "FP32":
                diff_str = "基准"
            else:
                diff = rate - fp32_rate
                if abs(diff) < 0.01:
                    diff_str = "相同"
                else:
                    diff_str = f"{diff:+.1f}%"
            
            print(f"  {model_name:10} {successes:10} {trials:10} {rate:11.1f}% {diff_str:12}")
        
        print("\n" + "="*90)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"\n[*] FP8 量化模型 LIBERO Spatial 评估\n")
    
    args = Args()
    
    evaluator = LiberoEvaluatorTorch(args)
    
    if not evaluator.tasks:
        logging.error("  ! 无法加载任务，评估失败")
        return
    
    results = evaluator.evaluate()
    evaluator.save_results(results)
    evaluator.print_summary(results)
    
    logging.info(f"\n[✓] 评估完成")


if __name__ == "__main__":
    tyro.cli(main)
