#!/usr/bin/env python3
"""提取所有量化模型的评估结果"""

import re
import json
from pathlib import Path

def extract_results_from_log(log_file):
    """从日志文件中提取成功率"""
    if not Path(log_file).exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 尝试多种模式
    patterns = [
        r'average success rate.*?(\d+\.\d+)%',
        r'Success Rate:\s*(\d+\.\d+)%',
        r'Overall:\s*\d+/\d+\s*\((\d+\.\d+)%\)',
        r'(\d+)/(\d+)\s*episodes succeeded',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            if len(match.groups()) == 1:
                return float(match.group(1))
            elif len(match.groups()) == 2:
                success = int(match.group(1))
                total = int(match.group(2))
                return (success / total * 100) if total > 0 else 0
    
    return None

def main():
    base_dir = Path('/home/taco/openpi-onnx')
    
    # 定义要检查的日志文件
    log_files = {
        'INT8 Spatial': base_dir / 'benchmark_logs/int8_spatial_20trials_v1.log',
        'INT8 Object': base_dir / 'benchmark_logs/int8_object_20trials_v1.log',
        'INT8 Goal': base_dir / 'benchmark_logs/int8_goal_20trials_v1.log',
        'INT8 10': base_dir / 'benchmark_logs/int8_10_20trials_v1.log',
        'INT8 Full': base_dir / 'benchmark_logs/int8_full_v1.log',
        'FP16 Spatial': base_dir / 'benchmark_logs/fp16_spatial_20trials.log',
    }
    
    print("=" * 60)
    print("量化模型评估结果汇总")
    print("=" * 60)
    
    results = {}
    for name, log_file in log_files.items():
        success_rate = extract_results_from_log(log_file)
        results[name] = success_rate
        
        if success_rate is not None:
            print(f"\n✅ {name}: {success_rate:.2f}%")
            if log_file.exists():
                size = log_file.stat().st_size / 1024 / 1024
                print(f"   日志大小: {size:.1f} MB")
        else:
            status = "未找到" if not log_file.exists() else "进行中或格式未知"
            print(f"\n❌ {name}: {status}")
    
    # 尝试从文本文件中提取
    result_files = [
        base_dir / 'logs/int8_verification/eval_int8_full_results.txt',
        base_dir / 'tmp/SPATIAL_EVAL_SUMMARY.md',
    ]
    
    print("\n" + "=" * 60)
    print("备用结果源")
    print("=" * 60)
    
    for result_file in result_files:
        if result_file.exists():
            print(f"\n📄 {result_file.name}:")
            success_rate = extract_results_from_log(result_file)
            if success_rate:
                print(f"   成功率: {success_rate:.2f}%")
    
    # 保存结果到 JSON
    output_file = base_dir / 'tmp/accuracy_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'timestamp': '2026-02-13',
        }, f, indent=2)
    
    print(f"\n💾 结果已保存到: {output_file}")

if __name__ == '__main__':
    main()
