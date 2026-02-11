#!/usr/bin/env python3
"""Analyze FP32 vs INT8 benchmark results."""
import re
from pathlib import Path

def parse_log(path):
    """Parse benchmark log file."""
    if not path.exists():
        return None
    
    content = path.read_text()
    result = {'suite': path.stem.split('_')[1]}
    
    # Extract success rate
    match = re.search(r'Total Success Rate:\s+([\d.]+)', content)
    if match:
        result['accuracy'] = float(match.group(1)) * 100
    
    # Extract latency
    match = re.search(r'Mean=([\d.]+)', content)
    if match:
        result['latency'] = float(match.group(1))
    
    # Extract memory
    match = re.search(r'Peak=([\d.]+)', content)
    if match:
        result['memory'] = float(match.group(1))
    
    return result

def main():
    log_dir = Path('benchmark_logs')
    results = {'fp32': {}, 'int8': {}}
    
    for suite in ['spatial', 'goal', 'object', '10']:
        fp32_log = log_dir / f'fp32_{suite}_20trials.log'
        int8_log = log_dir / f'int8_{suite}_20trials.log'
        
        fp32 = parse_log(fp32_log)
        int8 = parse_log(int8_log)
        
        if fp32:
            results['fp32'][suite] = fp32
        if int8:
            results['int8'][suite] = int8
    
    # Generate report
    print("\n" + "="*70)
    print("  FP32 vs INT8 Benchmark Comparison")
    print("="*70 + "\n")
    
    print("Accuracy Comparison:")
    print(f"{'Suite':<15} {'FP32':<10} {'INT8':<10} {'Diff':<10}")
    print("-" * 50)
    
    fp32_acc_sum = 0
    int8_acc_sum = 0
    count = 0
    
    for suite in ['spatial', 'goal', 'object', '10']:
        if suite in results['fp32'] and suite in results['int8']:
            fp32_acc = results['fp32'][suite]['accuracy']
            int8_acc = results['int8'][suite]['accuracy']
            diff = int8_acc - fp32_acc
            
            print(f"libero_{suite:<8} {fp32_acc:>6.1f}%    {int8_acc:>6.1f}%    {diff:>+6.1f}%")
            
            fp32_acc_sum += fp32_acc
            int8_acc_sum += int8_acc
            count += 1
    
    if count > 0:
        print("-" * 50)
        fp32_avg = fp32_acc_sum / count
        int8_avg = int8_acc_sum / count
        diff_avg = int8_avg - fp32_avg
        print(f"{'Average':<15} {fp32_avg:>6.1f}%    {int8_avg:>6.1f}%    {diff_avg:>+6.1f}%")
    
    print("\n" + "="*70)
    
    # Save detailed report
    report_path = Path('benchmark_results/COMPARISON_REPORT.md')
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(f"# Benchmark Comparison Report\n\n")
        f.write(f"## Accuracy\n")
        f.write(f"- FP32: {fp32_avg:.1f}%\n")
        f.write(f"- INT8: {int8_avg:.1f}%\n")
        f.write(f"- Difference: {diff_avg:+.1f}%\n")
    
    print(f"\n✓ Report saved: {report_path}\n")

if __name__ == '__main__':
    main()
