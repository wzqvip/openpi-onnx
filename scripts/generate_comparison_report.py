#!/usr/bin/env python3
"""
Generate comparison table from benchmark JSON results.
"""

import json
import sys
import pathlib
from tabulate import tabulate

def load_benchmark_results(results_dir="./benchmark_results"):
    """Load all benchmark JSON results."""
    results_dir = pathlib.Path(results_dir)
    results = {}
    
    for json_file in sorted(results_dir.glob("benchmark_*.json")):
        with open(json_file) as f:
            data = json.load(f)
            model_type = data["model_type"].upper()
            results[model_type] = data
    
    return results


def generate_summary_table(results):
    """Generate summary comparison table."""
    table_data = []
    
    for model_type in sorted(results.keys()):
        data = results[model_type]
        table_data.append([
            model_type,
            f"{data['engine_size_gb']:.2f} GB",
            f"{data['overall_success_rate_percent']:.2f}%",
            f"{data['overall_avg_latency_ms']:.2f}ms",
            f"{data['total_successes']}/{data['total_trials']}",
        ])
    
    headers = ["Model", "Engine Size", "Success Rate", "Avg Latency", "Successes"]
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY COMPARISON")
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*80 + "\n")


def generate_suite_table(results):
    """Generate per-suite comparison table."""
    # Collect suite names
    all_suites = set()
    for data in results.values():
        all_suites.update(data["suite_results"].keys())
    
    for suite_name in sorted(all_suites):
        print(f"\n{suite_name.upper()}")
        print("-" * 80)
        
        table_data = []
        for model_type in sorted(results.keys()):
            data = results[model_type]
            suite_data = data["suite_results"].get(suite_name, {})
            
            if suite_data:
                table_data.append([
                    model_type,
                    f"{suite_data['suite_success_rate']:.2f}%",
                    f"{suite_data['suite_avg_latency_ms']:.2f}ms",
                    f"{len(suite_data['tasks'])} tasks",
                ])
        
        headers = ["Model", "Success Rate", "Avg Latency", "Tasks"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))


def generate_detailed_comparison(results):
    """Generate detailed task-by-task comparison."""
    print("\n" + "="*80)
    print("DETAILED TASK COMPARISON")
    print("="*80 + "\n")
    
    # Get all suites and tasks
    all_suites = set()
    for data in results.values():
        all_suites.update(data["suite_results"].keys())
    
    for suite_name in sorted(all_suites):
        print(f"\n{suite_name.upper()}")
        print("-" * 80)
        
        # Get all task indices
        all_tasks = set()
        for model_data in results.values():
            suite_data = model_data["suite_results"].get(suite_name, {})
            all_tasks.update(suite_data.get("tasks", {}).keys())
        
        # Build table
        table_data = []
        for task_id in sorted(all_tasks):
            row = [task_id]
            for model_type in sorted(results.keys()):
                model_data = results[model_type]
                suite_data = model_data["suite_results"].get(suite_name, {})
                task_data = suite_data.get("tasks", {}).get(task_id, {})
                
                if task_data:
                    success_rate = task_data["success_rate"]
                    row.append(f"{success_rate:.1f}%")
                else:
                    row.append("N/A")
            
            table_data.append(row)
        
        headers = ["Task"] + sorted(results.keys())
        print(tabulate(table_data, headers=headers, tablefmt="simple"))


def generate_markdown_report(results, output_file="benchmark_results/BENCHMARK_REPORT.md"):
    """Generate markdown report."""
    output_path = pathlib.Path(output_file)
    
    with open(output_path, "w") as f:
        f.write("# Model Quantization Benchmark Report\n\n")
        f.write("## Executive Summary\n\n")
        
        # Summary table
        f.write("| Model | Engine Size | Success Rate | Avg Latency | Efficiency |\n")
        f.write("|-------|------------|--------------|-------------|------------|\n")
        
        for model_type in sorted(results.keys()):
            data = results[model_type]
            efficiency = f"{data['overall_success_rate_percent'] / data['overall_avg_latency_ms']:.2f}"
            f.write(f"| {model_type} | "
                   f"{data['engine_size_gb']:.2f} GB | "
                   f"{data['overall_success_rate_percent']:.2f}% | "
                   f"{data['overall_avg_latency_ms']:.2f}ms | "
                   f"{efficiency} |\n")
        
        f.write("\n## Detailed Results\n\n")
        
        # Per-suite results
        all_suites = set()
        for data in results.values():
            all_suites.update(data["suite_results"].keys())
        
        for suite_name in sorted(all_suites):
            f.write(f"### {suite_name}\n\n")
            f.write("| Model | Success Rate | Avg Latency |\n")
            f.write("|-------|--------------|-------------|\n")
            
            for model_type in sorted(results.keys()):
                data = results[model_type]
                suite_data = data["suite_results"].get(suite_name, {})
                
                if suite_data:
                    f.write(f"| {model_type} | "
                           f"{suite_data['suite_success_rate']:.2f}% | "
                           f"{suite_data['suite_avg_latency_ms']:.2f}ms |\n")
        
        f.write("\n## Observations\n\n")
        f.write("- Higher success rates indicate better model accuracy\n")
        f.write("- Lower latency indicates faster inference\n")
        f.write("- Smaller engine size is better for deployment\n")
        f.write("- Balance all metrics for optimal deployment choice\n")
        
    print(f"\nMarkdown report generated: {output_path}")


def main():
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "./benchmark_results"
    
    print(f"Loading benchmark results from: {results_dir}")
    results = load_benchmark_results(results_dir)
    
    if not results:
        print("No benchmark results found!")
        return 1
    
    print(f"Found {len(results)} model results: {', '.join(results.keys())}\n")
    
    # Generate tables
    generate_summary_table(results)
    generate_suite_table(results)
    generate_detailed_comparison(results)
    generate_markdown_report(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
