#!/usr/bin/env python3
"""
从 generated_tasks_refactored_20251228_123656.csv 中按app随机采样任务。

功能:
- 读取原始任务CSV文件
- 每个app随机选择25个任务
- 总共得到500个任务 (20 apps × 25 tasks)
- 保存到指定输出路径
"""

import pandas as pd
import os
from pathlib import Path
import argparse


def sample_tasks_by_app(
    input_csv: str,
    output_csv: str,
    tasks_per_app: int = 25,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    从输入CSV中按app随机采样任务。
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        tasks_per_app: 每个app采样的任务数量
        random_seed: 随机种子，用于复现结果
    
    Returns:
        采样后的DataFrame
    """
    # 读取原始CSV
    print(f"读取原始CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"原始数据共 {len(df)} 条记录")
    
    # 获取所有app名称及其任务数量
    app_counts = df['app_name'].value_counts()
    print(f"\n共有 {len(app_counts)} 个不同的app:")
    for app, count in app_counts.items():
        status = "✓" if count >= tasks_per_app else f"⚠️ (不足{tasks_per_app}个)"
        print(f"  - {app}: {count} 个任务 {status}")
    
    # 检查是否所有app都有足够的任务
    insufficient_apps = [app for app, count in app_counts.items() if count < tasks_per_app]
    if insufficient_apps:
        print(f"\n警告: 以下app任务数量不足 {tasks_per_app} 个: {insufficient_apps}")
        print("将采样该app所有可用的任务")
    
    # 按app分组采样
    sampled_dfs = []
    total_sampled = 0
    
    for app_name in sorted(app_counts.index):
        app_df = df[df['app_name'] == app_name]
        n_samples = min(tasks_per_app, len(app_df))
        
        # 随机采样
        sampled = app_df.sample(n=n_samples, random_state=random_seed)
        sampled_dfs.append(sampled)
        total_sampled += n_samples
        print(f"从 {app_name} 采样 {n_samples} 个任务")
    
    # 合并所有采样结果
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\n总共采样 {len(result_df)} 个任务")
    
    # 确保输出目录存在
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存结果
    result_df.to_csv(output_csv, index=False)
    print(f"结果已保存到: {output_csv}")
    
    # 验证结果
    print("\n采样结果验证:")
    result_app_counts = result_df['app_name'].value_counts()
    for app in sorted(result_app_counts.index):
        print(f"  - {app}: {result_app_counts[app]} 个任务")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='从任务CSV中按app随机采样任务'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/generated_tasks_refactored_20251230_160000.csv',
        help='输入CSV文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/rollout/25123101-20apps-500tasks/tasks_20251230_160000_sample500.csv',
        help='输出CSV文件路径'
    )
    parser.add_argument(
        '--tasks-per-app', '-n',
        type=int,
        default=25,
        help='每个app采样的任务数量 (默认: 25)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    args = parser.parse_args()
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 构建完整路径
    input_csv = project_root / args.input
    output_csv = project_root / args.output
    
    # 执行采样
    sample_tasks_by_app(
        input_csv=str(input_csv),
        output_csv=str(output_csv),
        tasks_per_app=args.tasks_per_app,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()

