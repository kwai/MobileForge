#!/usr/bin/env python3
"""
按应用名称提取 Rollout 数据脚本

根据指定的 app_name 从 rollout 结果目录中提取对应的数据文件夹和 results.csv 记录到新目录。

使用示例:
    python scripts/extract_app_data.py \
        --source results/rollout/session-mobileforge-rollout-debug-v25111005 \
        --app-name Settings

    # 指定自定义输出目录
    python scripts/extract_app_data.py \
        --source results/rollout/session-mobileforge-rollout-debug-v25111005 \
        --app-name Settings \
        --output results/rollout/session-mobileforge-rollout-settings-v25111005
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="按应用名称提取 Rollout 数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="源 session 目录路径，包含 results.csv 和数据文件夹"
    )
    parser.add_argument(
        "--app-name", "-a",
        required=True,
        help="要提取的应用名称，如 Settings、Clock、Calendar 等"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录路径（可选），默认根据源目录和应用名称自动生成"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要执行的操作，不实际复制文件"
    )
    return parser.parse_args()


def generate_output_path(source_path: str, app_name: str) -> str:
    """
    根据源路径和应用名称生成输出路径
    
    例如：
    source: results/rollout/session-mobileforge-rollout-debug-v25111005
    app_name: Settings
    output: results/rollout/session-mobileforge-rollout-settings-v25111005
    """
    source_dir = Path(source_path)
    parent_dir = source_dir.parent
    source_name = source_dir.name
    
    # 尝试从源目录名中提取版本号
    # 格式通常为 session-mobileforge-rollout-{type}-v{version}
    version_match = re.search(r'-v(\d+)$', source_name)
    if version_match:
        version = version_match.group(1)
        output_name = f"session-mobileforge-rollout-{app_name.lower()}-v{version}"
    else:
        # 如果没有版本号，直接使用应用名称
        output_name = f"session-mobileforge-rollout-{app_name.lower()}"
    
    return str(parent_dir / output_name)


def extract_app_data(source_path: str, app_name: str, output_path: str, dry_run: bool = False):
    """
    提取指定应用的数据
    
    Args:
        source_path: 源 session 目录路径
        app_name: 要提取的应用名称
        output_path: 输出目录路径
        dry_run: 是否仅预览操作
    """
    source_dir = Path(source_path)
    output_dir = Path(output_path)
    
    # 检查源目录是否存在
    if not source_dir.exists():
        print(f"错误: 源目录不存在: {source_dir}")
        sys.exit(1)
    
    # 检查 results.csv 是否存在
    results_csv = source_dir / "results.csv"
    if not results_csv.exists():
        print(f"错误: results.csv 不存在: {results_csv}")
        sys.exit(1)
    
    # 读取 CSV 文件
    print(f"正在读取: {results_csv}")
    df = pd.read_csv(results_csv)
    
    # 检查必要的列是否存在
    required_columns = ["task_identifier", "app_name"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: CSV 缺少必要的列: {missing_columns}")
        sys.exit(1)
    
    # 筛选指定应用的数据
    filtered_df = df[df["app_name"] == app_name]
    
    if filtered_df.empty:
        print(f"警告: 没有找到 app_name 为 '{app_name}' 的数据")
        print(f"可用的 app_name 值: {df['app_name'].unique().tolist()}")
        sys.exit(1)
    
    # 获取需要复制的 task_identifier 列表
    task_identifiers = filtered_df["task_identifier"].unique().tolist()
    
    print(f"\n=== 提取统计 ===")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {output_dir}")
    print(f"目标应用: {app_name}")
    print(f"匹配的 CSV 行数: {len(filtered_df)}")
    print(f"需要复制的文件夹数: {len(task_identifiers)}")
    
    if dry_run:
        print("\n[预览模式] 将要复制的文件夹:")
        for task_id in task_identifiers:
            folder_path = source_dir / task_id
            exists = "✓" if folder_path.exists() else "✗ (不存在)"
            print(f"  - {task_id} {exists}")
        print("\n[预览模式] 不会执行实际操作")
        return
    
    # 检查输出目录是否已存在
    if output_dir.exists():
        print(f"\n警告: 输出目录已存在: {output_dir}")
        response = input("是否覆盖? (y/N): ").strip().lower()
        if response != 'y':
            print("操作已取消")
            sys.exit(0)
        shutil.rmtree(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制文件夹
    print("\n正在复制文件夹...")
    copied_count = 0
    missing_folders = []
    
    for task_id in task_identifiers:
        source_folder = source_dir / task_id
        target_folder = output_dir / task_id
        
        if source_folder.exists() and source_folder.is_dir():
            shutil.copytree(source_folder, target_folder)
            copied_count += 1
            print(f"  已复制: {task_id}")
        else:
            missing_folders.append(task_id)
            print(f"  跳过 (不存在): {task_id}")
    
    # 保存筛选后的 results.csv
    output_csv = output_dir / "results.csv"
    filtered_df.to_csv(output_csv, index=False)
    print(f"\n已保存 results.csv: {output_csv}")
    
    # 输出统计结果
    print("\n=== 完成统计 ===")
    print(f"已复制文件夹: {copied_count}/{len(task_identifiers)}")
    print(f"CSV 记录数: {len(filtered_df)}")
    
    if missing_folders:
        print(f"\n警告: 以下 {len(missing_folders)} 个文件夹不存在:")
        for folder in missing_folders:
            print(f"  - {folder}")
    
    print(f"\n✓ 提取完成: {output_dir}")


def main():
    args = parse_args()
    
    # 解析路径
    source_path = os.path.abspath(args.source)
    
    # 生成或使用指定的输出路径
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        output_path = generate_output_path(source_path, args.app_name)
    
    # 执行提取
    extract_app_data(
        source_path=source_path,
        app_name=args.app_name,
        output_path=output_path,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
