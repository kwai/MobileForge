#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速为所有agent添加IRR列，先处理简单案例
"""

import os
import sys
import json
import pandas as pd

sys.path.append('../..')
from mobilegym_critic.irr.irr_agent import get_agent_name_from_results_csv


def add_irr_columns_to_agent(agent_dir: str) -> str:
    """
    为单个agent添加IRR列，先处理简单案例
    """
    results_csv_path = os.path.join(agent_dir, "results.csv")
    if not os.path.exists(results_csv_path):
        print(f"Warning: {results_csv_path} not found")
        return ""
    
    # 读取现有结果
    df = pd.read_csv(results_csv_path)
    agent_name = get_agent_name_from_results_csv(results_csv_path)
    
    # 读取任务定义
    task_df = pd.read_csv("../../data/memgui-v25071601.csv")
    memory_tasks = task_df[task_df['requires_ui_memory'] == 'Y']['task_identifier'].tolist()
    
    print(f"Processing agent: {agent_name}")
    print(f"Found {len(memory_tasks)} tasks requiring UI memory")
    
    # 添加IRR相关列
    irr_columns = []
    for attempt in [1, 2, 3]:
        irr_columns.extend([
            f"{agent_name}_attempt_{attempt}_irr_total_units",
            f"{agent_name}_attempt_{attempt}_irr_correct_units", 
            f"{agent_name}_attempt_{attempt}_irr_percentage",
            f"{agent_name}_attempt_{attempt}_irr_reason"
        ])
    
    # 初始化新列
    for col in irr_columns:
        df[col] = None
    
    # 统计各种情况的数量
    stats = {
        'finish_signal_check': 0,
        'pre_evaluation': 0, 
        'success_other': 0,
        'need_analysis': 0,
        'no_evaluation': 0
    }
    
    # 处理每个任务
    for idx, row in df.iterrows():
        task_id = row['task_identifier']
        
        # 只处理需要UI记忆的任务
        if task_id not in memory_tasks:
            continue
        
        for attempt in [1, 2, 3]:
            eval_col = f"{agent_name}_direct_with_action_attempt_{attempt}_evaluation"
            details_col = f"{agent_name}_direct_with_action_attempt_{attempt}_details"
            method_col = f"{agent_name}_direct_with_action_attempt_{attempt}_evaluation_method"
            
            if eval_col not in df.columns:
                continue
                
            evaluation = row[eval_col]
            details = row[details_col] if details_col in df.columns else ""
            method = row[method_col] if method_col in df.columns else ""
            
            # 跳过空值
            if pd.isna(evaluation) or evaluation == "":
                stats['no_evaluation'] += 1
                continue
                
            irr_total_col = f"{agent_name}_attempt_{attempt}_irr_total_units"
            irr_correct_col = f"{agent_name}_attempt_{attempt}_irr_correct_units"
            irr_percentage_col = f"{agent_name}_attempt_{attempt}_irr_percentage"
            irr_reason_col = f"{agent_name}_attempt_{attempt}_irr_reason"
            
            # 根据evaluation_method确定IRR
            if method == "finish_signal_check":
                # finish_signal=0，直接标注IRR为0
                df.loc[idx, irr_total_col] = "N/A"
                df.loc[idx, irr_correct_col] = 0
                df.loc[idx, irr_percentage_col] = 0
                df.loc[idx, irr_reason_col] = "Task failed due to finish_signal=0"
                stats['finish_signal_check'] += 1
                
            elif method == "pre_evaluation":
                # pre_evaluation成功，IRR为100%
                df.loc[idx, irr_total_col] = "N/A" 
                df.loc[idx, irr_correct_col] = "N/A"
                df.loc[idx, irr_percentage_col] = 100
                df.loc[idx, irr_reason_col] = "Task succeeded in pre-evaluation"
                stats['pre_evaluation'] += 1
                
            elif evaluation == "S":
                # 其他方法但任务成功，IRR为100%
                df.loc[idx, irr_total_col] = "N/A"
                df.loc[idx, irr_correct_col] = "N/A" 
                df.loc[idx, irr_percentage_col] = 100
                df.loc[idx, irr_reason_col] = "Task succeeded"
                stats['success_other'] += 1
                
            elif evaluation == "F":
                # 其他方法但任务失败，标记需要详细分析
                df.loc[idx, irr_total_col] = "NEED_ANALYSIS"
                df.loc[idx, irr_correct_col] = "NEED_ANALYSIS"
                df.loc[idx, irr_percentage_col] = "NEED_ANALYSIS"
                df.loc[idx, irr_reason_col] = f"Need detailed analysis for {task_id} attempt {attempt}"
                stats['need_analysis'] += 1
    
    print(f"Statistics:")
    print(f"  finish_signal_check: {stats['finish_signal_check']}")
    print(f"  pre_evaluation: {stats['pre_evaluation']}")
    print(f"  success_other: {stats['success_other']}")
    print(f"  need_analysis: {stats['need_analysis']}")
    print(f"  no_evaluation: {stats['no_evaluation']}")
    
    # 保存新的CSV文件
    output_path = os.path.join(agent_dir, "results_add_irr.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved results with IRR columns to: {output_path}")
    
    return output_path


def process_all_agents():
    """快速为所有agent添加IRR列"""
    base_dir = "../../results/00_baselines"
    agent_dirs = [
        "250719_t3a_gemini-2.5-flash",
        "250719_T3A_gemini-2.5-pro", 
        "250721_M3A_gemini-2.5-pro",
        "250723_mobileagente_gemini-2.5-pro",
        "250819_tars_1.5_7b",
        "250831_gui_owl_7b",
        "25082701_agent_s2_gemini-2.5-pro"
    ]
    
    total_need_analysis = 0
    
    for agent_dir_name in agent_dirs:
        agent_path = os.path.join(base_dir, agent_dir_name)
        if os.path.exists(agent_path):
            print(f"\n=== Processing {agent_dir_name} ===")
            try:
                add_irr_columns_to_agent(agent_path)
                
                # 统计需要详细分析的任务数量
                df = pd.read_csv(os.path.join(agent_path, "results_add_irr.csv"))
                need_analysis = len([col for col in df.columns if 'irr_percentage' in col and 
                                   (df[col] == "NEED_ANALYSIS").any()])
                total_need_analysis += need_analysis
                
                print(f"✓ Completed {agent_dir_name}")
            except Exception as e:
                print(f"✗ Error processing {agent_dir_name}: {e}")
        else:
            print(f"Warning: {agent_path} not found")
    
    print(f"\n=== Summary ===")
    print(f"Total tasks needing detailed IRR analysis: {total_need_analysis}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick add IRR columns to all agents")
    parser.add_argument("--agent_dir", type=str, help="Path to specific agent directory")
    parser.add_argument("--all", action="store_true", help="Process all agents")
    
    args = parser.parse_args()
    
    if args.all:
        process_all_agents()
    elif args.agent_dir:
        if os.path.exists(args.agent_dir):
            add_irr_columns_to_agent(args.agent_dir)
        else:
            print(f"Agent directory not found: {args.agent_dir}")
    else:
        print("Please specify --agent_dir or --all")
