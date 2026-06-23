#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IRR功能演示和状态检查
"""

import os
import pandas as pd
import json


def check_irr_status():
    """检查所有agent的IRR处理状态"""
    
    print("="*80)
    print("MobileGym-Critic IRR处理状态检查")
    print("="*80)
    
    agents = [
        "250719_t3a_gemini-2.5-flash",
        "250719_T3A_gemini-2.5-pro", 
        "250721_M3A_gemini-2.5-pro",
        "250723_mobileagente_gemini-2.5-pro",
        "250819_tars_1.5_7b",
        "250831_gui_owl_7b",
        "25082701_agent_s2_gemini-2.5-pro"
    ]
    
    print("\n📁 文件存在性检查")
    print("-" * 50)
    
    all_files_exist = True
    for agent_name in agents:
        csv_path = f"../../results/00_baselines/{agent_name}/results_add_irr.csv"
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path) / 1024  # KB
            print(f"✅ {agent_name}: {file_size:.1f} KB")
        else:
            print(f"❌ {agent_name}: 文件不存在")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ 部分文件缺失，请先运行: python3 quick_add_irr_columns.py --all")
        return False
    
    # 统计IRR数据分布
    task_df = pd.read_csv("../../data/memgui-v25071601.csv")
    memory_tasks = task_df[task_df['requires_ui_memory'] == 'Y']['task_identifier'].tolist()
    
    total_stats = {
        'agents_with_irr': 0,
        'total_finish_signal': 0,
        'total_pre_eval': 0,
        'total_success': 0,
        'total_need_analysis': 0,
        'total_completed_analysis': 0
    }
    
    print(f"\n📊 IRR数据分布统计 (需要UI记忆的任务: {len(memory_tasks)})")
    print("-" * 50)
    
    for agent_name in agents:
        csv_path = f"../../results/00_baselines/{agent_name}/results_add_irr.csv"
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        total_stats['agents_with_irr'] += 1
        
        # 统计IRR相关列
        irr_percentage_cols = [col for col in df.columns if 'irr_percentage' in col]
        
        agent_stats = {
            'finish_signal': 0,
            'pre_eval': 0,
            'success': 0,
            'need_analysis': 0,
            'completed_analysis': 0
        }
        
        # 统计各种IRR情况
        for col in irr_percentage_cols:
            for idx, val in df[col].items():
                task_id = df.loc[idx, 'task_identifier']
                
                # 只统计需要UI记忆的任务
                if task_id not in memory_tasks:
                    continue
                    
                if pd.isna(val):
                    continue
                    
                if val == 0:
                    reason_col = col.replace('_percentage', '_reason')
                    if reason_col in df.columns:
                        reason = df.loc[idx, reason_col]
                        if 'finish_signal' in str(reason):
                            agent_stats['finish_signal'] += 1
                elif val == 100:
                    reason_col = col.replace('_percentage', '_reason')
                    if reason_col in df.columns:
                        reason = df.loc[idx, reason_col]
                        if 'pre-evaluation' in str(reason):
                            agent_stats['pre_eval'] += 1
                        else:
                            agent_stats['success'] += 1
                elif val == "NEED_ANALYSIS":
                    agent_stats['need_analysis'] += 1
                elif isinstance(val, (int, float)) and 0 < val < 100:
                    agent_stats['completed_analysis'] += 1
        
        print(f"{agent_name}:")
        print(f"  🟢 成功案例: {agent_stats['success'] + agent_stats['pre_eval']}")
        print(f"  🔴 失败案例: {agent_stats['finish_signal']}")
        print(f"  🟡 已完成分析: {agent_stats['completed_analysis']}")
        print(f"  ⏳ 待分析: {agent_stats['need_analysis']}")
        
        # 更新总计
        total_stats['total_finish_signal'] += agent_stats['finish_signal']
        total_stats['total_pre_eval'] += agent_stats['pre_eval']
        total_stats['total_success'] += agent_stats['success']
        total_stats['total_need_analysis'] += agent_stats['need_analysis']
        total_stats['total_completed_analysis'] += agent_stats['completed_analysis']
    
    print(f"\n📈 总体统计")
    print("-" * 50)
    print(f"✅ 已处理Agent数量: {total_stats['agents_with_irr']}/7")
    print(f"🟢 成功案例总数: {total_stats['total_success'] + total_stats['total_pre_eval']}")
    print(f"🔴 Finish Signal失败: {total_stats['total_finish_signal']}")
    print(f"🟡 已完成详细分析: {total_stats['total_completed_analysis']}")
    print(f"⏳ 仍需详细分析: {total_stats['total_need_analysis']}")
    
    # 推荐下一步操作
    if total_stats['total_need_analysis'] > 0:
        print(f"\n💡 建议操作:")
        print(f"   运行并行处理器来完成剩余{total_stats['total_need_analysis']}个案例的分析:")
        print(f"   python3 true_parallel_irr_processor.py --all")
        print(f"   预计耗时: {total_stats['total_need_analysis'] * 2 / 120:.1f}分钟 (2 QPS并行)")
    else:
        print(f"\n🎉 所有IRR分析已完成!")
    
    return True


def demo_irr_functionality():
    """演示IRR功能的实现"""
    
    print("="*80)
    print("MobileGym-Critic Information Retention Rate (IRR) 功能演示")
    print("="*80)
    
    print("\n📋 1. IRR指标说明")
    print("-" * 40)
    print("IRR (Information Retention Rate) 是一个细粒度的记忆保真度指标，")
    print("用于量化智能体在执行任务过程中正确回忆并利用关键信息的能力。")
    print("计算公式: IRR = (正确使用的信息单元数 / 总信息单元数) × 100%")
    
    print("\n📊 2. 处理规则")
    print("-" * 40)
    print("• finish_signal_check: IRR = 0% (任务因finish_signal=0而失败)")
    print("• pre_evaluation: IRR = 100% (任务在预评估阶段成功)")
    print("• 其他成功案例: IRR = 100%")
    print("• 其他失败案例: 需要IRR agent详细分析")
    
    print("\n📁 3. 生成的文件")
    print("-" * 40)
    
    agents = [
        "250719_t3a_gemini-2.5-flash",
        "250719_T3A_gemini-2.5-pro", 
        "250721_M3A_gemini-2.5-pro",
        "250723_mobileagente_gemini-2.5-pro",
        "250819_tars_1.5_7b",
        "250831_gui_owl_7b",
        "25082701_agent_s2_gemini-2.5-pro"
    ]
    
    for agent_name in agents:
        csv_path = f"../../results/00_baselines/{agent_name}/results_add_irr.csv"
        if os.path.exists(csv_path):
            print(f"✅ {agent_name}/results_add_irr.csv")
        else:
            print(f"❌ {agent_name}/results_add_irr.csv (未找到)")
    
    print("\n📈 4. 示例IRR数据 (M3A agent)")
    print("-" * 40)
    
    # 显示M3A agent的示例数据
    df = pd.read_csv("../../results/00_baselines/250721_M3A_gemini-2.5-pro/results_add_irr.csv")
    task_df = pd.read_csv("../../data/memgui-v25071601.csv")
    memory_tasks = task_df[task_df['requires_ui_memory'] == 'Y']['task_identifier'].tolist()
    
    print("任务ID | 尝试次数 | IRR% | 类型")
    print("-" * 50)
    
    sample_count = 0
    for task_id in memory_tasks:
        if task_id in df['task_identifier'].values and sample_count < 10:
            row = df[df['task_identifier'] == task_id].iloc[0]
            
            for attempt in [1, 2, 3]:
                percent_col = f'M3A_vivo_gemini_attempt_{attempt}_irr_percentage'
                reason_col = f'M3A_vivo_gemini_attempt_{attempt}_irr_reason'
                
                if percent_col in df.columns:
                    percent = row[percent_col]
                    reason = row[reason_col] if reason_col in df.columns else ''
                    
                    if pd.notna(percent) and percent != "":
                        # 确定类型
                        if percent == 0 and 'finish_signal' in str(reason):
                            irr_type = "finish_signal"
                        elif percent == 100 and 'pre-evaluation' in str(reason):
                            irr_type = "pre_eval"
                        elif percent == 100:
                            irr_type = "success"
                        elif percent == "NEED_ANALYSIS":
                            irr_type = "need_analysis"
                        elif isinstance(percent, (int, float)):
                            irr_type = "analyzed"
                        else:
                            irr_type = "other"
                        
                        print(f"{task_id[:20]:<20} | {attempt:^8} | {str(percent):<4} | {irr_type}")
                        sample_count += 1
                        break
            
            if sample_count >= 10:
                break
    
    print("\n🔧 5. 新增的IRR列")
    print("-" * 40)
    print("每个agent的results_add_irr.csv文件中新增了以下列:")
    print("• {agent_name}_attempt_{n}_irr_total_units: 总信息单元数")
    print("• {agent_name}_attempt_{n}_irr_correct_units: 正确使用的信息单元数")
    print("• {agent_name}_attempt_{n}_irr_percentage: IRR百分比")
    print("• {agent_name}_attempt_{n}_irr_reason: 分析原因")
    print("其中 n = 1, 2, 3 表示三次尝试")
    
    print("\n✨ 6. 使用说明")
    print("-" * 40)
    print("1. 所有agent的原始results.csv已保持不变")
    print("2. 新的results_add_irr.csv包含了所有原始数据 + IRR指标")
    print("3. 对于需要详细分析的失败案例，可以运行analyze_failed_irr_cases.py")
    print("4. IRR agent使用gemini-2.5-pro模型进行智能分析")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IRR功能演示和状态检查")
    parser.add_argument("--status", action="store_true", help="仅显示IRR处理状态")
    parser.add_argument("--demo", action="store_true", help="仅显示功能演示")
    
    args = parser.parse_args()
    
    if args.status:
        check_irr_status()
    elif args.demo:
        demo_irr_functionality()
    else:
        # 默认显示状态检查和功能演示
        print("🔍 首先检查IRR处理状态...\n")
        if check_irr_status():
            print("\n" + "="*80)
            print("📋 IRR功能演示")
            print("="*80)
            demo_irr_functionality()
