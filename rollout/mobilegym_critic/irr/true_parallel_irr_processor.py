#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IRR并行处理器 - 支持2 QPS的并行LLM调用
用于处理需要详细分析的失败案例
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from queue import Queue

sys.path.append('../..')
from mobilegym_critic.irr.irr_agent import get_agent_name_from_results_csv, calculate_irr_for_task
from mobilegym_critic.utils.common import parse_json_from_response

# 速率限制器：2 QPS
class RateLimiter:
    def __init__(self, max_calls_per_second: float = 2.0):
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """等待以确保不超过速率限制"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_call_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_call_time = time.time()

# 全局速率限制器
rate_limiter = RateLimiter(max_calls_per_second=1.8)  # 稍微保守一点，避免边界情况


def analyze_irr_case_with_rate_limit(case_info: Tuple) -> Tuple:
    """
    带速率限制的IRR案例分析
    """
    agent_dir, task_id, attempt, idx, task_desc, details, agent_name = case_info
    
    try:
        # 读取prompt_logs.json
        prompt_logs_path = os.path.join(
            agent_dir, task_id, agent_name, f"attempt_{attempt}", "prompt_logs.json"
        )
        
        if not os.path.exists(prompt_logs_path):
            return task_id, attempt, idx, {
                'total_information_units': 'No logs',
                'correctly_used_units': 'No logs',
                'irr_percentage': 'No logs',
                'analysis_reason': 'Prompt logs not found'
            }
            
        with open(prompt_logs_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        # 提取步骤描述
        step_descriptions = []
        for log in logs:
            if 'step_' in log.get('stage', '') and '_description' in log.get('stage', ''):
                try:
                    desc = parse_json_from_response(log['llm_response'])
                    if desc:
                        step_descriptions.append(desc)
                except Exception as e:
                    continue
        
        if not step_descriptions:
            return task_id, attempt, idx, {
                'total_information_units': 'No data',
                'correctly_used_units': 'No data',
                'irr_percentage': 'No data',
                'analysis_reason': 'No step descriptions found'
            }
        
        # 应用速率限制
        rate_limiter.wait_if_needed()
        
        # 调用IRR agent进行分析
        irr_result = calculate_irr_for_task(task_desc, str(details), step_descriptions)
        
        if irr_result:
            return task_id, attempt, idx, irr_result
        else:
            return task_id, attempt, idx, {
                'total_information_units': 'Error',
                'correctly_used_units': 'Error',
                'irr_percentage': 'Error',
                'analysis_reason': 'Failed to analyze IRR'
            }
            
    except Exception as e:
        return task_id, attempt, idx, {
            'total_information_units': 'Error',
            'correctly_used_units': 'Error',
            'irr_percentage': 'Error',
            'analysis_reason': f'Processing error: {str(e)}'
        }


def process_agent_irr_parallel(agent_dir: str, max_workers: int = 2):
    """
    真正的并行IRR处理 - 支持2 QPS的并行LLM调用
    """
    results_csv_path = os.path.join(agent_dir, "results_add_irr.csv")
    if not os.path.exists(results_csv_path):
        print(f"Warning: {results_csv_path} not found")
        return
    
    # 读取结果
    df = pd.read_csv(results_csv_path)
    agent_name = get_agent_name_from_results_csv(os.path.join(agent_dir, "results.csv"))
    
    print(f"🚀 Processing agent: {agent_name} (Parallel LLM calls: {max_workers} workers)")
    
    # 找到需要详细分析的案例
    failed_cases = []
    
    for idx, row in df.iterrows():
        task_id = row['task_identifier']
        
        for attempt in [1, 2, 3]:
            irr_percentage_col = f"{agent_name}_attempt_{attempt}_irr_percentage"
            details_col = f"{agent_name}_direct_with_action_attempt_{attempt}_details"
            
            if irr_percentage_col in df.columns:
                irr_value = row[irr_percentage_col]
                if irr_value == "NEED_ANALYSIS":
                    details = row[details_col] if details_col in df.columns else ""
                    failed_cases.append((
                        agent_dir, task_id, attempt, idx, 
                        row['task_description'], details, agent_name
                    ))
    
    print(f"📊 Found {len(failed_cases)} cases needing detailed analysis")
    
    if not failed_cases:
        print("✅ No cases need analysis")
        return
    
    # 真正的并行处理（受速率限制控制）
    print(f"🧠 Starting parallel IRR analysis with rate limiting (2 QPS)...")
    start_time = time.time()
    
    completed_count = 0
    error_count = 0
    
    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_case = {
            executor.submit(analyze_irr_case_with_rate_limit, case_info): case_info
            for case_info in failed_cases
        }
        
        # 处理完成的任务
        for i, future in enumerate(as_completed(future_to_case)):
            case_info = future_to_case[future]
            _, task_id, attempt, _, _, _, _ = case_info
            
            try:
                task_id_result, attempt_result, idx_result, irr_result = future.result()
                
                # 更新DataFrame
                irr_total_col = f"{agent_name}_attempt_{attempt_result}_irr_total_units"
                irr_correct_col = f"{agent_name}_attempt_{attempt_result}_irr_correct_units"
                irr_percentage_col = f"{agent_name}_attempt_{attempt_result}_irr_percentage"
                irr_reason_col = f"{agent_name}_attempt_{attempt_result}_irr_reason"
                
                df.loc[idx_result, irr_total_col] = irr_result.get('total_information_units', 'Error')
                df.loc[idx_result, irr_correct_col] = irr_result.get('correctly_used_units', 'Error')
                df.loc[idx_result, irr_percentage_col] = irr_result.get('irr_percentage', 'Error')
                df.loc[idx_result, irr_reason_col] = irr_result.get('analysis_reason', 'Error')
                
                irr_val = irr_result.get('irr_percentage', 'Error')
                if isinstance(irr_val, (int, float)):
                    print(f"  ✅ {i+1}/{len(failed_cases)}: {task_id} attempt {attempt} -> IRR: {irr_val}%")
                    completed_count += 1
                else:
                    print(f"  ❌ {i+1}/{len(failed_cases)}: {task_id} attempt {attempt} -> Error")
                    error_count += 1
                
                # 每处理20个案例保存一次
                if (i + 1) % 20 == 0:
                    df.to_csv(results_csv_path, index=False)
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = (len(failed_cases) - i - 1) * avg_time
                    success_rate = completed_count / (i + 1) * 100
                    print(f"    💾 Progress saved. Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s, Success: {success_rate:.1f}%")
                
            except Exception as e:
                print(f"  ❌ {i+1}/{len(failed_cases)}: {task_id} attempt {attempt} -> Exception: {e}")
                error_count += 1
    
    # 最终保存和统计
    df.to_csv(results_csv_path, index=False)
    total_time = time.time() - start_time
    
    print(f"\n🎉 Completed {agent_name}!")
    print(f"  ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  🚀 Effective QPS: {len(failed_cases)/total_time:.2f}")
    print(f"  ✅ Successful analyses: {completed_count}")
    print(f"  ❌ Failed analyses: {error_count}")
    print(f"  📈 Success rate: {completed_count/(completed_count+error_count)*100:.1f}%")
    print(f"  💾 Results saved to: {results_csv_path}")


def process_all_agents_parallel():
    """并行处理所有agent"""
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
    
    print("="*80)
    print("真正的并行IRR处理器 - 支持2 QPS并行LLM调用")
    print("="*80)
    print("🚀 优化特性:")
    print("  • 并行LLM调用（2个线程，1.8 QPS限制）")
    print("  • 智能速率限制器")
    print("  • 实时进度跟踪")
    print("  • 批量保存机制")
    print()
    
    total_start = time.time()
    
    for i, agent_dir_name in enumerate(agent_dirs):
        agent_path = os.path.join(base_dir, agent_dir_name)
        if os.path.exists(agent_path):
            print(f"\n{'='*60}")
            print(f"Processing {i+1}/{len(agent_dirs)}: {agent_dir_name}")
            print(f"{'='*60}")
            try:
                process_agent_irr_parallel(agent_path, max_workers=2)  # 2个并行线程
            except Exception as e:
                print(f"❌ Error processing {agent_dir_name}: {e}")
        else:
            print(f"⚠️  Warning: {agent_path} not found")
    
    total_time = time.time() - total_start
    print(f"\n🏁 All agents completed!")
    print(f"  ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  🚀 Average QPS achieved: ~{2*0.9:.1f} (with rate limiting)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="True parallel IRR processor (2 QPS)")
    parser.add_argument("--agent_dir", type=str, help="Path to specific agent directory")
    parser.add_argument("--all", action="store_true", help="Process all agents")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel LLM workers (max 2 for 2 QPS)")
    
    args = parser.parse_args()
    
    if args.workers > 2:
        print("⚠️  Warning: API supports 2 QPS, limiting workers to 2")
        args.workers = 2
    
    if args.all:
        process_all_agents_parallel()
    elif args.agent_dir:
        if os.path.exists(args.agent_dir):
            process_agent_irr_parallel(args.agent_dir, args.workers)
        else:
            print(f"Agent directory not found: {args.agent_dir}")
    else:
        print("Usage examples:")
        print("  python3 true_parallel_irr_processor.py --all")
        print("  python3 true_parallel_irr_processor.py --agent_dir ../../results/00_baselines/250721_M3A_gemini-2.5-pro")
        print("\n🚀 真正的并行处理 - 利用2 QPS API限制！")
        print("  • 2个线程并行调用LLM")
        print("  • 智能速率限制（1.8 QPS，留有余量）")
        print("  • 预期速度提升：~2倍")
