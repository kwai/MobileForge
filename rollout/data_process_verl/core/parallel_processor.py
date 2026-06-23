#!/usr/bin/env python3
"""
MobileForge并行处理模块

负责大规模数据的并行处理，包括：
1. 多进程并行处理
2. 断点续传功能
3. 实时保存和进度跟踪
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from .processor import MobileForgeDataProcessor

logger = logging.getLogger(__name__)

def process_single_task_worker(args) -> Dict[str, Any]:
    """
    单个任务处理的工作函数（用于并行处理）
    
    Args:
        args: (task_dir_path, rollout_dir, temp_dir) 元组
        
    Returns:
        处理结果字典
    """
    task_dir_path, rollout_dir, temp_dir = args
    
    try:
        # 创建新的处理器实例（每个进程一个）
        processor = MobileForgeDataProcessor(rollout_dir, temp_dir)
        task_dir = Path(task_dir_path)
        training_samples = processor.process_single_task(task_dir)
        
        # 过滤为正负样本
        positive_samples = []
        negative_samples = []
        
        for sample in training_samples:
            if sample['success'] and sample['evaluation_result'] == 1:
                positive_samples.append(sample)
            elif sample['evaluation_result'] == 0:
                negative_samples.append(sample)
        
        return {
            'task_id': task_dir.name,
            'success': True,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'stats': {
                'successful_trajectories': len([s for s in training_samples if s['success'] and s['evaluation_result'] == 1]),
                'failed_trajectories': len([s for s in training_samples if s['evaluation_result'] == 0]),
                'error_trajectories': len([s for s in training_samples if s['evaluation_result'] not in [0, 1]]),
                'image_mapping_fixes': processor.stats.get('image_mapping_fixes', 0),
                'placeholder_replacements': processor.stats.get('placeholder_replacements', 0)
            },
            'error': None
        }
        
    except Exception as e:
        logger.error(f"处理任务 {Path(task_dir_path).name} 时出错: {e}")
        return {
            'task_id': Path(task_dir_path).name,
            'success': False,
            'positive_samples': [],
            'negative_samples': [],
            'stats': {
                'successful_trajectories': 0,
                'failed_trajectories': 0,
                'error_trajectories': 0,
                'image_mapping_fixes': 0,
                'placeholder_replacements': 0
            },
            'error': str(e)
        }


class MobileForgeParallelProcessor:
    """
    MobileForge并行处理器
    
    支持大规模数据的并行处理、断点续传和实时保存
    """
    
    def __init__(self, rollout_dir: str, base_output_dir: str = "processed_data", 
                 max_workers: int = 4, save_interval: int = 10):
        """
        初始化并行处理器
        
        Args:
            rollout_dir: rollout结果目录
            base_output_dir: 基础输出目录
            max_workers: 最大并行worker数
            save_interval: 保存间隔（处理多少个任务后保存一次）
        """
        self.rollout_dir = Path(rollout_dir)
        self.base_output_dir = Path(base_output_dir)
        self.max_workers = max_workers
        self.save_interval = save_interval
        
        # 创建基础输出目录
        self.base_output_dir.mkdir(exist_ok=True)
        
        # 为本次运行创建带时间戳的会话目录
        import pandas as pd
        self.session_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_output_dir / f"session_{self.session_timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        
        # 创建临时保存目录
        self.temp_dir = self.session_dir / "temp_results"
        self.temp_dir.mkdir(exist_ok=True)
        
        # 获取所有任务目录
        self.all_task_dirs = [d for d in self.rollout_dir.glob("*") if d.is_dir()]
        
        # 统计信息
        self.global_stats = {
            'total_tasks': len(self.all_task_dirs),
            'processed_tasks': 0,
            'successful_trajectories': 0,
            'failed_trajectories': 0,
            'error_trajectories': 0,
            'processing_errors': 0,
            'image_mapping_fixes': 0,
            'placeholder_replacements': 0
        }
        
        # 存储处理结果
        self.all_positive_samples = []
        self.all_negative_samples = []
        
        logger.info(f"初始化并行处理器")
        logger.info(f"任务数: {len(self.all_task_dirs)}, Workers: {max_workers}")
        logger.info(f"会话目录: {self.session_dir}")
    
    def save_intermediate_results(self, timestamp: str = None) -> None:
        """
        保存中间结果
        
        Args:
            timestamp: 时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            import pandas as pd
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 保存正样本
            if self.all_positive_samples:
                positive_file = self.temp_dir / f"intermediate_positive_{timestamp}.json"
                with open(positive_file, 'w', encoding='utf-8') as f:
                    json.dump(self.all_positive_samples, f, indent=2, ensure_ascii=False)
                logger.info(f"已保存中间正样本: {len(self.all_positive_samples)} 个")
            
            # 保存负样本
            if self.all_negative_samples:
                negative_file = self.temp_dir / f"intermediate_negative_{timestamp}.json"
                with open(negative_file, 'w', encoding='utf-8') as f:
                    json.dump(self.all_negative_samples, f, indent=2, ensure_ascii=False)
                logger.info(f"已保存中间负样本: {len(self.all_negative_samples)} 个")
            
            # 保存统计信息
            stats_file = self.temp_dir / f"intermediate_stats_{timestamp}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.global_stats, f, indent=2)
            logger.info(f"已保存中间统计信息")
            
        except Exception as e:
            logger.error(f"保存中间结果时出错: {e}")
    
    def load_checkpoint(self) -> List[str]:
        """
        加载检查点，返回已处理的任务列表
        
        Returns:
            已处理任务ID列表
        """
        processed_tasks = []
        
        # 查找所有中间结果文件
        intermediate_files = list(self.temp_dir.glob("intermediate_positive_*.json"))
        
        if intermediate_files:
            # 找到最新的中间结果文件
            latest_file = max(intermediate_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"找到检查点文件: {latest_file}")
            
            try:
                # 加载已处理的样本
                with open(latest_file, 'r', encoding='utf-8') as f:
                    self.all_positive_samples = json.load(f)
                
                # 尝试加载对应的负样本
                timestamp = latest_file.stem.replace('intermediate_positive_', '')
                negative_file = self.temp_dir / f"intermediate_negative_{timestamp}.json"
                if negative_file.exists():
                    with open(negative_file, 'r', encoding='utf-8') as f:
                        self.all_negative_samples = json.load(f)
                
                # 尝试加载统计信息
                stats_file = self.temp_dir / f"intermediate_stats_{timestamp}.json"
                if stats_file.exists():
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        self.global_stats.update(json.load(f))
                
                # 提取已处理的任务ID
                processed_tasks = list(set([
                    sample['task_id'] for sample in self.all_positive_samples + self.all_negative_samples
                ]))
                
                logger.info(f"从检查点恢复: {len(processed_tasks)} 个已处理任务")
                
            except Exception as e:
                logger.error(f"加载检查点时出错: {e}")
                processed_tasks = []
        
        return processed_tasks
    
    def process_all_tasks_parallel(self, max_tasks: Optional[int] = None, 
                                 resume: bool = True) -> Dict[str, Any]:
        """
        并行处理所有任务，支持实时保存和断点续传
        
        Args:
            max_tasks: 最大处理任务数
            resume: 是否从检查点继续
            
        Returns:
            处理结果字典
        """
        logger.info("开始并行处理所有任务（图像映射修复版本）")
        
        # 加载检查点
        processed_task_ids = []
        if resume:
            processed_task_ids = self.load_checkpoint()
        
        # 过滤未处理的任务
        task_dirs_to_process = [
            task_dir for task_dir in self.all_task_dirs
            if task_dir.name not in processed_task_ids
        ]
        
        if max_tasks:
            remaining_slots = max_tasks - len(processed_task_ids)
            if remaining_slots > 0:
                task_dirs_to_process = task_dirs_to_process[:remaining_slots]
            else:
                task_dirs_to_process = []
        
        logger.info(f"需要处理的任务: {len(task_dirs_to_process)} 个")
        logger.info(f"已从检查点恢复: {len(processed_task_ids)} 个任务")
        
        if not task_dirs_to_process:
            logger.info("没有新任务需要处理")
            return {
                'positive_samples': self.all_positive_samples,
                'negative_samples': self.all_negative_samples,
                'all_samples': self.all_positive_samples + self.all_negative_samples,
                'statistics': self.global_stats
            }
        
        # 并行处理任务
        processed_count = len(processed_task_ids)
        error_count = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(process_single_task_worker, 
                              (str(task_dir), str(self.rollout_dir), str(self.temp_dir))): task_dir
                for task_dir in task_dirs_to_process
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                task_dir = future_to_task[future]
                
                try:
                    result = future.result()
                    
                    if result['success']:
                        # 累加样本
                        self.all_positive_samples.extend(result['positive_samples'])
                        self.all_negative_samples.extend(result['negative_samples'])
                        
                        # 更新统计信息
                        self.global_stats['successful_trajectories'] += result['stats']['successful_trajectories']
                        self.global_stats['failed_trajectories'] += result['stats']['failed_trajectories']
                        self.global_stats['error_trajectories'] += result['stats']['error_trajectories']
                        self.global_stats['image_mapping_fixes'] += result['stats']['image_mapping_fixes']
                        self.global_stats['placeholder_replacements'] += result['stats']['placeholder_replacements']
                        
                        logger.info(f"✓ 完成任务 {result['task_id']}: "
                                   f"正样本+{len(result['positive_samples'])}, "
                                   f"负样本+{len(result['negative_samples'])}")
                    else:
                        error_count += 1
                        self.global_stats['processing_errors'] += 1
                        logger.error(f"✗ 任务 {result['task_id']} 处理失败: {result['error']}")
                    
                    processed_count += 1
                    self.global_stats['processed_tasks'] = processed_count
                    
                    # 实时保存
                    if processed_count % self.save_interval == 0:
                        import pandas as pd
                        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        self.save_intermediate_results(timestamp)
                        logger.info(f"已处理 {processed_count}/{self.global_stats['total_tasks']} 个任务")
                    
                except Exception as e:
                    error_count += 1
                    self.global_stats['processing_errors'] += 1
                    logger.error(f"处理任务 {task_dir.name} 时发生异常: {e}")
        
        # 最终保存
        import pandas as pd
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.save_intermediate_results(timestamp)
        
        logger.info(f"并行处理完成: 成功 {processed_count - error_count}, 失败 {error_count}")
        
        return {
            'positive_samples': self.all_positive_samples,
            'negative_samples': self.all_negative_samples,
            'all_samples': self.all_positive_samples + self.all_negative_samples,
            'statistics': self.global_stats
        }
    
    def get_session_info(self) -> Dict[str, str]:
        """获取当前会话信息"""
        return {
            'session_timestamp': self.session_timestamp,
            'session_dir': str(self.session_dir),
            'base_output_dir': str(self.base_output_dir),
            'temp_dir': str(self.temp_dir)
        }


def process_parallel(rollout_dir: str, output_dir: str = "processed_data", 
                   max_tasks: Optional[int] = None, max_workers: int = 4,
                   save_interval: int = 10, resume: bool = True) -> Dict[str, Any]:
    """
    并行处理所有任务的便捷函数
    
    Args:
        rollout_dir: rollout结果目录
        output_dir: 输出目录
        max_tasks: 最大处理任务数
        max_workers: 最大并行worker数
        save_interval: 保存间隔（多少个任务后保存一次）
        resume: 是否从检查点继续
        
    Returns:
        处理结果字典
    """
    logger.info(f"开始并行处理: {rollout_dir} -> {output_dir}")
    logger.info(f"参数: max_tasks={max_tasks}, max_workers={max_workers}, "
                f"save_interval={save_interval}, resume={resume}")
    
    # 创建并行处理器
    parallel_processor = MobileForgeParallelProcessor(
        rollout_dir=rollout_dir,
        base_output_dir=output_dir,
        max_workers=max_workers,
        save_interval=save_interval
    )
    
    # 执行并行处理
    start_time = time.time()
    processed_data = parallel_processor.process_all_tasks_parallel(
        max_tasks=max_tasks,
        resume=resume
    )
    processing_time = time.time() - start_time
    
    # 更新处理时间统计
    processed_data['statistics']['processing_time'] = processing_time
    processed_data['statistics']['avg_time_per_task'] = (
        processing_time / max(1, processed_data['statistics']['processed_tasks'])
    )
    
    # 记录会话信息
    processed_data['session_info'] = parallel_processor.get_session_info()
    
    # 使用数据保存器生成最终的GRPO格式文件
    from .data_saver import MobileForgeDataSaver
    
    # 创建数据保存器，使用并行处理器的会话目录
    data_saver = MobileForgeDataSaver(str(parallel_processor.session_dir.parent))
    # 手动设置时间戳和会话目录以保持一致
    data_saver.timestamp = parallel_processor.session_timestamp
    data_saver.session_dir = parallel_processor.session_dir
    
    # 保存最终的训练数据
    saved_files = data_saver.save_training_data(processed_data, format_type="grpo")
    data_saver.save_session_summary(processed_data)
    
    # 更新处理结果
    processed_data['saved_files'] = saved_files
    processed_data['session_info'] = data_saver.get_session_info()
    
    logger.info(f"并行处理完成，耗时: {processing_time:.2f} 秒")
    logger.info(f"平均每任务: {processed_data['statistics']['avg_time_per_task']:.2f} 秒")
    
    return processed_data
