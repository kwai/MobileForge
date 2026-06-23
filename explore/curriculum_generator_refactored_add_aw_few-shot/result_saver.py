"""
结果保存器模块

负责保存处理结果和生成的任务数据
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import csv


class RefactoredResultSaver:
    """重构后的结果保存器类"""
    
    def __init__(self):
        """初始化结果保存器"""
        pass
    
    def save_app_results(self, app_package: str, app_name: str, 
                        results: List[Dict[str, Any]], output_dir: Path) -> None:
        """
        保存应用的所有处理结果
        
        Args:
            app_package: 应用包名
            app_name: 应用名称
            results: 处理结果列表
            output_dir: 输出目录路径
        """
        # 创建应用专用输出目录
        app_output_dir = output_dir / app_package
        app_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始结果JSON
        self._save_raw_results(results, app_output_dir)
        
        # 保存任务CSV
        self._save_tasks_csv(results, app_output_dir)
        
        # 保存统计信息
        self._save_statistics(app_package, app_name, results, app_output_dir)
        
        # 保存摘要报告
        self._save_summary_report(app_package, app_name, results, app_output_dir)
        
        print(f"\nResults saved to: {app_output_dir}")
    
    def _save_raw_results(self, results: List[Dict[str, Any]], output_dir: Path) -> None:
        """保存原始结果JSON"""
        raw_results_path = output_dir / "raw_results.json"
        
        try:
            with open(raw_results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved raw results: {raw_results_path}")
        except Exception as e:
            print(f"Failed to save raw results: {e}")
    
    def _save_tasks_csv(self, results: List[Dict[str, Any]], output_dir: Path) -> None:
        """保存生成的任务CSV"""
        tasks_csv_path = output_dir / "generated_tasks.csv"
        
        try:
            with open(tasks_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                # 用户要求的三个核心列 + 其他辅助列（移除difficulty_level）
                fieldnames = [
                    'task_identifier', 'task_description', 'golden_steps',  # 用户要求的核心列
                    'trajectory_id', 'original_goal', 'task_reasonable', 'task_completed',
                    'task_id', 'core_functionality', 'variation_type', 'prerequisites'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    trajectory_id = result['trajectory_id']
                    original_goal = result['original_goal']
                    evaluation = result['evaluation']
                    task_reasonable = evaluation.get('task_reasonable', False)
                    task_completed = evaluation.get('task_completed', False)
                    
                    generated_tasks = result.get('generated_tasks', [])
                    
                    if generated_tasks:
                        for i, task in enumerate(generated_tasks, 1):
                            # 生成task_identifier: 原始trajectory_id + 序号
                            task_identifier = f"{trajectory_id}-{i:03d}"
                            
                            row = {
                                'task_identifier': task_identifier,  # 唯一标识符
                                'task_description': task.get('instruction', ''),  # 任务描述
                                'golden_steps': task.get('estimated_steps', 0),  # 预估步数
                                'trajectory_id': trajectory_id,
                                'original_goal': original_goal,
                                'task_reasonable': task_reasonable,
                                'task_completed': task_completed,
                                'task_id': task.get('task_id', ''),
                                'core_functionality': task.get('core_functionality', ''),
                                'variation_type': task.get('variation_type', ''),
                                'prerequisites': task.get('prerequisites', 'None')
                            }
                            writer.writerow(row)
                    else:
                        # 即使没有生成任务，也记录原始任务信息
                        task_identifier = f"{trajectory_id}-000"
                        
                        row = {
                            'task_identifier': task_identifier,
                            'task_description': original_goal,  # 使用原始目标作为描述
                            'golden_steps': 0,  # 没有预估步数
                            'trajectory_id': trajectory_id,
                            'original_goal': original_goal,
                            'task_reasonable': task_reasonable,
                            'task_completed': task_completed,
                            'task_id': '',
                            'core_functionality': '',
                            'variation_type': '',
                            'prerequisites': ''
                        }
                        writer.writerow(row)
            
            print(f"Saved tasks CSV: {tasks_csv_path}")
        except Exception as e:
            print(f"Failed to save tasks CSV: {e}")
    
    def _save_statistics(self, app_package: str, app_name: str, 
                        results: List[Dict[str, Any]], output_dir: Path) -> None:
        """保存统计信息"""
        stats_path = output_dir / "statistics.json"
        
        # 计算统计信息
        total_trajectories = len(results)
        reasonable_tasks = sum(1 for r in results if r['evaluation'].get('task_reasonable', False))
        completed_tasks = sum(1 for r in results if r['evaluation'].get('task_completed', False))
        total_generated_tasks = sum(len(r.get('generated_tasks', [])) for r in results)
        
        # 按难度分组统计
        difficulty_stats = {'easy': 0, 'medium': 0, 'hard': 0}
        for result in results:
            for task in result.get('generated_tasks', []):
                difficulty = task.get('difficulty_level', '')
                if difficulty in difficulty_stats:
                    difficulty_stats[difficulty] += 1
        
        # 按变化类型分组统计
        variation_stats = {}
        for result in results:
            for task in result.get('generated_tasks', []):
                variation_type = task.get('variation_type', 'unknown')
                variation_stats[variation_type] = variation_stats.get(variation_type, 0) + 1
        
        statistics = {
            'app_package': app_package,
            'app_name': app_name,
            'processing_timestamp': self.get_timestamp(),
            'total_trajectories': total_trajectories,
            'reasonable_tasks': reasonable_tasks,
            'completed_tasks': completed_tasks,
            'total_generated_tasks': total_generated_tasks,
            'reasonable_task_rate': reasonable_tasks / total_trajectories if total_trajectories > 0 else 0,
            'completion_rate': completed_tasks / total_trajectories if total_trajectories > 0 else 0,
            'avg_tasks_per_trajectory': total_generated_tasks / total_trajectories if total_trajectories > 0 else 0,
            'difficulty_distribution': difficulty_stats,
            'variation_type_distribution': variation_stats
        }
        
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, indent=2, ensure_ascii=False)
            print(f"Saved statistics: {stats_path}")
        except Exception as e:
            print(f"Failed to save statistics: {e}")
    
    def _save_summary_report(self, app_package: str, app_name: str, 
                           results: List[Dict[str, Any]], output_dir: Path) -> None:
        """保存摘要报告"""
        report_path = output_dir / "summary_report.md"
        
        total_trajectories = len(results)
        reasonable_tasks = sum(1 for r in results if r['evaluation'].get('task_reasonable', False))
        completed_tasks = sum(1 for r in results if r['evaluation'].get('task_completed', False))
        total_generated_tasks = sum(len(r.get('generated_tasks', [])) for r in results)
        
        report = f"""# {app_name} 任务生成报告

## 基本信息
- **应用包名**: {app_package}
- **应用名称**: {app_name}
- **处理时间**: {self.get_timestamp()}

## 处理统计
- **总轨迹数**: {total_trajectories}
- **合理任务数**: {reasonable_tasks} ({reasonable_tasks/total_trajectories*100:.1f}%)
- **完成任务数**: {completed_tasks} ({completed_tasks/total_trajectories*100:.1f}%)
- **生成任务总数**: {total_generated_tasks}
- **平均每轨迹生成任务数**: {total_generated_tasks/total_trajectories:.1f}

## 生成任务示例

"""
        
        # 添加一些任务示例
        example_count = 0
        for result in results:
            if example_count >= 5:
                break
            
            generated_tasks = result.get('generated_tasks', [])
            if generated_tasks:
                original_goal = result['original_goal']
                report += f"### 原始任务: {original_goal}\n\n"
                
                for i, task in enumerate(generated_tasks[:3], 1):
                    instruction = task.get('instruction', '')
                    difficulty = task.get('difficulty_level', '')
                    steps = task.get('estimated_steps', 0)
                    report += f"{i}. **{instruction}** (难度: {difficulty}, 预估步数: {steps})\n"
                
                report += "\n"
                example_count += 1
        
        report += f"""
## 数据文件
- `raw_results.json`: 完整的处理结果数据
- `generated_tasks.csv`: 生成的任务列表（可用于训练）
- `statistics.json`: 详细统计信息

---
*报告生成时间: {self.get_timestamp()}*
"""
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Saved summary report: {report_path}")
        except Exception as e:
            print(f"Failed to save summary report: {e}")
    
    def get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def save_single_trajectory_result(self, result: Dict[str, Any], 
                                    output_dir: Path) -> None:
        """
        保存单个轨迹的处理结果
        
        Args:
            result: 单个轨迹的处理结果
            output_dir: 输出目录
        """
        trajectory_id = result.get('trajectory_id', 'unknown')
        result_path = output_dir / f"trajectory_{trajectory_id}.json"
        
        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"保存轨迹结果: {result_path}")
        except Exception as e:
            print(f"保存轨迹结果失败: {e}")
    
    def load_existing_results(self, output_dir: Path) -> List[Dict[str, Any]]:
        """
        加载已存在的处理结果
        
        Args:
            output_dir: 输出目录
            
        Returns:
            已存在的结果列表
        """
        raw_results_path = output_dir / "raw_results.json"
        
        if not raw_results_path.exists():
            return []
        
        try:
            with open(raw_results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"加载了 {len(results)} 个已存在的结果")
            return results
        except Exception as e:
            print(f"加载已存在结果失败: {e}")
            return []
    
    def create_master_summary(self, base_output_dir: Path) -> None:
        """
        创建所有应用的主汇总报告和全局任务CSV
        
        Args:
            base_output_dir: 基础输出目录
        """
        # 创建主汇总报告
        self._create_master_summary_report(base_output_dir)
        
        # 创建全局任务汇总CSV
        self._create_global_tasks_csv(base_output_dir)
    
    def _create_master_summary_report(self, base_output_dir: Path) -> None:
        """创建主汇总报告"""
        # 检查目录是否存在
        if not base_output_dir.exists():
            print(f"输出目录不存在: {base_output_dir}")
            return
        
        summary_path = base_output_dir / "master_summary.md"
        
        # 收集所有应用的统计信息
        app_stats = []
        
        for app_dir in base_output_dir.iterdir():
            if app_dir.is_dir():
                stats_file = app_dir / "statistics.json"
                if stats_file.exists():
                    try:
                        with open(stats_file, 'r', encoding='utf-8') as f:
                            stats = json.load(f)
                        app_stats.append(stats)
                    except Exception as e:
                        print(f"读取统计文件失败 {stats_file}: {e}")
        
        if not app_stats:
            print("没有找到任何应用统计信息")
            return
        
        # 生成主汇总报告
        total_apps = len(app_stats)
        total_trajectories = sum(s.get('total_trajectories', 0) for s in app_stats)
        total_generated_tasks = sum(s.get('total_generated_tasks', 0) for s in app_stats)
        
        report = f"""# 重构后的课程生成器 - 主汇总报告

## 总体统计
- **处理应用数**: {total_apps}
- **总轨迹数**: {total_trajectories}
- **总生成任务数**: {total_generated_tasks}
- **平均每应用生成任务数**: {total_generated_tasks/total_apps:.1f}

## 各应用详情

| 应用名称 | 轨迹数 | 生成任务数 | 合理率 | 完成率 |
|---------|--------|------------|--------|--------|
"""
        
        for stats in app_stats:
            app_name = stats.get('app_name', stats.get('app_package', 'Unknown'))
            trajectories = stats.get('total_trajectories', 0)
            generated_tasks = stats.get('total_generated_tasks', 0)
            reasonable_rate = stats.get('reasonable_task_rate', 0) * 100
            completion_rate = stats.get('completion_rate', 0) * 100
            
            report += f"| {app_name} | {trajectories} | {generated_tasks} | {reasonable_rate:.1f}% | {completion_rate:.1f}% |\n"
        
        report += f"""

## 数据使用说明

生成的任务数据保存在各应用目录下的 `generated_tasks.csv` 文件中，可直接用于：
1. GUI Agent 模型训练
2. 任务难度分析
3. 功能覆盖评估

---
*报告生成时间: {self.get_timestamp()}*
"""
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"保存主汇总报告: {summary_path}")
        except Exception as e:
            print(f"保存主汇总报告失败: {e}")
    
    def _create_global_tasks_csv(self, base_output_dir: Path) -> None:
        """创建全局任务汇总CSV"""
        # 检查目录是否存在
        if not base_output_dir.exists():
            print(f"输出目录不存在: {base_output_dir}")
            return
        
        global_csv_path = base_output_dir / "all_generated_tasks.csv"
        
        try:
            with open(global_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                # 保持与单应用CSV相同的列顺序，核心列在前
                fieldnames = [
                    'task_identifier', 'task_description', 'golden_steps',  # 用户要求的核心列
                    'app_package', 'app_name', 'trajectory_id', 'original_goal', 
                    'task_reasonable', 'task_completed', 'task_id', 
                    'difficulty_level', 'core_functionality', 'variation_type', 'prerequisites'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # 遍历所有应用目录
                for app_dir in base_output_dir.iterdir():
                    if not app_dir.is_dir():
                        continue
                    
                    app_package = app_dir.name
                    
                    # 读取应用的生成任务CSV
                    app_csv_path = app_dir / "generated_tasks.csv"
                    if not app_csv_path.exists():
                        continue
                    
                    # 读取应用名称
                    app_name = "Unknown"
                    stats_file = app_dir / "statistics.json"
                    if stats_file.exists():
                        try:
                            with open(stats_file, 'r', encoding='utf-8') as f:
                                stats = json.load(f)
                            app_name = stats.get('app_name', app_package)
                        except Exception as e:
                            print(f"读取应用名称失败 {stats_file}: {e}")
                    
                    # 读取并合并任务数据
                    try:
                        with open(app_csv_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                # 合并应用信息和任务数据，确保包含核心列
                                global_row = {
                                    'task_identifier': row.get('task_identifier', ''),
                                    'task_description': row.get('task_description', ''),
                                    'golden_steps': row.get('golden_steps', ''),
                                    'app_package': app_package,
                                    'app_name': app_name,
                                    'trajectory_id': row.get('trajectory_id', ''),
                                    'original_goal': row.get('original_goal', ''),
                                    'task_reasonable': row.get('task_reasonable', ''),
                                    'task_completed': row.get('task_completed', ''),
                                    'task_id': row.get('task_id', ''),
                                    'difficulty_level': row.get('difficulty_level', ''),
                                    'core_functionality': row.get('core_functionality', ''),
                                    'variation_type': row.get('variation_type', ''),
                                    'prerequisites': row.get('prerequisites', '')
                                }
                                writer.writerow(global_row)
                    except Exception as e:
                        print(f"读取应用CSV失败 {app_csv_path}: {e}")
                        continue
            
            print(f"保存全局任务汇总CSV: {global_csv_path}")
            
        except Exception as e:
            print(f"创建全局任务汇总CSV失败: {e}")
