"""
重构后的Curriculum Generator

根据exploration_output_vis数据生成高质量GUI Agent训练任务
使用统一的LLM调用进行任务评估和生成
"""

from .main import RefactoredCurriculumGenerator
from .trajectory_parser import TrajectoryParser
from .action_visualizer import ActionVisualizer
from .unified_task_processor import UnifiedTaskProcessor
from .result_saver import RefactoredResultSaver

__version__ = "2.0.0"
__author__ = "MobileForge Explore Team"

__all__ = [
    "RefactoredCurriculumGenerator",
    "TrajectoryParser",
    "ActionVisualizer", 
    "UnifiedTaskProcessor",
    "RefactoredResultSaver"
]
