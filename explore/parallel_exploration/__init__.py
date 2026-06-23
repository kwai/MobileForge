"""
并行探索模块初始化文件
"""
from .device_manager import DeviceManager, DeviceInfo
from .parallel_explorer import ParallelExplorer, run_batch_exploration

__version__ = "1.0.0"
__author__ = "MobileForge Explore Team"

__all__ = [
    "DeviceManager",
    "DeviceInfo",
    "ParallelExplorer",
    "run_batch_exploration"
]