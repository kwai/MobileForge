"""
MobileForge 数据分析器模块

交互式 rollout 数据分析工具，支持：
  - 多维度数据统计 (任务/轨迹/步骤)
  - 按 App 分组分析
  - 8 种可组合的筛选策略
  - 生成可交互 HTML 仪表板报告

用法 (在项目根目录 MobileForge Rollout 下运行):
  python -m data_analyzer --rollout_dir /path/to/session
"""

from .loader import RolloutDataLoader
from .metrics import MetricsComputer
from .filters import DataFilter
from .report import HTMLReportGenerator

__all__ = ["RolloutDataLoader", "MetricsComputer", "DataFilter", "HTMLReportGenerator"]
__version__ = "2.0.0"
