"""
轨迹解析器模块

负责解析exploration_output_vis_25100701中的轨迹数据和截图信息
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import glob


class TrajectoryParser:
    """轨迹解析器类"""
    
    def __init__(self, vis_data_dir: Path):
        """
        初始化轨迹解析器
        
        Args:
            vis_data_dir: 可视化数据目录路径
        """
        self.vis_data_dir = vis_data_dir
    
    def parse_app_info(self, app_package: str) -> Optional[Dict[str, Any]]:
        """
        解析应用信息
        
        Args:
            app_package: 应用包名
            
        Returns:
            应用信息字典，如果解析失败返回None
        """
        app_info_path = self.vis_data_dir / app_package / "app_info.json"
        
        if not app_info_path.exists():
            print(f"应用信息文件不存在: {app_info_path}")
            return None
        
        try:
            with open(app_info_path, 'r', encoding='utf-8') as f:
                app_info = json.load(f)
            return app_info
        except Exception as e:
            print(f"解析应用信息失败: {e}")
            return None
    
    def get_all_trajectories(self, app_package: str) -> List[Path]:
        """
        获取指定应用的所有轨迹文件
        
        Args:
            app_package: 应用包名
            
        Returns:
            轨迹文件路径列表
        """
        trajectories_dir = self.vis_data_dir / app_package / "trajectories"
        
        if not trajectories_dir.exists():
            print(f"轨迹目录不存在: {trajectories_dir}")
            return []
        
        # 查找所有JSON轨迹文件
        trajectory_files = list(trajectories_dir.glob("*.json"))
        trajectory_files.sort()  # 按文件名排序
        
        print(f"Found {len(trajectory_files)} trajectory files")
        return trajectory_files
    
    def parse_trajectory(self, trajectory_file: Path) -> Optional[Dict[str, Any]]:
        """
        解析单个轨迹文件
        
        Args:
            trajectory_file: 轨迹文件路径
            
        Returns:
            轨迹数据字典，如果解析失败返回None
        """
        try:
            with open(trajectory_file, 'r', encoding='utf-8') as f:
                trajectory_data = json.load(f)
            return trajectory_data
        except Exception as e:
            print(f"解析轨迹文件失败 {trajectory_file}: {e}")
            return None
    
    def get_trajectory_screenshots(self, app_package: str, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """
        获取轨迹对应的截图信息
        
        Args:
            app_package: 应用包名
            trajectory_id: 轨迹ID
            
        Returns:
            截图信息字典，包含每个步骤的截图路径
        """
        screenshots_dir = self.vis_data_dir / app_package / "screenshots" / trajectory_id
        
        if not screenshots_dir.exists():
            print(f"截图目录不存在: {screenshots_dir}")
            return None
        
        screenshot_info = {
            "trajectory_id": trajectory_id,
            "screenshots_dir": screenshots_dir,
            "steps": {}
        }
        
        # 查找所有步骤目录
        step_dirs = [d for d in screenshots_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
        step_dirs.sort()  # 按步骤顺序排序
        
        for step_dir in step_dirs:
            step_name = step_dir.name
            step_index = int(step_name.split("_")[1])
            
            # 查找before_screenshot.png
            before_screenshot = step_dir / "before_screenshot.png"
            
            if before_screenshot.exists():
                screenshot_info["steps"][step_index] = {
                    "step_dir": step_dir,
                    "before_screenshot": before_screenshot,
                    "step_name": step_name
                }
        
        print(f"Found {len(screenshot_info['steps'])} step screenshots")
        return screenshot_info
    
    def extract_converted_actions(self, trajectory_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从轨迹数据中提取转换后的动作信息
        
        Args:
            trajectory_data: 轨迹数据
            
        Returns:
            动作信息列表
        """
        actions = []
        steps = trajectory_data.get("steps", [])
        
        for step in steps:
            step_index = step.get("step_index", 0)
            
            # 从metadata中提取动作信息
            metadata = step.get("metadata", {})
            
            # 提取转换后的动作坐标
            actual_coordinates = metadata.get("actual_action_coordinates")
            target_element = metadata.get("target_element")
            
            # 构建动作信息
            action_info = {
                "step_index": step_index,
                "action_type": "unknown",
                "coordinates": actual_coordinates,
                "target_element": target_element,
                "summary": step.get("summary", "")
            }
            
            # 尝试从summary中解析动作类型
            summary = step.get("summary", "")
            if "click" in summary.lower():
                action_info["action_type"] = "click"
            elif "swipe" in summary.lower():
                action_info["action_type"] = "swipe"
            elif "type" in summary.lower() or "input" in summary.lower():
                action_info["action_type"] = "type"
            elif "scroll" in summary.lower():
                action_info["action_type"] = "scroll"
            
            actions.append(action_info)
        
        return actions
    
    def get_trajectory_summary(self, trajectory_data: Dict[str, Any]) -> str:
        """
        生成轨迹摘要
        
        Args:
            trajectory_data: 轨迹数据
            
        Returns:
            轨迹摘要文本
        """
        goal = trajectory_data.get("goal", "")
        steps = trajectory_data.get("steps", [])
        
        summary_parts = [f"目标: {goal}"]
        
        for i, step in enumerate(steps):
            step_summary = step.get("summary", f"步骤 {i+1}")
            summary_parts.append(f"步骤 {i+1}: {step_summary}")
        
        return "\n".join(summary_parts)
    
    def validate_trajectory_data(self, trajectory_data: Dict[str, Any]) -> bool:
        """
        验证轨迹数据的完整性
        
        Args:
            trajectory_data: 轨迹数据
            
        Returns:
            数据是否有效
        """
        required_fields = ["trajectory_id", "package_name", "goal", "steps"]
        
        for field in required_fields:
            if field not in trajectory_data:
                print(f"轨迹数据缺少必需字段: {field}")
                return False
        
        steps = trajectory_data.get("steps", [])
        if not steps:
            print("轨迹数据没有步骤信息")
            return False
        
        return True
    
    def get_app_statistics(self, app_package: str) -> Dict[str, Any]:
        """
        获取应用的统计信息
        
        Args:
            app_package: 应用包名
            
        Returns:
            统计信息字典
        """
        statistics_path = self.vis_data_dir / app_package / "statistics.json"
        
        if not statistics_path.exists():
            return {}
        
        try:
            with open(statistics_path, 'r', encoding='utf-8') as f:
                statistics = json.load(f)
            return statistics
        except Exception as e:
            print(f"读取统计信息失败: {e}")
            return {}
