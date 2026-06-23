"""
AndroidWorld任务加载器

负责从Excel文件中加载AndroidWorld任务，并根据应用名称匹配对应的任务作为few-shot示例
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class AndroidWorldTaskLoader:
    """AndroidWorld任务加载器类"""
    
    def __init__(self, excel_path: str = "251103-android-world-tasks-to-app.xlsx"):
        """
        初始化AndroidWorld任务加载器
        
        Args:
            excel_path: Excel文件路径
        """
        self.excel_path = Path(excel_path)
        self.tasks_df = None
        self._load_tasks()  # 先加载任务数据
        self.app_name_mapping = self._build_app_name_mapping()  # 再构建映射
    
    def _get_exploration_apps(self, exploration_dir: str = "../exploration_output_vis/exploration_output_vis_25102901") -> List[str]:
        """
        从exploration目录中读取实际的应用包名列表
        
        Args:
            exploration_dir: exploration数据目录路径
            
        Returns:
            应用包名列表
        """
        exploration_path = Path(exploration_dir)
        if not exploration_path.exists():
            print(f"Warning: Exploration directory not found: {exploration_path}")
            return []
        
        apps = []
        for app_dir in exploration_path.iterdir():
            if app_dir.is_dir() and not app_dir.name.startswith('.'):
                apps.append(app_dir.name)
        
        return sorted(apps)

    def _build_app_name_mapping(self) -> Dict[str, str]:
        """
        构建应用名称映射表，现在Excel和exploration目录中都是完整包名
        主要用于支持应用名称的各种变体
        
        Returns:
            应用名称映射字典
        """
        # 1. 从Excel中获取AndroidWorld应用列表
        android_world_apps = []
        if self.tasks_df is not None:
            android_world_apps = sorted(self.tasks_df['main_app'].unique())
            print(f"Found {len(android_world_apps)} AndroidWorld apps: {android_world_apps}")
        else:
            print("Warning: tasks_df is None, cannot read AndroidWorld apps")
        
        # 2. 从exploration目录中获取实际的应用包名
        exploration_apps = self._get_exploration_apps()
        print(f"Found {len(exploration_apps)} exploration apps: {exploration_apps}")
        
        # 3. 现在Excel和exploration目录中的应用名应该完全一致，创建简单的映射
        mapping = {}
        
        # 为每个应用包名创建多种变体的映射
        all_apps = set(android_world_apps + exploration_apps)
        
        for app_package in all_apps:
            # 完整包名映射到自身
            mapping[app_package.lower()] = app_package
            
            # 包名最后部分的映射
            if '.' in app_package:
                last_part = app_package.split('.')[-1]
                mapping[last_part.lower()] = app_package
            
            # 一些常见的应用名称变体
            app_name_variants = {
                "com.android.camera2": ["camera", "camera2"],
                "com.android.chrome": ["chrome", "chorme"],  # 包含拼写变体
                "com.android.settings": ["settings"],
                "com.arduia.expense": ["expense", "pro expense"],
                "com.dimowner.audiorecorder": ["audiorecorder", "audio recorder"],
                "com.flauschcode.broccoli": ["broccoli"],
                "com.google.android.contacts": ["contacts"],
                "com.google.android.deskclock": ["clock", "deskclock"],
                "com.google.android.documentsui": ["documentsui", "files"],
                "com.simplemobiletools.calendar.pro": ["calendar", "simple calendar"],
                "com.simplemobiletools.draw.pro": ["draw", "simple draw"],
                "com.simplemobiletools.gallery.pro": ["gallery", "simple gallery"],
                "com.simplemobiletools.smsmessenger": ["smsmessenger", "sms"],
                "de.dennisguse.opentracks": ["opentracks"],
                "net.cozic.joplin": ["joplin"],
                "net.gsantner.markor": ["markor"],
                "net.osmand": ["osmand"],
                "org.tasks": ["tasks"],
                "org.videolan.vlc": ["vlc"],
                "code.name.monkey.retromusic": ["retromusic", "retro music"]
            }
            
            # 添加预定义的变体
            if app_package in app_name_variants:
                for variant in app_name_variants[app_package]:
                    mapping[variant.lower()] = app_package
        
        print(f"Built mapping for {len(mapping)} app name variations")
        
        return mapping
    
    def _load_tasks(self):
        """加载Excel文件中的任务数据"""
        if not self.excel_path.exists():
            print(f"Warning: AndroidWorld Excel file not found: {self.excel_path}")
            return
        
        try:
            # 读取Excel文件的第一个sheet
            self.tasks_df = pd.read_excel(self.excel_path, sheet_name=0)
            print(f"Loaded {len(self.tasks_df)} AndroidWorld tasks from {self.excel_path}")
            
            # 打印列名以便调试
            print(f"Excel columns: {list(self.tasks_df.columns)}")
            
        except Exception as e:
            print(f"Error loading AndroidWorld Excel file: {e}")
            self.tasks_df = None
    
    def get_app_tasks(self, app_package: str, app_name: str = None) -> List[Dict[str, Any]]:
        """
        根据应用包名或应用名称获取对应的AndroidWorld任务
        
        Args:
            app_package: 应用包名，如 "com.android.camera2"
            app_name: 应用名称，如 "Camera"
            
        Returns:
            匹配的AndroidWorld任务列表
        """
        if self.tasks_df is None:
            return []
        
        matched_tasks = []
        
        print(f"Searching AndroidWorld tasks for app: {app_package}")
        
        # 在tasks_df中搜索匹配的任务 - 使用精确匹配
        for _, row in self.tasks_df.iterrows():
            main_app = str(row.get('main_app', ''))
            apps = str(row.get('apps', ''))
            task_name = str(row.get('task_name', ''))
            goal = str(row.get('goal', ''))
            
            # 检查是否匹配 - 使用精确包名匹配
            matched = False
            
            # 1. 检查main_app列是否精确匹配
            if main_app == app_package:
                matched = True
            
            # 2. 检查apps列中是否包含该应用（处理多个app的情况）
            if not matched and apps:
                # 将apps按逗号分割，去除空格后检查
                app_list = [app.strip() for app in apps.split(',')]
                if app_package in app_list:
                    matched = True
            
            if matched:
                task_info = {
                    'task_name': task_name,
                    'goal': goal,
                    'main_app': main_app,
                    'apps': apps
                }
                matched_tasks.append(task_info)
        
        print(f"Found {len(matched_tasks)} matching AndroidWorld tasks")
        
        # 打印匹配的任务以便调试
        for i, task in enumerate(matched_tasks[:5], 1):  # 只打印前5个
            print(f"  {i}. {task['task_name']}: {task['goal'][:80]}{'...' if len(task['goal']) > 80 else ''}")
        
        return matched_tasks
    
    def format_tasks_as_fewshot(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        将AndroidWorld任务格式化为few-shot示例格式
        
        Args:
            tasks: AndroidWorld任务列表
            
        Returns:
            格式化后的few-shot示例列表
        """
        fewshot_examples = []
        
        for task in tasks:
            example = {
                'instruction': task['goal'],
                'source': 'AndroidWorld',
                'task_name': task['task_name'],
                'main_app': task['main_app']
            }
            fewshot_examples.append(example)
        
        return fewshot_examples
    
    def get_app_fewshot_examples(self, app_package: str, app_name: str = None,
                                max_examples: int = None) -> List[Dict[str, str]]:
        """
        获取指定应用的few-shot示例
        
        Args:
            app_package: 应用包名
            app_name: 应用名称（可选）
            max_examples: 最大示例数量（None表示不限制）
            
        Returns:
            few-shot示例列表
        """
        # 查找匹配的任务
        matched_tasks = self.get_app_tasks(app_package, app_name)
        
        # 限制数量（如果指定了max_examples）
        if max_examples and len(matched_tasks) > max_examples:
            matched_tasks = matched_tasks[:max_examples]
        
        # 格式化为few-shot示例
        fewshot_examples = self.format_tasks_as_fewshot(matched_tasks)
        
        return fewshot_examples
    
    def get_all_apps(self) -> List[str]:
        """
        获取Excel中所有的应用名称
        
        Returns:
            应用名称列表
        """
        if self.tasks_df is None:
            return []
        
        apps = set()
        
        # 从main_app列获取
        for main_app in self.tasks_df['main_app'].dropna():
            apps.add(str(main_app))
        
        # 从apps列获取（可能包含多个app）
        for apps_str in self.tasks_df['apps'].dropna():
            for app in str(apps_str).split(','):
                apps.add(app.strip())
        
        return sorted(list(apps))
    
    def save_debug_info(self, debug_dir: Path):
        """
        保存调试信息
        
        Args:
            debug_dir: 调试输出目录
        """
        if self.tasks_df is None:
            return
        
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整的任务数据
        tasks_json_path = debug_dir / "android_world_tasks.json"
        tasks_data = self.tasks_df.to_dict('records')
        
        with open(tasks_json_path, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, indent=2, ensure_ascii=False)
        
        # 保存应用映射表
        mapping_path = debug_dir / "app_name_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.app_name_mapping, f, indent=2, ensure_ascii=False)
        
        # 保存所有应用列表
        apps_path = debug_dir / "all_apps.json"
        all_apps = self.get_all_apps()
        with open(apps_path, 'w', encoding='utf-8') as f:
            json.dump(all_apps, f, indent=2, ensure_ascii=False)
        
        print(f"AndroidWorld debug info saved to: {debug_dir}")


def test_android_world_loader():
    """测试AndroidWorld任务加载器"""
    print("Testing AndroidWorld Task Loader...")
    
    loader = AndroidWorldTaskLoader()
    
    # 测试几个应用
    test_apps = [
        ("com.android.camera2", "Camera"),
        ("com.android.contacts", "Contacts"),
        ("com.simplemobiletools.calendar.pro", "Simple Calendar"),
        ("net.gsantner.markor", "Markor"),
        ("unknown.app", "Unknown App")
    ]
    
    for app_package, app_name in test_apps:
        print(f"\n{'='*60}")
        print(f"Testing app: {app_package} ({app_name})")
        print(f"{'='*60}")
        
        fewshot_examples = loader.get_app_fewshot_examples(app_package, app_name, max_examples=3)
        
        print(f"Found {len(fewshot_examples)} few-shot examples:")
        for i, example in enumerate(fewshot_examples, 1):
            print(f"{i}. {example['instruction']}")
    
    # 保存调试信息
    debug_dir = Path("debug_android_world")
    loader.save_debug_info(debug_dir)


if __name__ == "__main__":
    test_android_world_loader()
