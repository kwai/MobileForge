"""
Memory-optimized data parser for exploration output files
"""

import os
import sys
import json
import time
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
import numpy as np
from PIL import Image
import hashlib
import tempfile
import shutil

# Add project root to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import required compression libraries
import pickle
import zstd

# Import progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Progress bars will not be shown.")


def load_object_from_disk(file_path: str) -> object:
    """从本地文件读取Zstandard压缩的pickle数据并反序列化为对象"""
    with open(file_path, "rb") as file:
        compressed_data = file.read()
    pickled_data = zstd.decompress(compressed_data)
    return pickle.loads(pickled_data)


@dataclass
class TrajectoryStep:
    """Represents a single step in a trajectory"""
    step_index: int
    goal: str
    summary: str
    activity: List[str]
    ui_elements: List[Dict]
    raw_screenshot: Optional[np.ndarray] = None
    before_screenshot: Optional[np.ndarray] = None
    after_screenshot: Optional[np.ndarray] = None
    before_screenshot_with_som: Optional[np.ndarray] = None
    after_screenshot_with_som: Optional[np.ndarray] = None
    converted_action: Any = None
    logical_screen_size: Tuple[int, int] = (1080, 1920)
    timestamp: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass 
class Trajectory:
    """Represents a complete trajectory"""
    trajectory_id: str
    package_name: str
    depth: int
    goal: str
    steps: List[TrajectoryStep]
    file_path: str
    app_info: Dict
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class AppData:
    """Represents all data for a single app"""
    package_name: str
    app_info: Dict
    trajectories: List[Trajectory] = field(default_factory=list)
    depth_structure: Dict[int, List[Trajectory]] = field(default_factory=dict)
    statistics: Dict = field(default_factory=dict)


class MemoryOptimizedParser:
    """Memory-optimized parser for exploration output data"""
    
    def __init__(self, config):
        self.config = config
        self.verbose = config.verbose
        self.memory_limit_gb = getattr(config, 'memory_limit', 0)
        self.batch_size = getattr(config, 'batch_size', 5)  # Process 5 trajectories at a time
        self.temp_dir = None
        
    def __enter__(self):
        """Setup temporary directory for batch processing"""
        self.temp_dir = tempfile.mkdtemp(prefix="trajectory_batch_")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def log(self, message: str, level: str = "info"):
        """Log a message if verbose mode is enabled"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level.upper()}: {message}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit"""
        if self.memory_limit_gb <= 0:
            return False
        
        current_memory = self.get_memory_usage()
        if current_memory > self.memory_limit_gb:
            self.log(f"⚠️  Memory usage ({current_memory:.2f}GB) exceeds limit ({self.memory_limit_gb}GB)", "warning")
            return True
        return False
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        gc.collect()
        self.log(f"🧹 Garbage collection completed. Current memory: {self.get_memory_usage():.2f}GB")
    
    def parse_all_apps(self, specific_app: Optional[str] = None) -> Dict[str, AppData]:
        """Parse data for all apps with memory optimization"""
        app_data = {}
        
        # Find all app directories
        exploration_dir = Path(self.config.exploration_output_dir)
        if not exploration_dir.exists():
            raise FileNotFoundError(f"Exploration directory not found: {exploration_dir}")
        
        app_dirs = [d for d in exploration_dir.iterdir() if d.is_dir()]
        if specific_app:
            app_dirs = [d for d in app_dirs if d.name == specific_app]
        
        if not app_dirs:
            self.log("No app directories found")
            return app_data
        
        print(f"📱 Found {len(app_dirs)} app(s) to process")
        
        # Use progress bar for app processing
        if TQDM_AVAILABLE:
            app_iterator = tqdm(app_dirs, desc="Processing Apps", unit="app")
        else:
            app_iterator = app_dirs
            
        for i, app_dir in enumerate(app_iterator):
            try:
                if TQDM_AVAILABLE:
                    app_iterator.set_description(f"Processing {app_dir.name}")
                else:
                    print(f"📱 Processing app {i+1}/{len(app_dirs)}: {app_dir.name}")
                
                # Force garbage collection before processing each app
                if i > 0:
                    self.force_garbage_collection()
                
                app_data[app_dir.name] = self.parse_app_streaming(app_dir)
                
                if not TQDM_AVAILABLE:
                    print(f"✅ Successfully processed app: {app_dir.name}")
                    
            except Exception as e:
                error_msg = f"❌ Error parsing app {app_dir.name}: {e}"
                if TQDM_AVAILABLE:
                    tqdm.write(error_msg)
                else:
                    print(error_msg)
                    
                import traceback
                if self.verbose:
                    traceback.print_exc()
                continue
        
        return app_data
    
    def parse_app_streaming(self, app_dir: Path) -> AppData:
        """Parse data for a single app using streaming approach"""
        package_name = app_dir.name
        
        # Load app info
        app_info_path = app_dir / "app_info.json"
        if not app_info_path.exists():
            raise FileNotFoundError(f"app_info.json not found in {app_dir}")
        
        with open(app_info_path, 'r', encoding='utf-8') as f:
            app_info = json.load(f)
        
        # Create app data object
        app_data = AppData(
            package_name=package_name,
            app_info=app_info
        )
        
        # Find all trajectory files
        pkl_files = list(app_dir.rglob("*.pkl.zst"))
        self.log(f"Found {len(pkl_files)} trajectory files for {package_name}")
        
        # Apply limits if configured
        max_trajectories = getattr(self.config, 'max_trajectories', 0)
        sample_only = getattr(self.config, 'sample_only', False)
        
        if sample_only:
            max_trajectories = min(5, len(pkl_files))
            self.log(f"Sample mode: processing only {max_trajectories} trajectories")
        elif max_trajectories > 0 and len(pkl_files) > max_trajectories:
            pkl_files = pkl_files[:max_trajectories]
            self.log(f"Limited to {max_trajectories} trajectories")
        
        # Process trajectories in batches
        processed_count = 0
        total_files = len(pkl_files)
        
        # Use batch processing to reduce memory usage
        for batch_start in range(0, total_files, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_files)
            batch_files = pkl_files[batch_start:batch_end]
            
            self.log(f"Processing batch {batch_start//self.batch_size + 1}/{(total_files + self.batch_size - 1)//self.batch_size} "
                    f"({len(batch_files)} files)")
            
            # Check memory before processing batch
            if self.check_memory_limit():
                self.force_garbage_collection()
                if self.memory_limit_gb > 0 and self.get_memory_usage() > (self.memory_limit_gb * 1.2):
                    self.log(f"⚠️  Memory limit significantly exceeded, stopping at {processed_count}/{total_files} trajectories", "warning")
                    break
            
            # Process current batch
            batch_trajectories = []
            for pkl_file in batch_files:
                try:
                    trajectory = self.parse_trajectory_lightweight(pkl_file, app_info)
                    batch_trajectories.append(trajectory)
                    processed_count += 1
                    
                except Exception as e:
                    self.log(f"❌ Error parsing trajectory {pkl_file.name}: {e}", "warning")
                    continue
            
            # Save batch immediately and clear from memory
            self.save_trajectory_batch(batch_trajectories, app_data)
            
            # Add to app_data for statistics (without heavy data)
            for trajectory in batch_trajectories:
                # Create lightweight version for in-memory storage
                lightweight_trajectory = self.create_lightweight_trajectory(trajectory)
                app_data.trajectories.append(lightweight_trajectory)
                
                # Organize by depth
                depth = trajectory.depth
                if depth not in app_data.depth_structure:
                    app_data.depth_structure[depth] = []
                app_data.depth_structure[depth].append(lightweight_trajectory)
            
            # Clear batch from memory
            del batch_trajectories
            self.force_garbage_collection()
            
            self.log(f"Batch processed. Total: {processed_count}/{total_files} trajectories")
        
        self.log(f"Successfully processed {processed_count}/{total_files} trajectories")
        
        # Generate statistics
        app_data.statistics = self.generate_app_statistics(app_data)
        
        # Save final app data
        self.save_app_data_final(app_data)
        
        return app_data
    
    def parse_trajectory_lightweight(self, pkl_file: Path, app_info: Dict) -> Trajectory:
        """Parse a single trajectory with memory optimization"""
        trajectory_id = pkl_file.stem.replace('.pkl', '')
        depth = self.extract_depth_from_path(pkl_file)
        
        # Load trajectory data
        trajectory_data = load_object_from_disk(str(pkl_file))
        
        if not trajectory_data or not isinstance(trajectory_data, list):
            raise ValueError(f"Invalid trajectory data in {pkl_file}")
        
        goal = trajectory_data[0].get('goal', 'Unknown goal') if trajectory_data else 'Unknown goal'
        
        # Parse steps with memory optimization
        steps = []
        for i, step_data in enumerate(trajectory_data):
            step = self.parse_trajectory_step_lightweight(i, step_data)
            steps.append(step)
        
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            package_name=app_info.get('app_pkg', ''),
            depth=depth,
            goal=goal,
            steps=steps,
            file_path=str(pkl_file),
            app_info=app_info
        )
        
        self.extract_trajectory_timestamps(trajectory)
        trajectory.success = self.analyze_trajectory_success(trajectory)
        
        return trajectory
    
    def parse_trajectory_step_lightweight(self, step_index: int, step_data: Dict) -> TrajectoryStep:
        """Parse a trajectory step with selective screenshot loading"""
        step = TrajectoryStep(
            step_index=step_index,
            goal=step_data.get('goal', ''),
            summary=step_data.get('summary', ''),
            activity=step_data.get('activity', []),
            ui_elements=step_data.get('ui_elements', []),
            converted_action=step_data.get('converted_action'),
            logical_screen_size=step_data.get('logical_screen_size', (1080, 1920)),
            timestamp=step_data.get('timestamp')
        )
        
        # Load all screenshot types if requested and within limits
        if self.config.include_screenshots and not self.check_memory_limit():
            step.raw_screenshot = self.extract_screenshot(step_data, 'raw_screenshot')
            step.before_screenshot = self.extract_screenshot(step_data, 'before_screenshot')
            step.after_screenshot = self.extract_screenshot(step_data, 'after_screenshot')
            step.before_screenshot_with_som = self.extract_screenshot(step_data, 'before_screenshot_with_som')
            step.after_screenshot_with_som = self.extract_screenshot(step_data, 'after_screenshot_with_som')
        
        # Extract metadata (lightweight)
        excluded_keys = ['goal', 'summary', 'activity', 'ui_elements', 'converted_action', 
                        'logical_screen_size', 'timestamp', 'raw_screenshot', 
                        'before_screenshot', 'after_screenshot', 'before_screenshot_with_som', 
                        'after_screenshot_with_som']
        
        step.metadata = {}
        for k, v in step_data.items():
            if k not in excluded_keys:
                try:
                    json.dumps(v)
                    step.metadata[k] = v
                except (TypeError, ValueError):
                    step.metadata[k] = str(v)[:100]  # Limit string length
        
        return step
    
    def create_lightweight_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """Create a lightweight version of trajectory for in-memory storage"""
        lightweight_steps = []
        for step in trajectory.steps:
            # Create step without screenshots
            lightweight_step = TrajectoryStep(
                step_index=step.step_index,
                goal=step.goal,
                summary=step.summary,
                activity=step.activity,
                ui_elements=step.ui_elements[:10],  # Limit UI elements
                converted_action=step.converted_action,
                logical_screen_size=step.logical_screen_size,
                timestamp=step.timestamp,
                metadata=step.metadata
            )
            lightweight_steps.append(lightweight_step)
        
        return Trajectory(
            trajectory_id=trajectory.trajectory_id,
            package_name=trajectory.package_name,
            depth=trajectory.depth,
            goal=trajectory.goal,
            steps=lightweight_steps,
            file_path=trajectory.file_path,
            app_info=trajectory.app_info,
            start_time=trajectory.start_time,
            end_time=trajectory.end_time,
            success=trajectory.success,
            metadata=trajectory.metadata
        )
    
    def save_trajectory_batch(self, trajectories: List[Trajectory], app_data: AppData):
        """Save a batch of trajectories with screenshots immediately"""
        if not self.config.include_screenshots:
            return
            
        output_dir = Path(self.config.output_dir) / app_data.package_name
        screenshots_dir = output_dir / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        for trajectory in trajectories:
            self.save_trajectory_screenshots(trajectory, screenshots_dir)
    
    def save_trajectory_screenshots(self, trajectory: Trajectory, screenshots_dir: Path):
        """Save screenshots for a single trajectory"""
        trajectory_screenshots_dir = screenshots_dir / trajectory.trajectory_id
        trajectory_screenshots_dir.mkdir(exist_ok=True)
        
        for step in trajectory.steps:
            if not any([step.before_screenshot is not None,
                        step.raw_screenshot is not None, 
                       step.after_screenshot is not None, 
                       step.after_screenshot_with_som is not None]):
                continue
                
            step_dir = trajectory_screenshots_dir / f"step_{step.step_index:02d}"
            step_dir.mkdir(exist_ok=True)
            
            # Save most important screenshot only to save space
            # Prioritize raw_screenshot first
            screenshot_to_save = None
            screenshot_name = None
            
            if step.before_screenshot is not None:
                screenshot_to_save = step.before_screenshot
                screenshot_name = "before_screenshot"
            elif step.raw_screenshot is not None:
                screenshot_to_save = step.raw_screenshot
                screenshot_name = "raw_screenshot"
            elif step.after_screenshot_with_som is not None:
                screenshot_to_save = step.after_screenshot_with_som
                screenshot_name = "after_screenshot_with_som"
            elif step.after_screenshot is not None:
                screenshot_to_save = step.after_screenshot
                screenshot_name = "after_screenshot"
            
            if screenshot_to_save is not None:
                try:
                    if screenshot_to_save.dtype != np.uint8:
                        screenshot_to_save = (screenshot_to_save * 255).astype(np.uint8)
                    
                    img = Image.fromarray(screenshot_to_save)
                    img.save(step_dir / f"{screenshot_name}.png", optimize=True, compress_level=6)
                    
                except Exception as e:
                    self.log(f"Error saving screenshot {screenshot_name}: {e}", "warning")
    
    def extract_screenshot(self, step_data: Dict, screenshot_key: str) -> Optional[np.ndarray]:
        """Extract and validate screenshot data"""
        screenshot = step_data.get(screenshot_key)
        if screenshot is None:
            return None
        
        if not isinstance(screenshot, np.ndarray):
            return None
        
        if screenshot.size == 0:
            return None
        
        return screenshot
    
    def extract_depth_from_path(self, pkl_file: Path) -> int:
        """Extract depth information from file path"""
        parts = pkl_file.parts
        depth = 1
        
        for part in parts:
            if part.startswith('depth_'):
                try:
                    depth = int(part.replace('depth_', ''))
                    break
                except ValueError:
                    continue
        
        return depth
    
    def extract_trajectory_timestamps(self, trajectory: Trajectory):
        """Extract start and end timestamps from trajectory"""
        if not trajectory.steps:
            return
        
        try:
            filename = Path(trajectory.file_path).stem
            timestamp_part = filename.split('_')[0]
            
            if '.' in timestamp_part:
                timestamp_str = timestamp_part.split('.')[0]
            else:
                timestamp_str = timestamp_part[:12]
            
            year = 2000 + int(timestamp_str[:2])
            month = int(timestamp_str[2:4])
            day = int(timestamp_str[4:6])
            hour = int(timestamp_str[6:8])
            minute = int(timestamp_str[8:10])
            second = int(timestamp_str[10:12])
            
            trajectory.start_time = datetime(year, month, day, hour, minute, second)
            
        except (ValueError, IndexError) as e:
            self.log(f"Could not parse timestamp from filename: {e}", "warning")
    
    def analyze_trajectory_success(self, trajectory: Trajectory) -> bool:
        """Analyze if trajectory was successful"""
        if not trajectory.steps:
            return False
        return len(trajectory.steps) > 1
    
    def generate_app_statistics(self, app_data: AppData) -> Dict:
        """Generate statistics for an app"""
        stats = {
            'total_trajectories': len(app_data.trajectories),
            'trajectories_by_depth': {},
            'successful_trajectories': sum(1 for t in app_data.trajectories if t.success),
            'total_steps': sum(len(t.steps) for t in app_data.trajectories),
            'average_steps_per_trajectory': 0,
            'unique_goals': len(set(t.goal for t in app_data.trajectories)),
            'depth_distribution': {}
        }
        
        if stats['total_trajectories'] > 0:
            stats['average_steps_per_trajectory'] = stats['total_steps'] / stats['total_trajectories']
        
        for depth, trajectories in app_data.depth_structure.items():
            stats['trajectories_by_depth'][depth] = len(trajectories)
            stats['depth_distribution'][depth] = {
                'count': len(trajectories),
                'successful': sum(1 for t in trajectories if t.success),
                'average_steps': sum(len(t.steps) for t in trajectories) / len(trajectories) if trajectories else 0
            }
        
        return stats
    
    def save_app_data_final(self, app_data: AppData):
        """Save final app data without heavy screenshot data"""
        output_dir = Path(self.config.output_dir) / app_data.package_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save app info and statistics
        with open(output_dir / "app_info.json", 'w', encoding='utf-8') as f:
            json.dump(app_data.app_info, f, indent=2, ensure_ascii=False)
        
        with open(output_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(app_data.statistics, f, indent=2, ensure_ascii=False)
        
        # Save trajectory summaries
        trajectory_summaries = []
        for trajectory in app_data.trajectories:
            summary = {
                'trajectory_id': trajectory.trajectory_id,
                'depth': trajectory.depth,
                'goal': trajectory.goal,
                'steps_count': len(trajectory.steps),
                'success': trajectory.success,
                'file_path': trajectory.file_path,
                'start_time': trajectory.start_time.isoformat() if trajectory.start_time else None
            }
            trajectory_summaries.append(summary)
        
        with open(output_dir / "trajectories_summary.json", 'w', encoding='utf-8') as f:
            json.dump(trajectory_summaries, f, indent=2, ensure_ascii=False)
        
        # Save individual trajectory data (without screenshots in JSON)
        trajectories_dir = output_dir / "trajectories"
        trajectories_dir.mkdir(exist_ok=True)
        
        for trajectory in app_data.trajectories:
            trajectory_file = trajectories_dir / f"{trajectory.trajectory_id}.json"
            
            trajectory_dict = {
                'trajectory_id': trajectory.trajectory_id,
                'package_name': trajectory.package_name,
                'depth': trajectory.depth,
                'goal': trajectory.goal,
                'file_path': trajectory.file_path,
                'success': trajectory.success,
                'start_time': trajectory.start_time.isoformat() if trajectory.start_time else None,
                'steps': []
            }
            
            for step in trajectory.steps:
                step_dict = {
                    'step_index': step.step_index,
                    'goal': step.goal,
                    'summary': step.summary,
                    'activity': step.activity,
                    'ui_elements': step.ui_elements,
                    'ui_elements_count': len(step.ui_elements),
                    'logical_screen_size': step.logical_screen_size,
                    'timestamp': step.timestamp,
                    'converted_action': str(step.converted_action),
                    'action_type': str(type(step.converted_action).__name__) if step.converted_action else None,
                    'metadata': step.metadata
                }
                trajectory_dict['steps'].append(step_dict)
            
            with open(trajectory_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory_dict, f, indent=2, ensure_ascii=False)


# Wrapper class to maintain compatibility
class ExplorationDataParser(MemoryOptimizedParser):
    """Backwards compatible wrapper for the optimized parser"""
    pass
