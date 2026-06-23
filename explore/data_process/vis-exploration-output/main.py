#!/usr/bin/env python3
"""
Memory-Optimized Exploration Output Visualization Tool

This tool parses and visualizes the exploration results from exploration_output folder
with memory optimization for handling large datasets.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image
import pandas as pd
import gc
import psutil

from parser import MemoryOptimizedParser, AppData


def check_parsed_data_exists(output_dir: str, specific_app: Optional[str] = None) -> bool:
    """Check if parsed data already exists in the output directory"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return False
    
    if specific_app:
        # Check for specific app
        app_dir = output_path / specific_app
        return (app_dir.exists() and 
                (app_dir / "app_info.json").exists() and
                (app_dir / "statistics.json").exists() and
                (app_dir / "trajectories_summary.json").exists())
    else:
        # Check if any app data exists
        app_dirs = [d for d in output_path.iterdir() if d.is_dir()]
        return any(
            (app_dir / "app_info.json").exists() and
            (app_dir / "statistics.json").exists() and
            (app_dir / "trajectories_summary.json").exists()
            for app_dir in app_dirs
        )


def load_existing_parsed_data(output_dir: str, specific_app: Optional[str] = None) -> Dict[str, AppData]:
    """Load existing parsed data from JSON files"""
    output_path = Path(output_dir)
    app_data = {}
    
    if specific_app:
        app_dirs = [output_path / specific_app] if (output_path / specific_app).exists() else []
    else:
        app_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    
    for app_dir in app_dirs:
        try:
            app_info_file = app_dir / "app_info.json"
            stats_file = app_dir / "statistics.json"
            trajectories_summary_file = app_dir / "trajectories_summary.json"
            
            if app_info_file.exists() and stats_file.exists():
                # Load app info
                with open(app_info_file, 'r', encoding='utf-8') as f:
                    app_info = json.load(f)
                
                # Load statistics
                with open(stats_file, 'r', encoding='utf-8') as f:
                    statistics = json.load(f)
                
                # Load trajectories summary if available
                trajectories = []
                depth_structure = {}
                
                if trajectories_summary_file.exists():
                    try:
                        with open(trajectories_summary_file, 'r', encoding='utf-8') as f:
                            trajectories_data = json.load(f)
                            
                        # Reconstruct trajectory objects from JSON
                        from parser import Trajectory, TrajectoryStep
                        
                        for traj_data in trajectories_data:
                            # Create empty steps list - summary format doesn't include full step details
                            # We'll create placeholder steps based on steps_count
                            steps = []
                            steps_count = traj_data.get('steps_count', 0)
                            
                            # Load full trajectory data with actual step details
                            trajectories_dir = app_dir / "trajectories"
                            trajectory_file = trajectories_dir / f"{traj_data.get('trajectory_id')}.json"
                            
                            if trajectory_file.exists():
                                # Load detailed trajectory data
                                try:
                                    with open(trajectory_file, 'r', encoding='utf-8') as f:
                                        detailed_traj_data = json.load(f)
                                    
                                    # Create steps from detailed data
                                    for step_data in detailed_traj_data.get('steps', []):
                                        step = TrajectoryStep(
                                            step_index=step_data.get('step_index', 0),
                                            goal=step_data.get('goal', ''),
                                            summary=step_data.get('summary', ''),
                                            activity=step_data.get('activity', []),
                                            ui_elements=step_data.get('ui_elements', []),  # Load UI elements from parsed data
                                            converted_action=None,
                                            logical_screen_size=tuple(step_data.get('logical_screen_size', [1080, 1920])),
                                            timestamp=step_data.get('timestamp'),
                                            metadata=step_data.get('metadata', {})
                                        )
                                        steps.append(step)
                                except Exception as e:
                                    print(f"    ⚠️  Failed to load detailed trajectory data for {traj_data.get('trajectory_id')}: {e}")
                                    # Fallback to placeholder steps with actual count
                                    for i in range(steps_count):
                                        step = TrajectoryStep(
                                            step_index=i,
                                            goal=traj_data.get('goal'),
                                            summary=f"Step {i+1} - detailed data not available",
                                            activity=[],
                                            ui_elements=[],
                                            converted_action=None,
                                            logical_screen_size=(1080, 1920),
                                            timestamp=None,
                                            metadata={}
                                        )
                                        steps.append(step)
                            else:
                                # Fallback to placeholder steps with actual count
                                for i in range(steps_count):
                                    step = TrajectoryStep(
                                        step_index=i,
                                        goal=traj_data.get('goal'),
                                        summary=f"Step {i+1} - detailed data not available",
                                        activity=[],
                                        ui_elements=[],
                                        converted_action=None,
                                        logical_screen_size=(1080, 1920),
                                        timestamp=None,
                                        metadata={}
                                    )
                                    steps.append(step)
                            
                            # Parse start_time if available
                            start_time = None
                            if 'start_time' in traj_data and traj_data['start_time']:
                                try:
                                    from datetime import datetime
                                    start_time = datetime.fromisoformat(traj_data['start_time'].replace('Z', '+00:00'))
                                except:
                                    start_time = None
                            
                            # Create trajectory
                            trajectory = Trajectory(
                                trajectory_id=traj_data.get('trajectory_id'),
                                package_name=app_dir.name,
                                depth=traj_data.get('depth', 0),
                                goal=traj_data.get('goal'),
                                steps=steps,
                                file_path=traj_data.get('file_path', ''),
                                app_info=app_info,
                                start_time=start_time,
                                end_time=None,
                                success=traj_data.get('success', False)
                            )
                            
                            trajectories.append(trajectory)
                            
                            # Build depth structure
                            depth = trajectory.depth
                            if depth not in depth_structure:
                                depth_structure[depth] = []
                            depth_structure[depth].append(trajectory)
                        
                        print(f"  📋 Loaded {len(trajectories)} trajectory summaries for {app_dir.name}")
                        
                    except Exception as e:
                        print(f"  ⚠️  Failed to load trajectory summaries for {app_dir.name}: {e}")
                        # Continue with empty trajectories but still load basic stats
                
                # Create AppData object
                app_data_obj = AppData(
                    package_name=app_dir.name,
                    app_info=app_info,
                    trajectories=trajectories,
                    statistics=statistics,
                    depth_structure=depth_structure
                )
                
                app_data[app_dir.name] = app_data_obj
                print(f"  📱 Loaded {app_dir.name}: {statistics.get('total_trajectories', 0)} trajectories, {len(depth_structure)} depths")
                
        except Exception as e:
            print(f"⚠️  Failed to load data for {app_dir.name}: {e}")
            continue
    
    return app_data


@dataclass
class Config:
    """Configuration for exploration output visualization"""
    exploration_output_dir: str = "./exploration_output"
    output_dir: str = "./exploration_output_uncompress"
    include_screenshots: bool = True
    generate_html_report: bool = True
    generate_statistics: bool = True
    verbose: bool = False
    max_trajectories: int = 0
    sample_only: bool = False
    memory_limit: float = 0
    batch_size: int = 5  # Process 5 trajectories at a time


def get_memory_usage() -> float:
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024


def estimate_app_memory_requirements(app_dir: Path) -> float:
    """Estimate memory requirements for processing an app"""
    pkl_files = list(app_dir.rglob("*.pkl.zst"))
    
    if not pkl_files:
        return 0.0
    
    # Sample a few files to estimate average size
    sample_size = min(3, len(pkl_files))
    total_size = 0
    
    for pkl_file in pkl_files[:sample_size]:
        total_size += pkl_file.stat().st_size
    
    avg_size_mb = (total_size / sample_size) / (1024 * 1024)
    
    # Estimate decompressed size (typically 3-5x compressed size)
    # Plus overhead for processing (2x)
    estimated_gb = (avg_size_mb * len(pkl_files) * 4 * 2) / 1024
    
    return estimated_gb


def main():
    parser = argparse.ArgumentParser(
        description="Parse and visualize exploration output data with memory optimization"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./exploration_output",
        help="Input directory containing exploration results (default: ./exploration_output)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="./exploration_output_uncompress",
        help="Output directory for processed data (default: ./exploration_output_uncompress)"
    )
    parser.add_argument(
        "--app_package",
        type=str,
        help="Specific app package to process (default: process all apps)"
    )
    parser.add_argument(
        "--no_screenshots",
        action="store_true",
        help="Skip screenshot extraction (faster processing, less memory usage)"
    )
    parser.add_argument(
        "--no_html",
        action="store_true", 
        help="Skip HTML report generation"
    )
    parser.add_argument(
        "--no_stats",
        action="store_true",
        help="Skip statistics generation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=0,
        help="Maximum number of trajectories to process per app (0 = no limit)"
    )
    parser.add_argument(
        "--sample_only",
        action="store_true",
        help="Process only a sample of data for quick preview"
    )
    parser.add_argument(
        "--memory_limit",
        type=float,
        default=16.0,
        help="Memory limit in GB (default: 16GB, 0 = no limit)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Number of trajectories to process in each batch (default: 3, lower = less memory usage)"
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Lightweight mode: process only 10 trajectories per app, no screenshots"
    )
    parser.add_argument(
        "--aggressive_cleanup",
        action="store_true",
        help="Enable aggressive memory cleanup (slower but uses less memory)"
    )
    parser.add_argument(
        "--skip_large_apps",
        action="store_true",
        help="Skip apps that are estimated to exceed memory limits"
    )
    parser.add_argument(
        "--force_reparse",
        action="store_true",
        help="Force re-parsing even if parsed data already exists"
    )
    parser.add_argument(
        "--html_only",
        action="store_true",
        help="Only generate HTML reports from existing parsed data (skip parsing entirely)"
    )
    
    args = parser.parse_args()
    
    # Handle lightweight mode
    if args.lightweight:
        max_trajectories = 10
        include_screenshots = False
        batch_size = 2
        print("🪶 Lightweight mode enabled: processing only 10 trajectories per app, no screenshots")
    else:
        max_trajectories = args.max_trajectories
        include_screenshots = not args.no_screenshots
        batch_size = args.batch_size
    
    # Adjust batch size based on memory limit
    if args.memory_limit > 0:
        if args.memory_limit <= 8:
            batch_size = min(batch_size, 2)
            print(f"⚠️  Low memory limit ({args.memory_limit}GB), reducing batch size to {batch_size}")
        elif args.memory_limit <= 16:
            batch_size = min(batch_size, 3)
    
    # Create configuration
    config = Config(
        exploration_output_dir=args.input_dir,
        output_dir=args.output_dir,
        include_screenshots=include_screenshots,
        generate_html_report=not args.no_html,
        generate_statistics=not args.no_stats,
        verbose=args.verbose,
        max_trajectories=max_trajectories,
        sample_only=args.sample_only,
        memory_limit=args.memory_limit,
        batch_size=batch_size
    )
    
    print("=" * 80)
    print("🔍 Memory-Optimized Exploration Output Visualization Tool")
    print("=" * 80)
    print(f"Input directory: {config.exploration_output_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Include screenshots: {config.include_screenshots}")
    print(f"Memory limit: {config.memory_limit}GB" + (" (no limit)" if config.memory_limit <= 0 else ""))
    print(f"Batch size: {config.batch_size} trajectories")
    print(f"Current memory usage: {get_memory_usage():.2f}GB")
    print("=" * 80)
    
    # Validate input directory
    if not os.path.exists(config.exploration_output_dir):
        print(f"❌ Error: Input directory '{config.exploration_output_dir}' does not exist!")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Pre-flight memory check
    if args.skip_large_apps and config.memory_limit > 0:
        print("\n🔍 Analyzing app memory requirements...")
        exploration_dir = Path(config.exploration_output_dir)
        app_dirs = [d for d in exploration_dir.iterdir() if d.is_dir()]
        if args.app_package:
            app_dirs = [d for d in app_dirs if d.name == args.app_package]
        
        for app_dir in app_dirs:
            estimated_memory = estimate_app_memory_requirements(app_dir)
            print(f"  📱 {app_dir.name}: ~{estimated_memory:.1f}GB estimated")
            
            if estimated_memory > config.memory_limit * 0.8:  # 80% of limit
                print(f"    ⚠️  High memory usage estimated, consider using --max_trajectories or --no_screenshots")
    
    try:
        # Check if parsed data already exists
        parsed_data_exists = check_parsed_data_exists(config.output_dir, args.app_package)
        
        if parsed_data_exists and not args.force_reparse:
            print("\n🔍 Found existing parsed data!")
            print("📥 Loading from existing JSON files instead of re-parsing...")
            print("   (Use --force_reparse to re-parse from scratch)")
            
            app_data = load_existing_parsed_data(config.output_dir, args.app_package)
            
            if not app_data:
                print("❌ Failed to load existing data! Will re-parse...")
                # Fall back to normal parsing
                parsed_data_exists = False
        
        if not parsed_data_exists or args.force_reparse:
            # Initialize parser with context manager for proper cleanup
            with MemoryOptimizedParser(config) as data_parser:
                # Parse exploration data
                print(f"\n📊 Parsing exploration data (batch size: {config.batch_size})...")
                
                if args.aggressive_cleanup:
                    print("🧹 Aggressive cleanup mode enabled")
                    # Force garbage collection before starting
                    gc.collect()
                
                app_data = data_parser.parse_all_apps(
                    specific_app=args.app_package
                )
                
                if not app_data:
                    print("❌ No valid exploration data found!")
                    sys.exit(1)
                
                print(f"✅ Successfully parsed data for {len(app_data)} app(s)")
                print(f"💾 Final memory usage: {get_memory_usage():.2f}GB")
        else:
            print(f"✅ Successfully loaded data for {len(app_data)} app(s)")
            
            # Generate statistics (lightweight operation)
            if config.generate_statistics:
                print("\n📈 Generating statistics...")
                for app_name, data in app_data.items():
                    stats = data.statistics
                    print(f"  📱 {app_name}: {stats['total_trajectories']} trajectories, "
                          f"{stats['total_steps']} steps, "
                          f"{stats['successful_trajectories']} successful")
                print("✅ Statistics generated")
            
            # Generate HTML reports
            if config.generate_html_report:
                print("\n🎨 Generating HTML reports...")
                try:
                    from visualizer import ExplorationVisualizer
                    visualizer = ExplorationVisualizer(config)
                    visualizer.generate_html_report(app_data)
                    print("✅ HTML reports generated successfully")
                    print(f"📄 Open {config.output_dir}/overview.html to view the main report")
                except Exception as e:
                    print(f"⚠️  HTML generation failed: {e}")
                    if config.verbose:
                        import traceback
                        traceback.print_exc()
                    print("   (JSON data is still available for analysis)")
        
        print(f"\n🎉 Processing completed! Results saved to: {config.output_dir}")
        print("\n📝 Output files for each app:")
        print("   - app_info.json: Application information")
        print("   - statistics.json: Processing statistics")
        print("   - trajectories_summary.json: Summary of all trajectories")
        print("   - trajectories/: Individual trajectory files")
        if config.include_screenshots:
            print("   - screenshots/: Extracted screenshots")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        print("🧹 Cleaning up temporary files...")
        sys.exit(1)
    except MemoryError:
        print("\n\n❌ Out of memory! Try these options:")
        print("   --no_screenshots: Skip screenshot extraction")
        print("   --batch_size 1: Process one trajectory at a time")
        print("   --max_trajectories 20: Limit trajectories per app")
        print("   --lightweight: Use lightweight processing mode")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
