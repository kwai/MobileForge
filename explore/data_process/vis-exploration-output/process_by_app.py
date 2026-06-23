#!/usr/bin/env python3
"""
App-by-App Processing Script

This script processes exploration data one app at a time to avoid memory issues.
It can be used when the full dataset is too large to process at once.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def get_app_directories(exploration_output_dir):
    """Get list of app directories in exploration output"""
    app_dirs = []
    if os.path.exists(exploration_output_dir):
        for item in os.listdir(exploration_output_dir):
            item_path = os.path.join(exploration_output_dir, item)
            if os.path.isdir(item_path):
                app_dirs.append(item)
    return sorted(app_dirs)

def process_single_app(app_package, input_dir, output_dir, args):
    """Process a single app"""
    print(f"\n{'='*60}")
    print(f"🔍 Processing: {app_package}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        sys.executable, 
        "main.py",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--app_package", app_package
    ]
    
    # Add optional arguments
    if args.no_screenshots:
        cmd.append("--no_screenshots")
    if args.verbose:
        cmd.append("--verbose")
    if args.memory_limit > 0:
        cmd.extend(["--memory_limit", str(args.memory_limit)])
    if args.max_trajectories > 0:
        cmd.extend(["--max_trajectories", str(args.max_trajectories)])
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Successfully processed {app_package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to process {app_package}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error processing {app_package}: {e}")
        return False

def merge_results(output_dir, app_packages):
    """Merge individual app results into a combined overview"""
    print(f"\n{'='*60}")
    print("🔄 Merging results...")
    print(f"{'='*60}")
    
    # Import the visualizer to generate combined overview
    try:
        from visualizer import ExplorationVisualizer
        from statistics import StatisticsGenerator
        from parser import ExplorationDataParser
        from main import Config
        
        # Create a config for merging
        config = Config(
            exploration_output_dir="./temp",
            output_dir=output_dir,
            include_screenshots=False,
            generate_html_report=True,
            generate_statistics=True,
            verbose=True
        )
        
        # Load data from individual app results
        app_data = {}
        for app_package in app_packages:
            app_output_dir = os.path.join(output_dir, app_package)
            if os.path.exists(app_output_dir):
                # Try to load the processed data
                try:
                    # This is a simplified merge - in a real implementation,
                    # you'd want to load the actual processed data
                    print(f"📦 Found results for {app_package}")
                except Exception as e:
                    print(f"⚠️  Could not load data for {app_package}: {e}")
        
        # Generate combined overview (simplified)
        print("📊 Generating combined overview...")
        
        # Create a simple combined overview HTML
        overview_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Combined Exploration Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .app-link {{ display: block; margin: 10px 0; padding: 10px; background: #f5f5f5; text-decoration: none; color: #333; }}
        .app-link:hover {{ background: #e0e0e0; }}
    </style>
</head>
<body>
    <h1>🔍 Combined Exploration Results</h1>
    <p>Processed {len(app_packages)} applications individually:</p>
    <div class="app-links">
"""
        
        for app_package in app_packages:
            app_report_path = f"{app_package}/report.html"
            if os.path.exists(os.path.join(output_dir, app_report_path)):
                overview_html += f'        <a href="{app_report_path}" class="app-link">📱 {app_package}</a>\n'
        
        overview_html += """
    </div>
    <p><em>Each link opens the detailed report for that application.</em></p>
</body>
</html>
"""
        
        # Save combined overview
        overview_path = os.path.join(output_dir, "overview.html")
        with open(overview_path, 'w', encoding='utf-8') as f:
            f.write(overview_html)
        
        print(f"✅ Combined overview saved to: {overview_path}")
        
    except Exception as e:
        print(f"⚠️  Could not generate combined overview: {e}")
        print("Individual app results are still available in their respective folders.")

def main():
    parser = argparse.ArgumentParser(
        description="Process exploration data one app at a time to avoid memory issues"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./exploration_output",
        help="Input directory containing exploration results"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="./exploration_output_by_app",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--no_screenshots",
        action="store_true",
        help="Skip screenshot extraction (faster processing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--memory_limit",
        type=float,
        default=8,
        help="Memory limit in GB per app (default: 8)"
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=0,
        help="Maximum number of trajectories to process per app (0 = no limit)"
    )
    parser.add_argument(
        "--skip_apps",
        type=str,
        nargs="*",
        default=[],
        help="List of app packages to skip"
    )
    parser.add_argument(
        "--only_apps",
        type=str,
        nargs="*",
        default=[],
        help="Process only these app packages"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 App-by-App Exploration Data Processor")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Memory limit per app: {args.memory_limit}GB")
    print("=" * 80)
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory '{args.input_dir}' does not exist!")
        sys.exit(1)
    
    # Get list of apps to process
    all_apps = get_app_directories(args.input_dir)
    
    if not all_apps:
        print(f"❌ No app directories found in {args.input_dir}")
        sys.exit(1)
    
    # Filter apps based on arguments
    apps_to_process = []
    if args.only_apps:
        apps_to_process = [app for app in all_apps if app in args.only_apps]
    else:
        apps_to_process = [app for app in all_apps if app not in args.skip_apps]
    
    if not apps_to_process:
        print("❌ No apps to process after filtering!")
        sys.exit(1)
    
    print(f"📱 Found {len(apps_to_process)} app(s) to process:")
    for app in apps_to_process:
        print(f"   - {app}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each app individually
    successful_apps = []
    failed_apps = []
    
    for i, app_package in enumerate(apps_to_process):
        print(f"\n🔄 Progress: {i+1}/{len(apps_to_process)}")
        
        if process_single_app(app_package, args.input_dir, args.output_dir, args):
            successful_apps.append(app_package)
        else:
            failed_apps.append(app_package)
        
        # Force cleanup between apps
        import gc
        gc.collect()
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 Processing Summary")
    print(f"{'='*80}")
    print(f"✅ Successful: {len(successful_apps)}/{len(apps_to_process)}")
    print(f"❌ Failed: {len(failed_apps)}/{len(apps_to_process)}")
    
    if successful_apps:
        print("\n✅ Successfully processed:")
        for app in successful_apps:
            print(f"   - {app}")
    
    if failed_apps:
        print("\n❌ Failed to process:")
        for app in failed_apps:
            print(f"   - {app}")
    
    # Merge results if we have successful apps
    if successful_apps:
        merge_results(args.output_dir, successful_apps)
        print(f"\n🎉 Processing completed! Results saved to: {args.output_dir}")
    else:
        print("\n❌ No apps were processed successfully!")

if __name__ == "__main__":
    main()
