#!/usr/bin/env python3
"""
Build page graph script - Build application page navigation graph based on existing trajectory data
Usage:
python build_page_graph.py --app_package com.android.camera2
python build_page_graph.py --batch_mode --input_dir ./exploration_output
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir  # Script is in project root
sys.path.insert(0, project_root)

from utils.page_graph_builder import PageGraphBuilder
from utils.page_graph_visualizer import PageGraphVisualizer

def build_single_app_graph(app_package: str, exploration_output_dir: str = "./exploration_output",
                          output_dir: str = "./page_graphs", visualize: bool = True,
                          max_trajectories: int = None, batch_size: int = 10,
                          use_disk_cache: bool = True, use_mllm: bool = False):
    """Build page graph for a single application"""
    print(f"\n{'='*60}")
    print(f"Building page graph for application: {app_package}")
    print(f"Memory optimization settings: batch_size={batch_size}, disk_cache={use_disk_cache}, MLLM={use_mllm}")
    print(f"{'='*60}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build page graph (using streaming builder)
    builder = PageGraphBuilder(app_package, exploration_output_dir, batch_size, use_disk_cache, use_mllm)
    graph_data = builder.build_graph_from_trajectories(max_trajectories)

    if not graph_data or len(graph_data.get("pages", {})) == 0:
        print(f"No valid page data found for application {app_package}")
        return None

    # Save graph data
    graph_file = os.path.join(output_dir, f"{app_package}_page_graph.json")
    builder.save_graph(graph_data, graph_file)

    # Generate visualization
    if visualize:
        print("\nGenerating visualization...")
        visualizer = PageGraphVisualizer()
        viz_file = os.path.join(output_dir, f"{app_package}_visualization.html")
        visualizer.visualize_graph(graph_data, viz_file)

    print(f"\nPage graph construction completed for application {app_package}!")
    print(f"- Graph data file: {graph_file}")
    if visualize:
        print(f"- Visualization file: {viz_file}")

    return graph_data

def build_batch_graphs(input_dir: str = "./exploration_output", output_dir: str = "./page_graphs",
                      visualize: bool = True, max_trajectories: int = None,
                      batch_size: int = 10, use_disk_cache: bool = True, use_mllm: bool = False):
    """Build page graphs for all applications in batch"""
    print(f"\n{'='*60}")
    print(f"Batch building page graphs - Input directory: {input_dir}")
    print(f"Memory optimization settings: batch_size={batch_size}, disk_cache={use_disk_cache}, MLLM={use_mllm}")
    print(f"{'='*60}")

    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    # Find all application packages
    app_packages = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Check if contains trajectory files
            from glob import glob
            traj_files = glob(os.path.join(item_path, "**", "*.pkl.zst"), recursive=True)
            if traj_files:
                app_packages.append(item)

    if not app_packages:
        print("No application packages with trajectory data found")
        return

    print(f"Found {len(app_packages)} application packages: {', '.join(app_packages)}")

    # Batch processing
    successful_builds = []
    failed_builds = []

    for i, app_package in enumerate(app_packages, 1):
        try:
            print(f"\nProcessing progress: {i}/{len(app_packages)}")
            graph_data = build_single_app_graph(app_package, input_dir, output_dir, visualize,
                                              max_trajectories, batch_size, use_disk_cache, use_mllm)
            if graph_data:
                successful_builds.append(app_package)
            else:
                failed_builds.append(app_package)
        except Exception as e:
            print(f"Failed to build page graph for application {app_package}: {e}")
            failed_builds.append(app_package)

    # Output summary
    print(f"\n{'='*60}")
    print(f"Batch construction completed!")
    print(f"{'='*60}")
    print(f"Successfully built: {len(successful_builds)} applications")
    for app in successful_builds:
        print(f"  ✓ {app}")

    if failed_builds:
        print(f"\nFailed builds: {len(failed_builds)} applications")
        for app in failed_builds:
            print(f"  ✗ {app}")

def generate_comparison_report(output_dir: str = "./page_graphs"):
    """Generate comparison report between applications"""
    import json
    from glob import glob

    graph_files = glob(os.path.join(output_dir, "*_page_graph.json"))
    if len(graph_files) < 2:
        print("At least 2 application page graph data files are needed to generate comparison report")
        return

    print(f"\nGenerating comparison report for {len(graph_files)} applications...")

    comparison_data = []
    for graph_file in graph_files:
        with open(graph_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        stats = graph_data.get("statistics", {})
        comparison_data.append({
            "app_package": graph_data.get("app_package", "unknown"),
            "total_pages": stats.get("total_pages", 0),
            "total_transitions": stats.get("total_transitions", 0),
            "processed_trajectories": stats.get("processed_trajectories", 0),
            "avg_success_rate": stats.get("avg_success_rate", 0),
            "key_pages": stats.get("key_pages", []),
            "memory_optimized": stats.get("memory_optimized", False),
            "batch_size": stats.get("batch_size", "N/A")
        })

    # Generate HTML report
    report_html = generate_comparison_html(comparison_data)
    report_file = os.path.join(output_dir, "comparison_report.html")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_html)

    print(f"Comparison report generated: {report_file}")

def generate_comparison_html(comparison_data):
    """Generate comparison report HTML"""
    rows_html = ""
    for data in comparison_data:
        memory_opt = "✓" if data.get("memory_optimized", False) else "✗"
        rows_html += f"""
        <tr>
            <td>{data['app_package']}</td>
            <td>{data['total_pages']}</td>
            <td>{data['total_transitions']}</td>
            <td>{data['processed_trajectories']}</td>
            <td>{data['avg_success_rate']:.2%}</td>
            <td>{len(data['key_pages'])}</td>
            <td>{memory_opt}</td>
            <td>{data.get('batch_size', 'N/A')}</td>
        </tr>
        """

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Application Page Graph Comparison Report (Memory Optimized)</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .stats {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
        .optimized {{ color: green; font-weight: bold; }}
        .not-optimized {{ color: red; }}
    </style>
</head>
<body>
    <h1>Application Page Graph Comparison Report (Memory Optimized)</h1>

    <div class="stats">
        <h3>Overall Statistics</h3>
        <p>Number of compared applications: {len(comparison_data)}</p>
        <p>Total pages: {sum(d['total_pages'] for d in comparison_data)}</p>
        <p>Total transitions: {sum(d['total_transitions'] for d in comparison_data)}</p>
        <p>Memory optimized applications: {sum(1 for d in comparison_data if d.get('memory_optimized', False))}</p>
    </div>

    <table>
        <thead>
            <tr>
                <th>Application Package</th>
                <th>Pages</th>
                <th>Transitions</th>
                <th>Trajectories</th>
                <th>Avg Success Rate</th>
                <th>Key Pages</th>
                <th>Memory Optimized</th>
                <th>Batch Size</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <div class="stats">
        <h3>Memory Optimization Notes</h3>
        <p>✓ Indicates usage of streaming processing and disk cache memory optimization</p>
        <p>✗ Indicates usage of traditional full-memory processing</p>
        <p>Smaller batch sizes result in lower memory usage but may take slightly longer to process</p>
    </div>
</body>
</html>
    """

def main():
    parser = argparse.ArgumentParser(description="Build application page navigation graph based on existing trajectory data (Memory optimized version)")

    # Basic parameters
    parser.add_argument("--app_package", type=str, help="Application package name to process, e.g., com.android.camera2")
    parser.add_argument("--input_dir", type=str, default="./exploration_output", help="Exploration output directory")
    parser.add_argument("--output_dir", type=str, default="./page_graphs", help="Page graph output directory")

    # Mode parameters
    parser.add_argument("--batch_mode", action="store_true", help="Batch processing mode, process all applications in input directory")
    parser.add_argument("--comparison_report", action="store_true", help="Generate application comparison report")

    # Optional parameters
    parser.add_argument("--no_visualize", action="store_true", help="Do not generate visualization files")
    parser.add_argument("--max_trajectories", type=int, help="Limit the number of trajectory files to process (for testing)")

    # Memory optimization parameters
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size, number of trajectory files to process at once (default 10, reduce to lower memory usage)")
    parser.add_argument("--no_disk_cache", action="store_true",
                       help="Disable disk cache, use all memory (not recommended, may cause out of memory)")
    parser.add_argument("--use_mllm", action="store_true",
                       help="Enable MLLM page analysis (requires network connection, otherwise use fast simplified analysis)")

    args = parser.parse_args()

    # Validate parameters
    if not args.batch_mode and not args.app_package and not args.comparison_report:
        parser.error("Must specify --app_package or use --batch_mode or --comparison_report")

    if args.app_package and args.batch_mode:
        parser.error("Cannot use both --app_package and --batch_mode")

    # Memory optimization settings
    use_disk_cache = not args.no_disk_cache
    batch_size = args.batch_size
    use_mllm = args.use_mllm

    if batch_size > 20:
        print(f"Warning: Batch size {batch_size} is large, may cause high memory usage")
    if not use_disk_cache:
        print(f"Warning: Disk cache disabled, may cause out of memory when processing large amounts of trajectories")
    if use_mllm:
        print(f"Info: MLLM page analysis enabled, processing will be slower but more accurate")
    else:
        print(f"Info: Using fast simplified analysis, add --use_mllm parameter for higher accuracy")

    # Execute corresponding operations
    try:
        if args.comparison_report:
            generate_comparison_report(args.output_dir)

        elif args.batch_mode:
            build_batch_graphs(args.input_dir, args.output_dir, not args.no_visualize,
                             args.max_trajectories, batch_size, use_disk_cache, use_mllm)

        else:
            build_single_app_graph(args.app_package, args.input_dir, args.output_dir,
                                 not args.no_visualize, args.max_trajectories,
                                 batch_size, use_disk_cache, use_mllm)

    except KeyboardInterrupt:
        print("\n\nUser interrupted operation")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nExecution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()