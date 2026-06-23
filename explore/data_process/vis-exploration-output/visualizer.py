"""
Visualization generator for exploration output data
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import base64
from io import BytesIO
import numpy as np
from PIL import Image

# Import local modules
try:
    from .parser import AppData, Trajectory, TrajectoryStep
except ImportError:
    from parser import AppData, Trajectory, TrajectoryStep


class ExplorationVisualizer:
    """Generates visualizations for exploration data"""
    
    def __init__(self, config):
        self.config = config
        self.verbose = config.verbose
        
    def log(self, message: str, level: str = "info"):
        """Log a message if verbose mode is enabled"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level.upper()}: {message}")
    
    def generate_html_report(self, app_data: Dict[str, AppData]):
        """Generate comprehensive HTML report for all apps"""
        output_dir = Path(self.config.output_dir)
        
        # Generate individual app reports
        for package_name, data in app_data.items():
            self.generate_app_html_report(data)
            # Generate individual trajectory detail pages
            self.generate_trajectory_detail_pages(data)
        
        # Generate overview report
        self.generate_overview_html_report(app_data)
    
    def generate_overview_html_report(self, app_data: Dict[str, AppData]):
        """Generate overview HTML report for all apps"""
        output_dir = Path(self.config.output_dir)
        
        html_content = self.create_overview_html(app_data)
        
        with open(output_dir / "overview.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.log(f"Overview HTML report saved to {output_dir / 'overview.html'}")
    
    def generate_app_html_report(self, app_data: AppData):
        """Generate HTML report for a single app"""
        output_dir = Path(self.config.output_dir) / app_data.package_name
        
        html_content = self.create_app_html(app_data)
        
        with open(output_dir / "report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.log(f"HTML report for {app_data.package_name} saved to {output_dir / 'report.html'}")
    
    def generate_trajectory_detail_pages(self, app_data: AppData):
        """Generate individual HTML pages for each trajectory with grid layout"""
        output_dir = Path(self.config.output_dir) / app_data.package_name
        
        for trajectory in app_data.trajectories:
            html_content = self.create_trajectory_detail_html(trajectory, app_data)
            
            trajectory_file = output_dir / f"trajectory_{trajectory.trajectory_id}.html"
            with open(trajectory_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        self.log(f"Generated {len(app_data.trajectories)} trajectory detail pages for {app_data.package_name}")
    
    def create_trajectory_detail_html(self, trajectory: Trajectory, app_data: AppData) -> str:
        """Create HTML content for trajectory detail page with grid layout"""
        status_class = "success" if trajectory.success else "failure"
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trajectory {trajectory.trajectory_id} - {app_data.app_info.get('app_name', app_data.package_name)}</title>
    <style>
        {self.get_trajectory_detail_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="breadcrumb">
                <a href="report.html">← Back to {app_data.app_info.get('app_name', app_data.package_name)}</a>
            </div>
            <h1>📱 Trajectory Details</h1>
            <div class="trajectory-info">
                <span class="trajectory-id">ID: {trajectory.trajectory_id}</span>
                <span class="status {status_class}">{status_class.title()}</span>
                <span class="depth">Depth: {trajectory.depth}</span>
                <span class="steps-count">Steps: {len(trajectory.steps)}</span>
            </div>
        </header>
        
        <div class="goal-section">
            <h2>🎯 Goal</h2>
            <p class="goal-text">{trajectory.goal}</p>
        </div>
        
        <div class="trajectory-metadata">
            <div class="meta-item">
                <strong>Start Time:</strong> {trajectory.start_time.strftime('%Y-%m-%d %H:%M:%S') if trajectory.start_time else 'Unknown'}
            </div>
            <div class="meta-item">
                <strong>Package:</strong> {trajectory.package_name}
            </div>
            <div class="meta-item">
                <strong>Success:</strong> {'✅ Yes' if trajectory.success else '❌ No'}
            </div>
        </div>
        
        <div class="steps-section">
            <h2>📋 Steps ({len(trajectory.steps)} total)</h2>
            <div class="steps-grid">
                {self.generate_trajectory_steps_grid(trajectory)}
            </div>
        </div>
    </div>
    
    <script>
        {self.get_trajectory_detail_js()}
    </script>
</body>
</html>
        """
        return html
    
    def generate_trajectory_steps_grid(self, trajectory: Trajectory) -> str:
        """Generate grid layout for trajectory steps (4 per row)"""
        steps_html = []
        
        for i, step in enumerate(trajectory.steps):
            action_info = self.extract_action_info(step)
            is_last_step = (i == len(trajectory.steps) - 1)
            screenshot_info = self.get_screenshot_info(step, is_last_step, trajectory.trajectory_id, trajectory.package_name)
            
            step_html = f"""
                <div class="step-card" data-step="{i+1}">
                    <div class="step-header">
                        <div class="step-number">{i+1}</div>
                        <div class="step-timestamp">{step.timestamp or 'No timestamp'}</div>
                    </div>
                    
                    <div class="step-content">
                        <div class="step-summary">
                            <strong>Summary:</strong>
                            <p>{step.summary or 'No summary available'}</p>
                        </div>
                        
                        <div class="step-action">
                            <strong>Action:</strong>
                            <p>{action_info}</p>
                        </div>
                        
                        <div class="step-screenshots">
                            {screenshot_info}
                        </div>
                        
                        {f'<div class="step-activity"><strong>Activity:</strong> {", ".join(step.activity)}</div>' if step.activity else ''}
                        {f'<div class="step-ui-elements"><strong>UI Elements:</strong> {len(step.ui_elements)} detected</div>' if step.ui_elements else ''}
                    </div>
                    
                    {f'<div class="step-metadata"><strong>Metadata:</strong> {self.format_metadata(step.metadata)}</div>' if step.metadata else ''}
                </div>
            """
            steps_html.append(step_html)
        
        return ''.join(steps_html)
    
    def get_trajectory_detail_css(self) -> str:
        """Get CSS styles specifically for trajectory detail pages"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .breadcrumb {
            margin-bottom: 15px;
        }
        
        .breadcrumb a {
            color: #3498db;
            text-decoration: none;
            font-size: 0.9em;
        }
        
        .breadcrumb a:hover {
            text-decoration: underline;
        }
        
        header h1 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 2.2em;
        }
        
        .trajectory-info {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        
        .trajectory-id, .depth, .steps-count {
            background: #ecf0f1;
            padding: 5px 12px;
            border-radius: 5px;
            font-size: 0.9em;
            color: #2c3e50;
        }
        
        .status {
            padding: 5px 12px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        
        .status.failure {
            background: #f8d7da;
            color: #721c24;
        }
        
        .goal-section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .goal-section h2 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .goal-text {
            font-size: 1.1em;
            line-height: 1.6;
            color: #555;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .trajectory-metadata {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .meta-item {
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .steps-section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .steps-section h2 {
            color: #2c3e50;
            margin-bottom: 25px;
            font-size: 1.5em;
        }
        
        .steps-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        /* Ensure exactly 4 columns on large screens */
        @media (min-width: 1200px) {
            .steps-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }
        
        @media (min-width: 900px) and (max-width: 1199px) {
            .steps-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        
        @media (min-width: 600px) and (max-width: 899px) {
            .steps-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (max-width: 599px) {
            .steps-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .step-card {
            border: 1px solid #e1e8ed;
            border-radius: 10px;
            padding: 15px;
            background: #f8f9fa;
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }
        
        .step-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .step-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }
        
        .step-number {
            background: #3498db;
            color: white;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        
        .step-timestamp {
            font-size: 0.75em;
            color: #7f8c8d;
            background: white;
            padding: 3px 8px;
            border-radius: 3px;
        }
        
        .step-content {
            margin-bottom: 15px;
        }
        
        .step-summary, .step-action {
            margin-bottom: 12px;
        }
        
        .step-summary strong, .step-action strong {
            color: #2c3e50;
            display: block;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        
        .step-summary p, .step-action p {
            font-size: 0.85em;
            line-height: 1.4;
            color: #555;
            background: white;
            padding: 8px;
            border-radius: 4px;
            margin: 0;
        }
        
        .step-activity, .step-ui-elements {
            font-size: 0.8em;
            color: #7f8c8d;
            margin-bottom: 8px;
        }
        
        .step-screenshots {
            margin: 15px 0;
        }
        
        .screenshot-status {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 10px;
        }
        
        .screenshot-available {
            background: #d4edda;
            color: #155724;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.7em;
            font-weight: bold;
        }
        
        .screenshot-missing {
            background: #f8d7da;
            color: #721c24;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.7em;
        }
        
        .screenshots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .screenshot-item {
            text-align: center;
        }
        
        .screenshot-label {
            font-size: 0.7em;
            color: #7f8c8d;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        .screenshot-image {
            max-width: 100%;
            height: auto;
            max-height: 300px;
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .screenshot-image:hover {
            transform: scale(1.1);
        }
        
        .step-metadata {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #e1e8ed;
            font-size: 0.75em;
            color: #7f8c8d;
            background: white;
            padding: 8px;
            border-radius: 4px;
        }
        
        /* Screenshot Modal */
        .screenshot-modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }
        
        .screenshot-modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
        }
        
        .screenshot-modal img {
            max-width: 100%;
            max-height: 100%;
            border: 2px solid white;
            border-radius: 8px;
        }
        
        .screenshot-modal .close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .screenshot-modal .close:hover {
            color: #bbb;
        }
        """
    
    def get_trajectory_detail_js(self) -> str:
        """Get JavaScript for trajectory detail page interactions"""
        return """
        function openScreenshotModal(img) {
            const modal = document.createElement('div');
            modal.className = 'screenshot-modal';
            modal.innerHTML = `
                <div class="screenshot-modal-content">
                    <span class="close">&times;</span>
                    <img src="${img.src}" alt="${img.alt}">
                </div>
            `;
            
            document.body.appendChild(modal);
            modal.style.display = 'block';
            
            // Close modal when clicking X or outside image
            modal.addEventListener('click', function(e) {
                if (e.target === modal || e.target.className === 'close') {
                    document.body.removeChild(modal);
                }
            });
            
            // Close modal with Escape key
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape' && modal.parentNode) {
                    document.body.removeChild(modal);
                }
            });
        }
        
        // Add click handler for step cards
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('.step-card').forEach(card => {
                card.addEventListener('click', function(e) {
                    // Don't trigger if clicking on a screenshot
                    if (e.target.classList.contains('screenshot-image')) {
                        return;
                    }
                    
                    // Toggle card expansion or highlight
                    this.classList.toggle('expanded');
                });
            });
        });
        """
    
    def create_overview_html(self, app_data: Dict[str, AppData]) -> str:
        """Create overview HTML content"""
        apps_summary = []
        total_trajectories = 0
        total_successful = 0
        total_steps = 0
        
        for package_name, data in app_data.items():
            stats = data.statistics
            apps_summary.append({
                'package_name': package_name,
                'app_name': data.app_info.get('app_name', package_name),
                'trajectories': stats['total_trajectories'],
                'successful': stats['successful_trajectories'],
                'steps': stats['total_steps'],
                'unique_goals': stats['unique_goals'],
                'avg_steps': stats['average_steps_per_trajectory']
            })
            
            total_trajectories += stats['total_trajectories']
            total_successful += stats['successful_trajectories']
            total_steps += stats['total_steps']
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exploration Output Overview</title>
    <style>
        {self.get_css_styles()}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Mobile App Exploration Overview</h1>
            <p class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <div class="summary-cards">
            <div class="card">
                <h3>Total Apps</h3>
                <div class="number">{len(app_data)}</div>
            </div>
            <div class="card">
                <h3>Total Trajectories</h3>
                <div class="number">{total_trajectories}</div>
            </div>
            <div class="card">
                <h3>Successful Trajectories</h3>
                <div class="number">{total_successful}</div>
                <div class="percentage">({total_successful/total_trajectories*100:.1f}%)</div>
            </div>
            <div class="card">
                <h3>Total Steps</h3>
                <div class="number">{total_steps}</div>
            </div>
        </div>
        
        <div class="charts-container">
            <div class="chart-box">
                <h3>Trajectories by App</h3>
                <canvas id="trajectoriesChart"></canvas>
            </div>
            <div class="chart-box">
                <h3>Success Rate by App</h3>
                <canvas id="successChart"></canvas>
            </div>
        </div>
        
        <div class="apps-table">
            <h2>Apps Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>App Name</th>
                        <th>Package Name</th>
                        <th>Trajectories</th>
                        <th>Successful</th>
                        <th>Success Rate</th>
                        <th>Total Steps</th>
                        <th>Avg Steps</th>
                        <th>Unique Goals</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {self.generate_apps_table_rows(apps_summary)}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        {self.generate_overview_charts_js(apps_summary)}
    </script>
</body>
</html>
        """
        return html
    
    def create_app_html(self, app_data: AppData) -> str:
        """Create HTML content for a single app"""
        stats = app_data.statistics
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{app_data.app_info.get('app_name', app_data.package_name)} - Exploration Report</title>
    <style>
        {self.get_css_styles()}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>📱 {app_data.app_info.get('app_name', app_data.package_name)}</h1>
            <p class="subtitle">{app_data.package_name}</p>
            <div class="app-info">
                <span>Version: {app_data.app_info.get('app_version_name', 'Unknown')}</span>
                <span>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
        </header>
        
        <div class="summary-cards">
            <div class="card">
                <h3>Total Trajectories</h3>
                <div class="number">{stats['total_trajectories']}</div>
            </div>
            <div class="card">
                <h3>Successful</h3>
                <div class="number">{stats['successful_trajectories']}</div>
                    <div class="percentage">({(stats['successful_trajectories']/stats['total_trajectories']*100) if stats['total_trajectories'] > 0 else 0:.1f}%)</div>
            </div>
            <div class="card">
                <h3>Total Steps</h3>
                <div class="number">{stats['total_steps']}</div>
            </div>
            <div class="card">
                <h3>Avg Steps</h3>
                <div class="number">{stats['average_steps_per_trajectory']:.1f}</div>
            </div>
        </div>
        
        <div class="charts-container">
            <div class="chart-box">
                <h3>Trajectories by Depth</h3>
                <canvas id="depthChart"></canvas>
            </div>
            <div class="chart-box">
                <h3>Success Rate by Depth</h3>
                <canvas id="depthSuccessChart"></canvas>
            </div>
        </div>
        
        <div class="depth-analysis">
            <h2>Depth Analysis</h2>
            {self.generate_depth_analysis(app_data)}
        </div>
        
        <div class="trajectories-section">
            <h2>Trajectories ({len(app_data.trajectories)})</h2>
            <div class="trajectories-grid">
                {self.generate_trajectories_cards(app_data.trajectories)}
            </div>
        </div>
    </div>
    
    <script>
        {self.generate_app_charts_js(app_data)}
    </script>
</body>
</html>
        """
        return html
    
    def generate_apps_table_rows(self, apps_summary: List[Dict]) -> str:
        """Generate table rows for apps summary"""
        rows = []
        for app in apps_summary:
            success_rate = (app['successful'] / app['trajectories'] * 100) if app['trajectories'] > 0 else 0
            rows.append(f"""
                <tr>
                    <td>{app['app_name']}</td>
                    <td>{app['package_name']}</td>
                    <td>{app['trajectories']}</td>
                    <td>{app['successful']}</td>
                    <td>{success_rate:.1f}%</td>
                    <td>{app['steps']}</td>
                    <td>{app['avg_steps']:.1f}</td>
                    <td>{app['unique_goals']}</td>
                    <td><a href="{app['package_name']}/report.html">View Details</a></td>
                </tr>
            """)
        return ''.join(rows)
    
    def generate_depth_analysis(self, app_data: AppData) -> str:
        """Generate depth analysis HTML"""
        depth_cards = []
        for depth in sorted(app_data.depth_structure.keys()):
            trajectories = app_data.depth_structure[depth]
            
            # Calculate stats from actual data if not available in statistics
            if 'depth_distribution' in app_data.statistics and depth in app_data.statistics['depth_distribution']:
                stats = app_data.statistics['depth_distribution'][depth]
            else:
                # Calculate stats from trajectories directly
                successful = sum(1 for t in trajectories if t.success)
                total_steps = sum(len(t.steps) for t in trajectories)
                stats = {
                    'count': len(trajectories),
                    'successful': successful,
                    'average_steps': total_steps / len(trajectories) if trajectories else 0
                }
            
            depth_cards.append(f"""
                <div class="depth-card">
                    <h4>Depth {depth}</h4>
                    <div class="depth-stats">
                        <div class="stat">
                            <span class="label">Trajectories:</span>
                            <span class="value">{stats['count']}</span>
                        </div>
                        <div class="stat">
                            <span class="label">Successful:</span>
                            <span class="value">{stats['successful']}</span>
                        </div>
                        <div class="stat">
                            <span class="label">Success Rate:</span>
                            <span class="value">{stats['successful']/stats['count']*100:.1f}%</span>
                        </div>
                        <div class="stat">
                            <span class="label">Avg Steps:</span>
                            <span class="value">{stats['average_steps']:.1f}</span>
                        </div>
                    </div>
                    <div class="goals-list">
                        <strong>Goals:</strong>
                        <ul>
                            {self.generate_goals_list(trajectories[:5])}
                        </ul>
                        {f'<p>... and {len(trajectories)-5} more</p>' if len(trajectories) > 5 else ''}
                    </div>
                </div>
            """)
        
        return f'<div class="depth-cards">{"".join(depth_cards)}</div>'
    
    def generate_goals_list(self, trajectories: List[Trajectory]) -> str:
        """Generate list of goals for trajectories"""
        return ''.join([f'<li>{t.goal}</li>' for t in trajectories])
    
    def generate_trajectories_cards(self, trajectories: List[Trajectory]) -> str:
        """Generate trajectory cards with detailed step information"""
        cards = []
        for trajectory in trajectories[:20]:  # Limit to first 20 for performance
            status_class = "success" if trajectory.success else "failure"
            
            # Generate steps preview
            steps_preview = self.generate_steps_preview(trajectory.steps[:3])
            
            cards.append(f"""
                <div class="trajectory-card {status_class}">
                    <div class="trajectory-header">
                        <h4>Depth {trajectory.depth}</h4>
                        <span class="status">{status_class.title()}</span>
                    </div>
                    <div class="trajectory-goal">
                        <strong>Goal:</strong> {trajectory.goal}
                    </div>
                    <div class="trajectory-stats">
                        <span>Steps: {len(trajectory.steps)}</span>
                        <span>Time: {trajectory.start_time.strftime('%H:%M:%S') if trajectory.start_time else 'Unknown'}</span>
                    </div>
                    
                    <div class="steps-preview">
                        <h5>Steps Preview:</h5>
                        {steps_preview}
                        {f'<p class="more-steps">... and {len(trajectory.steps)-3} more steps</p>' if len(trajectory.steps) > 3 else ''}
                    </div>
                    
                    <div class="trajectory-actions">
                        <a href="trajectory_{trajectory.trajectory_id}.html" class="btn-details" style="text-decoration: none; display: inline-block;">
                            View Full Trajectory
                        </a>
                    </div>
                </div>
            """)
        
        if len(trajectories) > 20:
            cards.append(f'<div class="more-trajectories">... and {len(trajectories)-20} more trajectories</div>')
        
        return ''.join(cards)
    
    def generate_steps_preview(self, steps: List[TrajectoryStep]) -> str:
        """Generate HTML preview for trajectory steps"""
        if not steps:
            return "<p>No steps available</p>"
        
        preview_html = "<div class='steps-preview-list'>"
        for i, step in enumerate(steps):
            action_info = self.extract_action_info(step)
            preview_html += f"""
                <div class='step-preview'>
                    <span class='step-number'>{i+1}</span>
                    <div class='step-content'>
                        <div class='step-summary'>{step.summary or 'No summary'}</div>
                        <div class='step-action'>{action_info}</div>
                    </div>
                </div>
            """
        preview_html += "</div>"
        return preview_html
    
    def generate_full_trajectory_details(self, trajectory: Trajectory) -> str:
        """Generate complete HTML details for a trajectory"""
        steps_html = "<div class='trajectory-steps'>"
        
        for i, step in enumerate(trajectory.steps):
            action_info = self.extract_action_info(step)
            is_last_step = (i == len(trajectory.steps) - 1)
            screenshot_info = self.get_screenshot_info(step, is_last_step, trajectory.trajectory_id, trajectory.package_name)
            
            steps_html += f"""
                <div class='step-detail'>
                    <div class='step-header'>
                        <h6>Step {i+1}</h6>
                        <span class='step-timestamp'>{step.timestamp or 'No timestamp'}</span>
                    </div>
                    
                    <div class='step-body'>
                        <div class='step-text-info'>
                            <div class='step-goal'><strong>Goal:</strong> {step.goal or 'No goal'}</div>
                            <div class='step-summary'><strong>Summary:</strong> {step.summary or 'No summary'}</div>
                            <div class='step-action'><strong>Action:</strong> {action_info}</div>
                            
                            {f'<div class="step-activity"><strong>Activity:</strong> {", ".join(step.activity)}</div>' if step.activity else ''}
                            {f'<div class="step-ui-elements"><strong>UI Elements:</strong> {len(step.ui_elements)} elements detected</div>' if step.ui_elements else ''}
                        </div>
                        
                        <div class='step-screenshots'>
                            {screenshot_info}
                        </div>
                    </div>
                    
                    {f'<div class="step-metadata"><strong>Metadata:</strong> {self.format_metadata(step.metadata)}</div>' if step.metadata else ''}
                </div>
            """
        
        steps_html += "</div>"
        return steps_html
    
    def extract_action_info(self, step: TrajectoryStep) -> str:
        """Extract readable action information from step"""
        if not step.converted_action:
            return "No action"
        
        action_type = str(type(step.converted_action).__name__)
        
        # Try to extract meaningful action details
        if hasattr(step.converted_action, '__dict__'):
            action_dict = step.converted_action.__dict__
            if 'action_type' in action_dict:
                action_type = action_dict['action_type']
            
            # Extract coordinates for click actions
            if 'x' in action_dict and 'y' in action_dict:
                return f"{action_type} at ({action_dict['x']}, {action_dict['y']})"
            elif 'text' in action_dict:
                return f"{action_type}: '{action_dict['text']}'"
        
        return action_type
    
    def get_screenshot_info(self, step: TrajectoryStep, is_last_step: bool = False, trajectory_id: str = None, package_name: str = None) -> str:
        """Get screenshot availability info and display screenshots"""
        screenshots_html = []
        
        # Status indicators
        status_indicators = []
        screenshot_images = []
        
        # For trajectory visualization: prioritize before_screenshot as requested
        # For regular steps: show before_screenshot first, then before_screenshot_with_som
        # For last step: show both before and after screenshots
        if is_last_step:
            screenshot_types = [
                ('before_screenshot', 'Before', 'before_screenshot.png'),
                ('after_screenshot_with_som', 'After (SOM)', 'after_screenshot_with_som.png')
            ]
        else:
            screenshot_types = [
                ('before_screenshot', 'Before', 'before_screenshot.png')
            ]
        
        for attr_name, display_name, filename in screenshot_types:
            # First check if screenshot is available in memory
            screenshot = getattr(step, attr_name, None)
            screenshot_available = False
            screenshot_path = None
            
            if screenshot is not None:
                screenshot_available = True
            else:
                # Check if screenshot file exists on disk
                if trajectory_id and package_name:
                    screenshot_file_path = Path(self.config.output_dir) / package_name / "screenshots" / trajectory_id / f"step_{step.step_index:02d}" / filename
                    if screenshot_file_path.exists():
                        screenshot_available = True
                        screenshot_path = screenshot_file_path
            
            if screenshot_available:
                status_indicators.append(f"<span class='screenshot-available'>{display_name} ✓</span>")
                if self.config.include_screenshots:
                    if screenshot is not None:
                        # Convert in-memory screenshot to base64 for HTML embedding
                        img_base64 = self.numpy_to_base64(screenshot)
                        if img_base64:
                            screenshot_images.append(f"""
                                <div class='screenshot-item'>
                                    <div class='screenshot-label'>{display_name}</div>
                                    <img src="data:image/png;base64,{img_base64}" 
                                         alt="{display_name}" 
                                         class='screenshot-image'
                                         onclick="openScreenshotModal(this)">
                                </div>
                            """)
                    elif screenshot_path:
                        # Use relative path to screenshot file
                        # Since the HTML file is in the app directory, make path relative to that
                        app_dir = Path(self.config.output_dir) / package_name
                        relative_path = screenshot_path.relative_to(app_dir)
                        screenshot_images.append(f"""
                            <div class='screenshot-item'>
                                <div class='screenshot-label'>{display_name}</div>
                                <img src="{relative_path}" 
                                     alt="{display_name}" 
                                     class='screenshot-image'
                                     onclick="openScreenshotModal(this)">
                            </div>
                        """)
            else:
                status_indicators.append(f"<span class='screenshot-missing'>{display_name} ✗</span>")
        
        screenshots_section = ""
        if screenshot_images:
            screenshots_section = f"""
                <div class='screenshots-grid'>
                    {''.join(screenshot_images)}
                </div>
            """
        
        return f"""
            <div class='screenshot-status'>{''.join(status_indicators)}</div>
            {screenshots_section}
        """
    
    def format_metadata(self, metadata: dict) -> str:
        """Format metadata for display"""
        if not metadata:
            return "None"
        
        formatted = []
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                formatted.append(f"{key}: {len(value)} items")
            else:
                formatted.append(f"{key}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
        
        return "; ".join(formatted[:5]) + ("..." if len(formatted) > 5 else "")
    
    def numpy_to_base64(self, image_array: np.ndarray) -> str:
        """Convert numpy array to base64 string for HTML embedding"""
        try:
            if image_array is None:
                return None
            
            # Check memory size - skip very large images
            image_size = image_array.nbytes
            if image_size > 50 * 1024 * 1024:  # Skip images larger than 50MB
                self.log(f"Skipping large image ({image_size / 1024 / 1024:.1f}MB)", "warning")
                return None
            
            # Ensure the array is in the right format
            if len(image_array.shape) == 3:
                # RGB or RGBA image
                if image_array.shape[2] == 4:
                    # RGBA
                    image = Image.fromarray(image_array.astype('uint8'), 'RGBA')
                else:
                    # RGB
                    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
            elif len(image_array.shape) == 2:
                # Grayscale image
                image = Image.fromarray(image_array.astype('uint8'), 'L')
            else:
                return None
            
            # Resize image for web display (larger size for better visibility)
            image.thumbnail((400, 800), Image.Resampling.LANCZOS)
            
            # Convert to JPEG with compression to reduce size
            buffer = BytesIO()
            if image.mode in ('RGBA', 'LA'):
                # Convert RGBA to RGB for JPEG
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                rgb_image.save(buffer, format='JPEG', quality=70, optimize=True)
            else:
                image.save(buffer, format='JPEG', quality=70, optimize=True)
            
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            # Check final size
            if len(img_str) > 1024 * 1024:  # Skip if base64 > 1MB
                self.log(f"Skipping large base64 image ({len(img_str) / 1024:.1f}KB)", "warning")
                return None
            
            return img_str
            
        except Exception as e:
            self.log(f"Error converting image to base64: {e}", "error")
            return None
    
    def generate_overview_charts_js(self, apps_summary: List[Dict]) -> str:
        """Generate JavaScript for overview charts"""
        app_names = [app['app_name'][:15] + '...' if len(app['app_name']) > 15 else app['app_name'] for app in apps_summary]
        trajectories_data = [app['trajectories'] for app in apps_summary]
        success_rates = [(app['successful'] / app['trajectories'] * 100) if app['trajectories'] > 0 else 0 for app in apps_summary]
        
        return f"""
        // Trajectories Chart
        const trajCtx = document.getElementById('trajectoriesChart').getContext('2d');
        new Chart(trajCtx, {{
            type: 'bar',
            data: {{
                labels: {app_names},
                datasets: [{{
                    label: 'Trajectories',
                    data: {trajectories_data},
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Success Rate Chart
        const successCtx = document.getElementById('successChart').getContext('2d');
        new Chart(successCtx, {{
            type: 'bar',
            data: {{
                labels: {app_names},
                datasets: [{{
                    label: 'Success Rate (%)',
                    data: {success_rates},
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
        """
    
    def generate_app_charts_js(self, app_data: AppData) -> str:
        """Generate JavaScript for app-specific charts"""
        depths = sorted(app_data.depth_structure.keys())
        depth_counts = [len(app_data.depth_structure[d]) for d in depths]
        
        # Calculate success rates safely
        depth_success_rates = []
        for d in depths:
            if 'depth_distribution' in app_data.statistics and d in app_data.statistics['depth_distribution']:
                stats = app_data.statistics['depth_distribution'][d]
                success_rate = (stats['successful'] / stats['count'] * 100) if stats['count'] > 0 else 0
            else:
                # Calculate from actual data
                trajectories = app_data.depth_structure[d]
                successful = sum(1 for t in trajectories if t.success)
                success_rate = (successful / len(trajectories) * 100) if trajectories else 0
            depth_success_rates.append(success_rate)
        
        return f"""
        // Depth Distribution Chart
        const depthCtx = document.getElementById('depthChart').getContext('2d');
        new Chart(depthCtx, {{
            type: 'bar',
            data: {{
                labels: {[f'Depth {d}' for d in depths]},
                datasets: [{{
                    label: 'Trajectories',
                    data: {depth_counts},
                    backgroundColor: 'rgba(153, 102, 255, 0.6)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Depth Success Rate Chart
        const depthSuccessCtx = document.getElementById('depthSuccessChart').getContext('2d');
        new Chart(depthSuccessCtx, {{
            type: 'line',
            data: {{
                labels: {[f'Depth {d}' for d in depths]},
                datasets: [{{
                    label: 'Success Rate (%)',
                    data: {depth_success_rates},
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
        
        function toggleTrajectoryDetails(trajectoryId) {{
            const detailsDiv = document.getElementById('details-' + trajectoryId);
            const button = event.target;
            
            if (detailsDiv.style.display === 'none') {{
                detailsDiv.style.display = 'block';
                button.textContent = 'Hide Full Trajectory';
            }} else {{
                detailsDiv.style.display = 'none';
                button.textContent = 'View Full Trajectory';
            }}
        }}
        
        function openScreenshotModal(img) {{
            const modal = document.createElement('div');
            modal.className = 'screenshot-modal';
            modal.innerHTML = `
                <div class="screenshot-modal-content">
                    <span class="close">&times;</span>
                    <img src="${{img.src}}" alt="${{img.alt}}">
                </div>
            `;
            
            document.body.appendChild(modal);
            modal.style.display = 'block';
            
            // Close modal when clicking X or outside image
            modal.addEventListener('click', function(e) {{
                if (e.target === modal || e.target.className === 'close') {{
                    document.body.removeChild(modal);
                }}
            }});
            
            // Close modal with Escape key
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape' && modal.parentNode) {{
                    document.body.removeChild(modal);
                }}
            }});
        }}
        """
    
    def get_css_styles(self) -> str:
        """Get CSS styles for HTML reports"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }
        
        header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #7f8c8d;
            font-size: 1.1em;
        }
        
        .app-info {
            margin-top: 15px;
            color: #95a5a6;
        }
        
        .app-info span {
            margin: 0 15px;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .card h3 {
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        
        .card .number {
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .card .percentage {
            color: #27ae60;
            font-weight: bold;
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .chart-box {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .chart-box h3 {
            margin-bottom: 20px;
            color: #2c3e50;
        }
        
        .apps-table, .trajectories-section, .depth-analysis {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .apps-table h2, .trajectories-section h2, .depth-analysis h2 {
            margin-bottom: 20px;
            color: #2c3e50;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        
        tr:hover {
            background-color: #f8f9fa;
        }
        
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .depth-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .depth-card {
            border: 1px solid #e1e8ed;
            border-radius: 8px;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .depth-card h4 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        
        .depth-stats {
            margin-bottom: 15px;
        }
        
        .stat {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .stat .label {
            color: #7f8c8d;
        }
        
        .stat .value {
            font-weight: bold;
        }
        
        .goals-list ul {
            margin-left: 20px;
            margin-top: 10px;
        }
        
        .goals-list li {
            margin-bottom: 5px;
            color: #555;
        }
        
        .trajectories-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .trajectory-card {
            border: 1px solid #e1e8ed;
            border-radius: 8px;
            padding: 15px;
            background: white;
        }
        
        .trajectory-card.success {
            border-left: 4px solid #27ae60;
        }
        
        .trajectory-card.failure {
            border-left: 4px solid #e74c3c;
        }
        
        .trajectory-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .trajectory-header h4 {
            color: #2c3e50;
        }
        
        .status {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .status.Success {
            background: #d4edda;
            color: #155724;
        }
        
        .status.Failure {
            background: #f8d7da;
            color: #721c24;
        }
        
        .trajectory-goal {
            margin-bottom: 10px;
            color: #555;
        }
        
        .trajectory-stats {
            display: flex;
            justify-content: space-between;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        
        .trajectory-actions {
            text-align: right;
        }
        
        .more-trajectories {
            grid-column: 1 / -1;
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            padding: 20px;
        }
        
        /* Steps Preview Styles */
        .steps-preview {
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        
        .steps-preview h5 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        
        .steps-preview-list {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        .step-preview {
            display: flex;
            align-items: flex-start;
            padding: 5px;
            background: white;
            border-radius: 3px;
            border-left: 2px solid #3498db;
        }
        
        .step-number {
            background: #3498db;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 10px;
            flex-shrink: 0;
        }
        
        .step-content {
            flex: 1;
        }
        
        .step-summary {
            font-size: 0.85em;
            color: #2c3e50;
            margin-bottom: 2px;
        }
        
        .step-action {
            font-size: 0.8em;
            color: #7f8c8d;
            font-style: italic;
        }
        
        .more-steps {
            margin-top: 5px;
            font-size: 0.8em;
            color: #7f8c8d;
            text-align: center;
        }
        
        /* Trajectory Details Styles */
        .trajectory-details {
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e1e8ed;
        }
        
        .trajectory-steps {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .step-detail {
            background: white;
            border-radius: 5px;
            padding: 15px;
            border-left: 3px solid #3498db;
        }
        
        .step-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e1e8ed;
        }
        
        .step-header h6 {
            color: #2c3e50;
            font-size: 1em;
            margin: 0;
        }
        
        .step-timestamp {
            font-size: 0.8em;
            color: #7f8c8d;
        }
        
        .step-body {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 15px;
            margin-bottom: 10px;
        }
        
        .step-text-info {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .step-goal, .step-summary, .step-action, .step-activity, .step-ui-elements {
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .step-goal strong, .step-summary strong, .step-action strong {
            color: #2c3e50;
        }
        
        .step-screenshots {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 120px;
        }
        
        .screenshot-status {
            display: flex;
            flex-direction: column;
            gap: 2px;
            font-size: 0.75em;
        }
        
        .screenshot-available {
            color: #27ae60;
            font-weight: bold;
        }
        
        .screenshot-missing {
            color: #e74c3c;
        }
        
        .step-metadata {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #e1e8ed;
            font-size: 0.8em;
            color: #7f8c8d;
        }
        
        /* Button Styles */
        .btn-details {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background-color 0.3s;
        }
        
        .btn-details:hover {
            background: #2980b9;
        }
        
        /* Screenshot Styles */
        .screenshots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        
        .screenshot-item {
            text-align: center;
        }
        
        .screenshot-label {
            font-size: 0.7em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        
        .screenshot-image {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .screenshot-image:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Screenshot Modal */
        .screenshot-modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }
        
        .screenshot-modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
        }
        
        .screenshot-modal img {
            max-width: 100%;
            max-height: 100%;
            border: 2px solid white;
            border-radius: 8px;
        }
        
        .screenshot-modal .close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .screenshot-modal .close:hover {
            color: #bbb;
        }
        
        @media (max-width: 768px) {
            .charts-container {
                grid-template-columns: 1fr;
            }
            
            .summary-cards {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .trajectories-grid {
                grid-template-columns: 1fr;
            }
            
            .step-body {
                grid-template-columns: 1fr;
                gap: 10px;
            }
            
            .step-screenshots {
                min-width: auto;
            }
            
            .screenshot-status {
                flex-direction: row;
                flex-wrap: wrap;
                gap: 5px;
            }
        }
        """
