"""
Statistics generator for exploration output data
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from parser import AppData, Trajectory, TrajectoryStep


class StatisticsGenerator:
    """Generates comprehensive statistics for exploration data"""
    
    def __init__(self, config):
        self.config = config
        self.verbose = config.verbose
        
    def log(self, message: str, level: str = "info"):
        """Log a message if verbose mode is enabled"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level.upper()}: {message}")
    
    def generate_statistics(self, app_data: Dict[str, AppData]):
        """Generate comprehensive statistics for all apps"""
        output_dir = Path(self.config.output_dir)
        
        # Generate overview statistics
        overview_stats = self.generate_overview_statistics(app_data)
        
        # Save overview statistics
        with open(output_dir / "overview_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(overview_stats, f, indent=2, ensure_ascii=False)
        
        # Generate CSV summaries
        self.generate_csv_summaries(app_data, output_dir)
        
        # Generate detailed analysis
        detailed_analysis = self.generate_detailed_analysis(app_data)
        
        with open(output_dir / "detailed_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)
        
        self.log("Statistics generation completed")
    
    def generate_overview_statistics(self, app_data: Dict[str, AppData]) -> Dict:
        """Generate overview statistics across all apps"""
        stats = {
            'summary': {
                'total_apps': len(app_data),
                'total_trajectories': 0,
                'total_successful_trajectories': 0,
                'total_steps': 0,
                'total_unique_goals': 0,
                'generation_time': datetime.now().isoformat()
            },
            'apps_breakdown': {},
            'depth_analysis': self.analyze_depth_distribution(app_data),
            'goal_analysis': self.analyze_goals_across_apps(app_data),
            'success_patterns': self.analyze_success_patterns(app_data),
            'temporal_analysis': self.analyze_temporal_patterns(app_data),
            'complexity_metrics': self.calculate_complexity_metrics(app_data)
        }
        
        # Calculate summary statistics
        for package_name, data in app_data.items():
            app_stats = data.statistics
            stats['summary']['total_trajectories'] += app_stats['total_trajectories']
            stats['summary']['total_successful_trajectories'] += app_stats['successful_trajectories']
            stats['summary']['total_steps'] += app_stats['total_steps']
            stats['summary']['total_unique_goals'] += app_stats['unique_goals']
            
            # Store individual app breakdown
            stats['apps_breakdown'][package_name] = {
                'app_name': data.app_info.get('app_name', package_name),
                'trajectories': app_stats['total_trajectories'],
                'successful': app_stats['successful_trajectories'],
                'success_rate': (app_stats['successful_trajectories'] / app_stats['total_trajectories'] * 100) if app_stats['total_trajectories'] > 0 else 0,
                'steps': app_stats['total_steps'],
                'avg_steps': app_stats['average_steps_per_trajectory'],
                'unique_goals': app_stats['unique_goals'],
                'depth_range': [min(data.depth_structure.keys()), max(data.depth_structure.keys())] if data.depth_structure else [0, 0]
            }
        
        # Calculate derived metrics
        if stats['summary']['total_trajectories'] > 0:
            stats['summary']['overall_success_rate'] = stats['summary']['total_successful_trajectories'] / stats['summary']['total_trajectories'] * 100
            stats['summary']['average_steps_per_trajectory'] = stats['summary']['total_steps'] / stats['summary']['total_trajectories']
            stats['summary']['average_trajectories_per_app'] = stats['summary']['total_trajectories'] / len(app_data)
        else:
            stats['summary']['overall_success_rate'] = 0
            stats['summary']['average_steps_per_trajectory'] = 0
            stats['summary']['average_trajectories_per_app'] = 0
        
        return stats
    
    def analyze_depth_distribution(self, app_data: Dict[str, AppData]) -> Dict:
        """Analyze depth distribution across all apps"""
        depth_stats = defaultdict(lambda: {
            'total_trajectories': 0,
            'successful_trajectories': 0,
            'total_steps': 0,
            'apps_with_depth': 0,
            'goals': []
        })
        
        apps_with_depth = defaultdict(set)
        
        for package_name, data in app_data.items():
            for depth, trajectories in data.depth_structure.items():
                depth_stats[depth]['total_trajectories'] += len(trajectories)
                depth_stats[depth]['successful_trajectories'] += sum(1 for t in trajectories if t.success)
                depth_stats[depth]['total_steps'] += sum(len(t.steps) for t in trajectories)
                depth_stats[depth]['goals'].extend([t.goal for t in trajectories])
                apps_with_depth[depth].add(package_name)
        
        # Convert to regular dict and add derived metrics
        result = {}
        for depth, stats in depth_stats.items():
            result[depth] = {
                'total_trajectories': stats['total_trajectories'],
                'successful_trajectories': stats['successful_trajectories'],
                'success_rate': (stats['successful_trajectories'] / stats['total_trajectories'] * 100) if stats['total_trajectories'] > 0 else 0,
                'total_steps': stats['total_steps'],
                'average_steps': stats['total_steps'] / stats['total_trajectories'] if stats['total_trajectories'] > 0 else 0,
                'apps_with_depth': len(apps_with_depth[depth]),
                'unique_goals': len(set(stats['goals'])),
                'most_common_goals': self.get_most_common_items(stats['goals'], 5)
            }
        
        return result
    
    def analyze_goals_across_apps(self, app_data: Dict[str, AppData]) -> Dict:
        """Analyze goal patterns across all apps"""
        all_goals = []
        goal_success_map = defaultdict(list)  # goal -> list of success flags
        goal_apps_map = defaultdict(set)  # goal -> set of apps
        goal_steps_map = defaultdict(list)  # goal -> list of step counts
        
        for package_name, data in app_data.items():
            for trajectory in data.trajectories:
                all_goals.append(trajectory.goal)
                goal_success_map[trajectory.goal].append(trajectory.success)
                goal_apps_map[trajectory.goal].add(package_name)
                goal_steps_map[trajectory.goal].append(len(trajectory.steps))
        
        # Analyze goal patterns
        goal_analysis = {
            'total_unique_goals': len(set(all_goals)),
            'most_common_goals': self.get_most_common_items(all_goals, 10),
            'cross_app_goals': {},
            'goal_success_patterns': {},
            'goal_complexity_analysis': {}
        }
        
        # Find goals that appear across multiple apps
        for goal, apps in goal_apps_map.items():
            if len(apps) > 1:
                successes = goal_success_map[goal]
                steps = goal_steps_map[goal]
                goal_analysis['cross_app_goals'][goal] = {
                    'apps': list(apps),
                    'app_count': len(apps),
                    'total_attempts': len(successes),
                    'success_count': sum(successes),
                    'success_rate': sum(successes) / len(successes) * 100,
                    'average_steps': np.mean(steps),
                    'steps_std': np.std(steps)
                }
        
        # Analyze success patterns by goal type
        goal_keywords = ['open', 'search', 'add', 'delete', 'edit', 'view', 'create', 'save', 'send']
        for keyword in goal_keywords:
            matching_goals = [g for g in all_goals if keyword.lower() in g.lower()]
            if matching_goals:
                successes = []
                steps = []
                for goal in matching_goals:
                    successes.extend(goal_success_map[goal])
                    steps.extend(goal_steps_map[goal])
                
                goal_analysis['goal_success_patterns'][keyword] = {
                    'goal_count': len(set(matching_goals)),
                    'attempt_count': len(successes),
                    'success_rate': sum(successes) / len(successes) * 100 if successes else 0,
                    'average_steps': np.mean(steps) if steps else 0,
                    'complexity_score': np.mean(steps) * (100 - (sum(successes) / len(successes) * 100)) / 100 if successes else 0
                }
        
        return goal_analysis
    
    def analyze_success_patterns(self, app_data: Dict[str, AppData]) -> Dict:
        """Analyze patterns in successful vs failed trajectories"""
        successful_trajectories = []
        failed_trajectories = []
        
        for package_name, data in app_data.items():
            for trajectory in data.trajectories:
                if trajectory.success:
                    successful_trajectories.append(trajectory)
                else:
                    failed_trajectories.append(trajectory)
        
        return {
            'success_distribution': {
                'successful_count': len(successful_trajectories),
                'failed_count': len(failed_trajectories),
                'overall_success_rate': len(successful_trajectories) / (len(successful_trajectories) + len(failed_trajectories)) * 100 if (len(successful_trajectories) + len(failed_trajectories)) > 0 else 0
            },
            'step_analysis': {
                'successful_avg_steps': np.mean([len(t.steps) for t in successful_trajectories]) if successful_trajectories else 0,
                'failed_avg_steps': np.mean([len(t.steps) for t in failed_trajectories]) if failed_trajectories else 0,
                'step_difference': np.mean([len(t.steps) for t in successful_trajectories]) - np.mean([len(t.steps) for t in failed_trajectories]) if successful_trajectories and failed_trajectories else 0
            },
            'depth_success_correlation': self.calculate_depth_success_correlation(app_data),
            'goal_length_impact': self.analyze_goal_length_impact(successful_trajectories, failed_trajectories)
        }
    
    def analyze_temporal_patterns(self, app_data: Dict[str, AppData]) -> Dict:
        """Analyze temporal patterns in exploration"""
        temporal_data = []
        
        for package_name, data in app_data.items():
            for trajectory in data.trajectories:
                if trajectory.start_time:
                    temporal_data.append({
                        'app': package_name,
                        'timestamp': trajectory.start_time,
                        'hour': trajectory.start_time.hour,
                        'day': trajectory.start_time.day,
                        'success': trajectory.success,
                        'steps': len(trajectory.steps),
                        'depth': trajectory.depth
                    })
        
        if not temporal_data:
            return {'message': 'No temporal data available'}
        
        # Analyze patterns by hour
        hourly_stats = defaultdict(lambda: {'count': 0, 'success': 0, 'steps': []})
        for item in temporal_data:
            hour = item['hour']
            hourly_stats[hour]['count'] += 1
            if item['success']:
                hourly_stats[hour]['success'] += 1
            hourly_stats[hour]['steps'].append(item['steps'])
        
        hourly_analysis = {}
        for hour, stats in hourly_stats.items():
            hourly_analysis[hour] = {
                'trajectory_count': stats['count'],
                'success_rate': stats['success'] / stats['count'] * 100,
                'average_steps': np.mean(stats['steps'])
            }
        
        return {
            'total_temporal_records': len(temporal_data),
            'time_span': {
                'earliest': min(item['timestamp'] for item in temporal_data).isoformat(),
                'latest': max(item['timestamp'] for item in temporal_data).isoformat()
            },
            'hourly_patterns': hourly_analysis,
            'peak_activity_hour': max(hourly_stats.keys(), key=lambda h: hourly_stats[h]['count']) if hourly_stats else None,
            'most_successful_hour': max(hourly_stats.keys(), key=lambda h: hourly_stats[h]['success']) if hourly_stats else None
        }
    
    def calculate_complexity_metrics(self, app_data: Dict[str, AppData]) -> Dict:
        """Calculate various complexity metrics"""
        all_trajectories = []
        for data in app_data.values():
            all_trajectories.extend(data.trajectories)
        
        if not all_trajectories:
            return {}
        
        step_counts = [len(t.steps) for t in all_trajectories]
        depth_counts = [t.depth for t in all_trajectories]
        
        return {
            'step_complexity': {
                'min_steps': min(step_counts),
                'max_steps': max(step_counts),
                'mean_steps': np.mean(step_counts),
                'median_steps': np.median(step_counts),
                'std_steps': np.std(step_counts),
                'percentiles': {
                    '25th': np.percentile(step_counts, 25),
                    '75th': np.percentile(step_counts, 75),
                    '90th': np.percentile(step_counts, 90),
                    '95th': np.percentile(step_counts, 95)
                }
            },
            'depth_complexity': {
                'min_depth': min(depth_counts),
                'max_depth': max(depth_counts),
                'mean_depth': np.mean(depth_counts),
                'median_depth': np.median(depth_counts),
                'depth_distribution': dict(Counter(depth_counts))
            },
            'exploration_coverage': {
                'apps_with_single_depth': sum(1 for data in app_data.values() if len(data.depth_structure) == 1),
                'apps_with_multiple_depths': sum(1 for data in app_data.values() if len(data.depth_structure) > 1),
                'max_depth_reached': max(depth_counts),
                'average_depths_per_app': np.mean([len(data.depth_structure) for data in app_data.values()])
            }
        }
    
    def calculate_depth_success_correlation(self, app_data: Dict[str, AppData]) -> Dict:
        """Calculate correlation between depth and success rate"""
        depth_success_data = []
        
        for data in app_data.values():
            for depth, trajectories in data.depth_structure.items():
                for trajectory in trajectories:
                    depth_success_data.append((depth, 1 if trajectory.success else 0))
        
        if len(depth_success_data) < 2:
            return {'correlation': 0, 'message': 'Insufficient data for correlation'}
        
        depths = [d[0] for d in depth_success_data]
        successes = [d[1] for d in depth_success_data]
        
        correlation = np.corrcoef(depths, successes)[0, 1] if len(set(depths)) > 1 else 0
        
        return {
            'correlation': correlation,
            'interpretation': self.interpret_correlation(correlation),
            'data_points': len(depth_success_data)
        }
    
    def analyze_goal_length_impact(self, successful_trajectories: List[Trajectory], failed_trajectories: List[Trajectory]) -> Dict:
        """Analyze impact of goal length on success"""
        if not successful_trajectories and not failed_trajectories:
            return {}
        
        successful_goal_lengths = [len(t.goal.split()) for t in successful_trajectories]
        failed_goal_lengths = [len(t.goal.split()) for t in failed_trajectories]
        
        return {
            'successful_trajectories': {
                'avg_goal_length': np.mean(successful_goal_lengths) if successful_goal_lengths else 0,
                'median_goal_length': np.median(successful_goal_lengths) if successful_goal_lengths else 0
            },
            'failed_trajectories': {
                'avg_goal_length': np.mean(failed_goal_lengths) if failed_goal_lengths else 0,
                'median_goal_length': np.median(failed_goal_lengths) if failed_goal_lengths else 0
            },
            'length_difference': (np.mean(successful_goal_lengths) - np.mean(failed_goal_lengths)) if successful_goal_lengths and failed_goal_lengths else 0
        }
    
    def generate_detailed_analysis(self, app_data: Dict[str, AppData]) -> Dict:
        """Generate detailed analysis with insights and recommendations"""
        analysis = {
            'insights': [],
            'recommendations': [],
            'quality_assessment': {},
            'exploration_effectiveness': {}
        }
        
        # Generate insights
        total_trajectories = sum(len(data.trajectories) for data in app_data.values())
        successful_trajectories = sum(data.statistics['successful_trajectories'] for data in app_data.values())
        
        if total_trajectories > 0:
            overall_success_rate = successful_trajectories / total_trajectories * 100
            
            analysis['insights'].append(f"Overall exploration success rate: {overall_success_rate:.1f}%")
            
            if overall_success_rate > 80:
                analysis['insights'].append("High success rate indicates effective exploration strategy")
            elif overall_success_rate < 50:
                analysis['insights'].append("Low success rate suggests need for exploration strategy improvement")
            
            # App-specific insights
            best_app = max(app_data.items(), key=lambda x: x[1].statistics['successful_trajectories'] / x[1].statistics['total_trajectories'] if x[1].statistics['total_trajectories'] > 0 else 0)
            worst_app = min(app_data.items(), key=lambda x: x[1].statistics['successful_trajectories'] / x[1].statistics['total_trajectories'] if x[1].statistics['total_trajectories'] > 0 else 1)
            
            analysis['insights'].append(f"Best performing app: {best_app[0]} ({best_app[1].statistics['successful_trajectories']}/{best_app[1].statistics['total_trajectories']} successful)")
            analysis['insights'].append(f"Lowest performing app: {worst_app[0]} ({worst_app[1].statistics['successful_trajectories']}/{worst_app[1].statistics['total_trajectories']} successful)")
        
        # Generate recommendations
        analysis['recommendations'] = self.generate_recommendations(app_data)
        
        # Quality assessment
        analysis['quality_assessment'] = self.assess_data_quality(app_data)
        
        # Exploration effectiveness
        analysis['exploration_effectiveness'] = self.assess_exploration_effectiveness(app_data)
        
        return analysis
    
    def generate_recommendations(self, app_data: Dict[str, AppData]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        total_trajectories = sum(len(data.trajectories) for data in app_data.values())
        if total_trajectories == 0:
            return ["No trajectories found - consider running exploration first"]
        
        successful_trajectories = sum(data.statistics['successful_trajectories'] for data in app_data.values())
        success_rate = successful_trajectories / total_trajectories * 100
        
        if success_rate < 30:
            recommendations.append("Consider adjusting exploration parameters to improve success rate")
            recommendations.append("Review failed trajectories to identify common failure patterns")
        
        if success_rate > 90:
            recommendations.append("High success rate achieved - consider increasing exploration depth or complexity")
        
        # Depth-based recommendations
        max_depth = max((max(data.depth_structure.keys()) if data.depth_structure else 0) for data in app_data.values()) if app_data else 0
        if max_depth < 3:
            recommendations.append("Consider increasing maximum exploration depth for more comprehensive coverage")
        
        # App-specific recommendations
        low_coverage_apps = [name for name, data in app_data.items() if len(data.trajectories) < 10]
        if low_coverage_apps:
            recommendations.append(f"Increase exploration coverage for apps: {', '.join(low_coverage_apps)}")
        
        return recommendations
    
    def assess_data_quality(self, app_data: Dict[str, AppData]) -> Dict:
        """Assess the quality of exploration data"""
        quality_metrics = {
            'completeness': {},
            'consistency': {},
            'coverage': {}
        }
        
        total_apps = len(app_data)
        apps_with_trajectories = sum(1 for data in app_data.values() if data.trajectories)
        apps_with_multiple_depths = sum(1 for data in app_data.values() if len(data.depth_structure) > 1)
        
        quality_metrics['completeness'] = {
            'apps_with_data': apps_with_trajectories,
            'data_coverage_percentage': apps_with_trajectories / total_apps * 100 if total_apps > 0 else 0,
            'apps_with_depth_exploration': apps_with_multiple_depths,
            'depth_exploration_percentage': apps_with_multiple_depths / total_apps * 100 if total_apps > 0 else 0
        }
        
        # Check for consistent data structure
        apps_with_screenshots = sum(1 for data in app_data.values() 
                                   if any(any(getattr(step, f'{screen_type}_screenshot', None) is not None 
                                            for screen_type in ['raw', 'before', 'after', 'before_with_som', 'after_with_som'])
                                         for trajectory in data.trajectories 
                                         for step in trajectory.steps))
        
        quality_metrics['consistency'] = {
            'apps_with_screenshots': apps_with_screenshots,
            'screenshot_availability_percentage': apps_with_screenshots / total_apps * 100 if total_apps > 0 else 0
        }
        
        return quality_metrics
    
    def assess_exploration_effectiveness(self, app_data: Dict[str, AppData]) -> Dict:
        """Assess the effectiveness of exploration strategy"""
        effectiveness = {
            'breadth_metrics': {},
            'depth_metrics': {},
            'efficiency_metrics': {}
        }
        
        all_goals = []
        for data in app_data.values():
            all_goals.extend([t.goal for t in data.trajectories])
        
        unique_goals = set(all_goals)
        
        effectiveness['breadth_metrics'] = {
            'total_unique_goals': len(unique_goals),
            'average_goals_per_app': len(unique_goals) / len(app_data) if app_data else 0,
            'goal_diversity_score': len(unique_goals) / len(all_goals) if all_goals else 0
        }
        
        max_depth_per_app = [max(data.depth_structure.keys()) if data.depth_structure else 0 for data in app_data.values()]
        
        effectiveness['depth_metrics'] = {
            'max_depth_achieved': max(max_depth_per_app) if max_depth_per_app else 0,
            'average_max_depth': np.mean(max_depth_per_app) if max_depth_per_app else 0,
            'apps_reaching_depth_3_plus': sum(1 for depth in max_depth_per_app if depth >= 3)
        }
        
        total_steps = sum(len(trajectory.steps) for data in app_data.values() for trajectory in data.trajectories)
        total_trajectories = sum(len(data.trajectories) for data in app_data.values())
        
        effectiveness['efficiency_metrics'] = {
            'average_steps_per_trajectory': total_steps / total_trajectories if total_trajectories > 0 else 0,
            'exploration_efficiency_score': len(unique_goals) / total_steps if total_steps > 0 else 0
        }
        
        return effectiveness
    
    def generate_csv_summaries(self, app_data: Dict[str, AppData], output_dir: Path):
        """Generate CSV summaries for easy analysis"""
        import csv
        
        # App-level summary
        with open(output_dir / "apps_summary.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Package Name', 'App Name', 'Version', 'Total Trajectories', 'Successful Trajectories', 
                           'Success Rate (%)', 'Total Steps', 'Average Steps', 'Unique Goals', 'Max Depth'])
            
            for package_name, data in app_data.items():
                stats = data.statistics
                success_rate = (stats['successful_trajectories'] / stats['total_trajectories'] * 100) if stats['total_trajectories'] > 0 else 0
                max_depth = max(data.depth_structure.keys()) if data.depth_structure else 0
                
                writer.writerow([
                    package_name,
                    data.app_info.get('app_name', ''),
                    data.app_info.get('app_version_name', ''),
                    stats['total_trajectories'],
                    stats['successful_trajectories'],
                    f"{success_rate:.1f}",
                    stats['total_steps'],
                    f"{stats['average_steps_per_trajectory']:.1f}",
                    stats['unique_goals'],
                    max_depth
                ])
        
        # Trajectory-level summary
        with open(output_dir / "trajectories_summary.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['App Package', 'App Name', 'Trajectory ID', 'Depth', 'Goal', 'Steps Count', 
                           'Success', 'Start Time'])
            
            for package_name, data in app_data.items():
                for trajectory in data.trajectories:
                    writer.writerow([
                        package_name,
                        data.app_info.get('app_name', ''),
                        trajectory.trajectory_id,
                        trajectory.depth,
                        trajectory.goal,
                        len(trajectory.steps),
                        trajectory.success,
                        trajectory.start_time.isoformat() if trajectory.start_time else ''
                    ])
        
        self.log("CSV summaries generated")
    
    def get_most_common_items(self, items: List[str], top_n: int = 5) -> List[Tuple[str, int]]:
        """Get most common items from a list"""
        return Counter(items).most_common(top_n)
    
    def interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient"""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return "negligible"
        elif abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.5:
            return "moderate"
        elif abs_corr < 0.7:
            return "strong"
        else:
            return "very strong"
