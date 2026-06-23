#!/usr/bin/env python3
"""
Convert generated tasks from JSON format to rollout CSV format.

This script processes all JSON task files in the generated_tasks directory
and converts them to the rollout format CSV as specified in rollout-format-example.csv.
"""

import os
import json
import csv
from pathlib import Path
import random
import glob
from typing import Dict, List, Tuple


def generate_task_identifier() -> str:
    """Generate a unique 16-character task identifier with random lowercase letters and numbers."""
    import string
    # Generate exactly 16 characters of lowercase letters and numbers
    chars = string.ascii_lowercase + string.digits
    identifier = ''.join(random.choice(chars) for _ in range(16))
    # Ensure exactly 16 characters
    assert len(identifier) == 16, f"Generated identifier length is {len(identifier)}, expected 16"
    return identifier


def extract_app_name_from_package(package_name: str) -> str:
    """Extract app name from package name."""
    app_mapping = {
        "com.google.android.contacts": "Contacts",
        "com.google.android.deskclock": "Clock", 
        "com.android.camera2": "Camera",
        "com.arduia.expense": "Pro Expense",
        "net.gsantner.markor": "Markor",
        "net.osmand": "OsmAnd",
        "com.simplemobiletools.calendar.pro": "Simple Calendar Pro",
        "com.simplemobiletools.smsmessenger": "Simple SMS Messenger",
        "com.android.settings": "System"
    }
    return app_mapping.get(package_name, package_name.split('.')[-1].title())


def determine_task_difficulty_number(difficulty_level: str) -> int:
    """Convert difficulty level string to number."""
    difficulty_mapping = {
        "low_level": 1,
        "medium_level": 2, 
        "high_level": 3
    }
    return difficulty_mapping.get(difficulty_level, 1)


def determine_golden_steps(difficulty: int) -> int:
    """Determine golden_steps based on difficulty level."""
    golden_steps_mapping = {
        1: 10,  # low_level
        2: 20,  # medium_level
        3: 30   # high_level
    }
    return golden_steps_mapping.get(difficulty, 10)


def process_json_task(json_file_path: str) -> Tuple[str, str, List[str], int, int]:
    """
    Process a single JSON task file and extract required information.
    
    Returns:
        Tuple of (task_identifier, task_description, task_app, golden_steps, task_difficulty)
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        task_data = json.load(f)
    
    # Generate unique task identifier
    task_identifier = generate_task_identifier()
    
    # Extract task description
    task_description = task_data.get('new_task_description', task_data.get('original_task_goal', ''))
    
    # Extract and convert app information
    package_name = task_data.get('package_name', '')
    app_name = extract_app_name_from_package(package_name)
    task_app = [app_name]
    
    # Extract difficulty level and convert to number
    difficulty_level = task_data.get('difficulty_level', 'low_level')
    task_difficulty = determine_task_difficulty_number(difficulty_level)
    
    # Determine golden steps based on difficulty
    golden_steps = determine_golden_steps(task_difficulty)
    
    return task_identifier, task_description, task_app, golden_steps, task_difficulty


def find_all_json_task_files(base_dir: str) -> List[str]:
    """Find all JSON task files in the directory structure."""
    pattern = os.path.join(base_dir, "**", "*.json")
    json_files = glob.glob(pattern, recursive=True)
    
    # Filter out non-task JSON files (exclude files that don't contain task data)
    task_files = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if this is a task file by looking for expected keys
                if ('new_task_description' in data or 'original_task_goal' in data) and 'package_name' in data:
                    task_files.append(json_file)
        except (json.JSONDecodeError, Exception):
            # Skip files that can't be parsed or don't match expected format
            continue
    
    return task_files


def convert_tasks_to_csv(input_dir: str, output_file: str):
    """
    Convert all JSON task files to CSV format.
    
    Args:
        input_dir: Directory containing the generated tasks
        output_file: Output CSV file path
    """
    print(f"Searching for task files in: {input_dir}")
    
    # Find all JSON task files
    json_files = find_all_json_task_files(input_dir)
    print(f"Found {len(json_files)} task files")
    
    if not json_files:
        print("No task files found!")
        return
    
    # Process each file and collect data
    csv_rows = []
    processed_count = 0
    error_count = 0
    
    for json_file in json_files:
        try:
            task_identifier, task_description, task_app, golden_steps, task_difficulty = process_json_task(json_file)
            
            csv_rows.append({
                'task_identifier': task_identifier,  # Text format without quotes
                'task_description': task_description,
                'task_app': str(task_app),  # Convert list to string representation
                'golden_steps': golden_steps,
                'task_difficulty': task_difficulty
            })
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} files...")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            error_count += 1
            continue
    
    # Write to CSV
    if csv_rows:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['task_identifier', 'task_description', 'task_app', 'golden_steps', 'task_difficulty']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data rows
            for row in csv_rows:
                writer.writerow(row)
        
        print(f"\nConversion completed!")
        print(f"Successfully processed: {processed_count} tasks")
        print(f"Errors encountered: {error_count} files")
        print(f"Output written to: {output_file}")
        
        # Print summary statistics
        difficulty_counts = {}
        for row in csv_rows:
            diff = row['task_difficulty']
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        print(f"\nTask distribution by difficulty:")
        for difficulty, count in sorted(difficulty_counts.items()):
            difficulty_name = {1: "low_level", 2: "medium_level", 3: "high_level"}.get(difficulty, f"level_{difficulty}")
            golden_steps = determine_golden_steps(difficulty)
            print(f"  Difficulty {difficulty} ({difficulty_name}): {count} tasks (golden_steps: {golden_steps})")
    else:
        print("No tasks were successfully processed!")


def main():
    """Main function to run the conversion."""
    # Set up paths
    base_dir = str(Path(__file__).resolve().parents[2])
    input_dir = os.path.join(base_dir, "generated_tasks", "generated_tasks-251001")
    output_file = os.path.join(base_dir, "data_process", "explore-instruction-to-rollout-format", "converted_tasks.csv")
    
    print("Task Converter - JSON to Rollout CSV Format")
    print("=" * 50)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Run conversion
    convert_tasks_to_csv(input_dir, output_file)


if __name__ == "__main__":
    main()
