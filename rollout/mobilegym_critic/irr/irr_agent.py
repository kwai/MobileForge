#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Information Retention Rate (IRR) Agent
For calculating the information retention rate of agents during task execution
"""

import os
import sys
import json
import pandas as pd
import argparse
from typing import Dict, List, Tuple, Optional

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mobilegym_critic.utils.vivo.vivo_api import inference_chat_gemini_wo_image
from mobilegym_critic.utils.common import parse_json_from_response


def safe_parse_json_from_response(response: str) -> dict:
    """
    Safely parse JSON response, handling format errors
    """
    try:
        return parse_json_from_response(response)
    except Exception as e:
        # Try alternative parsing methods
        try:
            # Remove possible markdown markers
            clean_response = response.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            return json.loads(clean_response)
        except:
            print(f"    JSON parsing failed: {str(e)[:100]}...")
            return None


def get_irr_analysis_prompt(task_description: str, failure_reason: str, step_descriptions: List[Dict]) -> Tuple[str, str]:
    """
    Generate IRR analysis system and user prompts
    """
    system_prompt = """You are an expert in analyzing agent information retention capabilities. Your task is to precisely calculate the Information Retention Rate (IRR) of an agent based on the given task description, failure reason, and execution step descriptions.

## IRR Definition and Calculation Principles

IRR = (Number of correctly recalled and used information units / Total number of information units required by the task) × 100%

**Information Unit**: The smallest piece of information that the agent is required to remember and use in a task. Examples include:
- Product prices, ratings, specifications
- Contact phone numbers, email addresses
- Meeting dates, times, locations
- Order numbers, verification codes
- Product models, brands, features
- Addresses, rent prices, areas, etc.

## Detailed Calculation Rules

### 1. Task Success
If the task is ultimately successful, it means all required information has been correctly processed.
**IRR = 100%**

### 2. Partial Failure with Explicit Output
Applies to tasks that require explicit output of remembered information (e.g., taking notes, sending messages).
If the task fails but some information units are correctly output, IRR is calculated based on the proportion.
**Example**: Task requires remembering 9 pieces of information, agent correctly outputs 7.
**IRR = 7/9 = 77.8%**

### 3. Failure in Implicit Memory Tasks
Applies to tasks requiring agents to use memory for internal calculations or decisions, ultimately executing only one action.
In such cases, we cannot externally trace the specific correctness of the memory chain.
**For objectivity and consistency, if the final decision behavior is incorrect, IRR = 0%**

**Example**: "Search 6 courses on Coursera, remember each course's rating, review count, and language count, calculate a 'popularity score', and navigate to the highest-scoring course page."
- If the agent navigates to the wrong course page, we cannot determine whether it misremembered ratings, review counts, or made calculation errors.
- Since we cannot objectively assign "partial credit", IRR = 0%.

### 4. Early-Stage Failure
If the agent fails early in the task (e.g., unable to find the information source page), resulting in no information units being processed.
**IRR = 0%**

## Calculation Examples

### Example 1: E-commerce Comparison Task
**Task**: Compare three phones (A, B, C) for **price**, **memory**, and **rating** in an e-commerce app, and record this information in a notes app.

**Required Information Units**:
- Phone A: price, memory, rating (3 units)
- Phone B: price, memory, rating (3 units)  
- Phone C: price, memory, rating (3 units)
- **Total: 9 information units**

**Scenarios**:
- **Complete Success**: Agent finds all 9 pieces of information and records them accurately. **IRR = 9/9 = 100%**
- **Partial Memory Failure**: Agent finds all information but makes recording errors: Phone A's price is wrong, Phone C's rating is missing. Only 7 information units are correct in the final notes. **IRR = 7/9 = 77.8%**
- **Early Failure**: Agent fails to search for the three phones in the e-commerce app, task terminates early. **IRR = 0/9 = 0%**

### Example 2: Course Selection Task
**Task**: Search for programming courses, remember course details (instructor, duration, price), and enroll in the most suitable one.

**Analysis Approach**:
1. **Identify Information Units**: How many pieces of information need to be remembered?
2. **Trace Agent Behavior**: What information did the agent actually collect and use?
3. **Determine IRR Type**: Is this explicit output or implicit decision-making?
4. **Calculate Precisely**: Based on correct vs. total information units.

## Analysis Requirements

You must:
1. **Carefully analyze** the task description to identify ALL information units that need to be remembered
2. **Analyze the failure reason** to determine if it involves information memory issues
3. **Examine execution steps** to determine what information the agent actually collected and used
4. **Calculate accurate IRR** based on the specific scenario type
5. **Provide detailed reasoning** explaining your calculation process

## Output Format

Your response must be in JSON format containing:
- total_information_units: Total number of information units required (integer)
- correctly_used_units: Number of correctly used information units (integer)
- irr_percentage: IRR percentage (0-100, integer)
- analysis_reason: Detailed analysis reasoning (string)

## Important Notes

- Be precise in counting information units - each specific piece of data counts as one unit
- For implicit memory tasks with wrong final decisions, always assign IRR = 0%
- For explicit output tasks, count the actual correct information in the output
- Consider the task type carefully when applying calculation rules
- Provide clear, objective reasoning for your IRR calculation"""

    # Format step descriptions
    formatted_steps = []
    for i, step_data in enumerate(step_descriptions):
        if isinstance(step_data, dict):
            action = step_data.get("action_description", "N/A")
            ui = step_data.get("ui_description", "N/A") 
            formatted_steps.append(f"Step {i+1}:\n  Action: {action}\n  UI State: {ui}")
        else:
            formatted_steps.append(f"Step {i+1}: {step_data}")
    
    steps_text = "\n".join(formatted_steps)

    user_prompt = f"""Please analyze the Information Retention Rate (IRR) for the following task:

## Task Description
{task_description}

## Failure Reason
{failure_reason}

## Execution Step Descriptions
{steps_text}

Based on the above information and following the IRR calculation principles, please provide a precise analysis:

1. **Identify Information Units**: How many information units does this task require the agent to remember?
2. **Trace Agent Performance**: How many information units did the agent actually collect and use correctly?
3. **Determine Task Type**: Is this an explicit output task or implicit decision-making task?
4. **Calculate IRR**: Apply the appropriate calculation rule based on the task type and agent performance.
5. **Provide Detailed Reasoning**: Explain your analysis process and justify the IRR calculation.

## Analysis Guidelines:
- Count each specific piece of required information as one unit (e.g., price=1 unit, rating=1 unit, model=1 unit)
- For explicit output tasks: Count correct information in the final output
- For implicit decision tasks with wrong outcomes: IRR = 0%
- For early failures before information collection: IRR = 0%
- Be objective and consistent in your evaluation

Output in JSON format:
```json
{{
  "total_information_units": <integer>,
  "correctly_used_units": <integer>, 
  "irr_percentage": <0-100 integer>,
  "analysis_reason": "<detailed analysis reasoning>"
}}
```"""

    return system_prompt, user_prompt


def calculate_irr_for_task(
    task_description: str,
    failure_reason: str, 
    step_descriptions: List[Dict],
    model: str = "gemini-2.5-pro"
) -> Optional[Dict]:
    """
    Calculate IRR for a single task
    
    Args:
        task_description: Description of the task
        failure_reason: Reason why the task failed
        step_descriptions: List of step descriptions from agent execution
        model: LLM model to use for analysis
        
    Returns:
        Dictionary containing IRR analysis results or None if failed
    """
    try:
        system_prompt, user_prompt = get_irr_analysis_prompt(
            task_description, failure_reason, step_descriptions
        )
        
        response = inference_chat_gemini_wo_image(
            system_prompt, user_prompt, model=model
        )
        
        if isinstance(response, dict):
            response_str = response["content"]
        else:
            response_str = response
            
        irr_result = parse_json_from_response(response_str)
        return irr_result
        
    except Exception as e:
        print(f"Error calculating IRR: {e}")
        return None


def get_agent_name_from_results_csv(csv_path: str) -> str:
    """
    Extract agent name from results.csv file column names
    
    Args:
        csv_path: Path to the results.csv file
        
    Returns:
        Extracted agent name or "unknown_agent" if not found
    """
    df = pd.read_csv(csv_path, nrows=1)
    columns = df.columns.tolist()
    
    # Find columns containing _evaluation and extract agent name
    for col in columns:
        if '_evaluation' in col and 'attempt_1' in col:
            # Example: M3A_vivo_gemini_direct_with_action_attempt_1_evaluation
            # Extract: M3A_vivo_gemini
            parts = col.split('_')
            # Find the position of direct_with_action, agent name is before it
            if 'direct' in parts and 'with' in parts and 'action' in parts:
                direct_idx = parts.index('direct')
                agent_name = '_'.join(parts[:direct_idx])
                return agent_name
    
    return "unknown_agent"


def process_single_agent_directory(agent_dir: str) -> str:
    """
    Process a single agent directory and generate a new CSV file with IRR data
    
    Args:
        agent_dir: Path to the agent directory
        
    Returns:
        Path to the generated CSV file with IRR data
    """
    results_csv_path = os.path.join(agent_dir, "results.csv")
    if not os.path.exists(results_csv_path):
        print(f"Warning: {results_csv_path} not found")
        return ""
    
    # Read existing results
    df = pd.read_csv(results_csv_path)
    agent_name = get_agent_name_from_results_csv(results_csv_path)
    
    # Read task definitions
    task_df = pd.read_csv("data/memgui-v25071601.csv")
    memory_tasks = task_df[task_df['requires_ui_memory'] == 'Y']['task_identifier'].tolist()
    
    print(f"Processing agent: {agent_name}")
    print(f"Found {len(memory_tasks)} tasks requiring UI memory")
    
    # Add IRR-related columns
    irr_columns = []
    for attempt in [1, 2, 3]:
        irr_columns.extend([
            f"{agent_name}_attempt_{attempt}_irr_total_units",
            f"{agent_name}_attempt_{attempt}_irr_correct_units", 
            f"{agent_name}_attempt_{attempt}_irr_percentage",
            f"{agent_name}_attempt_{attempt}_irr_reason"
        ])
    
    # Initialize new columns
    for col in irr_columns:
        df[col] = ""
    
    # Process each task
    for idx, row in df.iterrows():
        task_id = row['task_identifier']
        
        # Only process tasks that require UI memory
        if task_id not in memory_tasks:
            continue
            
        print(f"Processing task: {task_id}")
        
        for attempt in [1, 2, 3]:
            eval_col = f"{agent_name}_direct_with_action_attempt_{attempt}_evaluation"
            details_col = f"{agent_name}_direct_with_action_attempt_{attempt}_details"
            method_col = f"{agent_name}_direct_with_action_attempt_{attempt}_evaluation_method"
            
            if eval_col not in df.columns:
                continue
                
            evaluation = row[eval_col]
            details = row[details_col] if details_col in df.columns else ""
            method = row[method_col] if method_col in df.columns else ""
            
            # Skip empty values
            if pd.isna(evaluation) or evaluation == "":
                continue
                
            irr_total_col = f"{agent_name}_attempt_{attempt}_irr_total_units"
            irr_correct_col = f"{agent_name}_attempt_{attempt}_irr_correct_units"
            irr_percentage_col = f"{agent_name}_attempt_{attempt}_irr_percentage"
            irr_reason_col = f"{agent_name}_attempt_{attempt}_irr_reason"
            
            # Determine IRR based on evaluation_method
            if method == "finish_signal_check":
                # finish_signal=0, directly mark IRR as 0
                df.loc[idx, irr_total_col] = "N/A"
                df.loc[idx, irr_correct_col] = 0
                df.loc[idx, irr_percentage_col] = 0
                df.loc[idx, irr_reason_col] = "Task failed due to finish_signal=0"
                
            elif method == "pre_evaluation":
                # pre_evaluation success, IRR is 100%
                df.loc[idx, irr_total_col] = "N/A" 
                df.loc[idx, irr_correct_col] = "N/A"
                df.loc[idx, irr_percentage_col] = 100
                df.loc[idx, irr_reason_col] = "Task succeeded in pre-evaluation"
                
            elif evaluation == "S":
                # Other methods but task succeeded, IRR is 100%
                df.loc[idx, irr_total_col] = "N/A"
                df.loc[idx, irr_correct_col] = "N/A" 
                df.loc[idx, irr_percentage_col] = 100
                df.loc[idx, irr_reason_col] = "Task succeeded"
                
            elif evaluation == "F":
                # Other methods but task failed, need detailed analysis
                print(f"  Need detailed IRR analysis for {task_id} attempt {attempt}")
                
                # Read prompt_logs.json to get step descriptions
                prompt_logs_path = os.path.join(
                    agent_dir, task_id, agent_name, f"attempt_{attempt}", "prompt_logs.json"
                )
                
                if os.path.exists(prompt_logs_path):
                    try:
                        with open(prompt_logs_path, 'r', encoding='utf-8') as f:
                            logs = json.load(f)
                        
                        # Extract step descriptions
                        step_descriptions = []
                        for log in logs:
                            if 'step_' in log.get('stage', '') and '_description' in log.get('stage', ''):
                                try:
                                    desc = safe_parse_json_from_response(log['llm_response'])
                                    if desc:
                                        step_descriptions.append(desc)
                                except Exception as e:
                                    print(f"      Warning: Failed to parse step description: {e}")
                                    continue
                        
                        if step_descriptions:
                            # Call IRR agent for analysis
                            task_desc = row['task_description']
                            irr_result = calculate_irr_for_task(task_desc, str(details), step_descriptions)
                            
                            if irr_result:
                                df.loc[idx, irr_total_col] = irr_result.get('total_information_units', 0)
                                df.loc[idx, irr_correct_col] = irr_result.get('correctly_used_units', 0)
                                df.loc[idx, irr_percentage_col] = irr_result.get('irr_percentage', 0)
                                df.loc[idx, irr_reason_col] = irr_result.get('analysis_reason', 'IRR analysis completed')
                            else:
                                df.loc[idx, irr_total_col] = "Error"
                                df.loc[idx, irr_correct_col] = "Error"
                                df.loc[idx, irr_percentage_col] = "Error" 
                                df.loc[idx, irr_reason_col] = "Failed to analyze IRR"
                        else:
                            df.loc[idx, irr_total_col] = "No data"
                            df.loc[idx, irr_correct_col] = "No data"
                            df.loc[idx, irr_percentage_col] = "No data"
                            df.loc[idx, irr_reason_col] = "No step descriptions found"
                            
                    except Exception as e:
                        print(f"    Error processing {prompt_logs_path}: {e}")
                        df.loc[idx, irr_total_col] = "Error"
                        df.loc[idx, irr_correct_col] = "Error" 
                        df.loc[idx, irr_percentage_col] = "Error"
                        df.loc[idx, irr_reason_col] = f"Processing error: {str(e)}"
                else:
                    df.loc[idx, irr_total_col] = "No logs"
                    df.loc[idx, irr_correct_col] = "No logs"
                    df.loc[idx, irr_percentage_col] = "No logs"
                    df.loc[idx, irr_reason_col] = "Prompt logs not found"
    
    # Save new CSV file
    output_path = os.path.join(agent_dir, "results_add_irr.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved results with IRR to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Calculate IRR for MemGUI evaluation results")
    parser.add_argument("--agent_dir", type=str, help="Path to specific agent directory")
    parser.add_argument("--all_agents", action="store_true", help="Process all agents in results/00_baselines")
    
    args = parser.parse_args()
    
    if args.all_agents:
        # Process all 7 agents
        base_dir = "results/00_baselines"
        agent_dirs = [
            "250719_t3a_gemini-2.5-flash",
            "250719_T3A_gemini-2.5-pro", 
            "250721_M3A_gemini-2.5-pro",
            "250723_mobileagente_gemini-2.5-pro",
            "250819_tars_1.5_7b",
            "250831_gui_owl_7b",
            "25082701_agent_s2_gemini-2.5-pro"
        ]
        
        for agent_dir_name in agent_dirs:
            agent_path = os.path.join(base_dir, agent_dir_name)
            if os.path.exists(agent_path):
                print(f"\n=== Processing {agent_dir_name} ===")
                try:
                    process_single_agent_directory(agent_path)
                except Exception as e:
                    print(f"Error processing {agent_dir_name}: {e}")
            else:
                print(f"Warning: {agent_path} not found")
                
    elif args.agent_dir:
        process_single_agent_directory(args.agent_dir)
    else:
        print("Please specify --agent_dir or --all_agents")


if __name__ == "__main__":
    main()
