"""
统一任务处理器模块

将任务评估和任务生成合并为一次LLM调用，提高效率和一致性
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import base64
from io import BytesIO

# Add the parent directory to the path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.utils import gpt4v_call


class UnifiedTaskProcessor:
    """统一任务处理器，将评估和生成合并为一次LLM调用"""
    
    def __init__(self):
        """初始化统一任务处理器"""
        self.token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "call_count": 0
        }
        self.last_llm_debug = None
    
    def process_task(self, app_name: str, original_goal: str, 
                    visualized_screenshots: List[Dict[str, Any]], 
                    fewshot_examples: List[Dict[str, Any]], 
                    task_principles: List[str],
                    existing_tasks: List[str] = None) -> Dict[str, Any]:
        """
        统一处理任务评估和生成
        
        Args:
            app_name: 应用名称
            original_goal: 原始任务目标
            visualized_screenshots: 可视化截图列表
            fewshot_examples: Few-shot示例
            task_principles: 任务生成原则
            
        Returns:
            包含评估结果和生成任务的字典
        """
        print(f"    Starting unified task processing: {original_goal}")
        
        # 构建统一的prompt
        prompt = self._build_unified_prompt(
            app_name, original_goal, fewshot_examples, task_principles, existing_tasks
        )
        
        # 准备图像数据
        image_data_list = self._prepare_image_data(visualized_screenshots)
        
        # 调用LLM进行统一处理
        try:
            response_text = gpt4v_call(
                prompt=prompt,
                images=image_data_list,
                max_tokens=18000,
                temperature=0.1
            )
            
            if response_text:
                print(f"    LLM response received, length: {len(response_text)}")
                self._update_token_usage(response_text)
                
                # 解析响应
                result = self._parse_llm_response(response_text)
                
                # 筛选包含app_name的任务
                generated_tasks = result.get('generated_tasks', [])
                if generated_tasks and app_name:
                    valid_tasks, filtered_count = self._filter_tasks_by_app_name(generated_tasks, app_name)
                    result['generated_tasks'] = valid_tasks
                    result['filtered_tasks_count'] = filtered_count
                    
                    if filtered_count > 0:
                        print(f"    Filtered {filtered_count} tasks not containing '{app_name}'")
                    print(f"    Valid tasks after app_name filtering: {len(valid_tasks)}")
                
                # 筛选掉包含中文字符的任务（包括乱码）
                remaining_tasks = result.get('generated_tasks', [])
                if remaining_tasks:
                    valid_tasks, chinese_filtered_count = self._filter_tasks_with_chinese(remaining_tasks)
                    result['generated_tasks'] = valid_tasks
                    result['chinese_filtered_tasks_count'] = chinese_filtered_count
                    
                    if chinese_filtered_count > 0:
                        print(f"    Filtered {chinese_filtered_count} tasks containing Chinese characters")
                    print(f"    Valid tasks after all filtering: {len(valid_tasks)}")
                
                # 保存调试信息
                self.last_llm_debug = {
                    "input_prompt": prompt,
                    "prompt_length": len(prompt),
                    "image_count": len(image_data_list),
                    "response_text": response_text,
                    "response_length": len(response_text),
                    "app_name": app_name,
                    "original_goal": original_goal
                }
                
                return result
            else:
                print("    LLM call failed, using fallback")
                return self._generate_fallback_result(app_name, original_goal)
                
        except Exception as e:
            print(f"    LLM call failed with error: {e}")
            return self._generate_fallback_result(app_name, original_goal)
    
    def _build_unified_prompt(self, app_name: str, original_goal: str, 
                            fewshot_examples: List[Dict[str, Any]], 
                            task_principles: List[str],
                            existing_tasks: List[str] = None) -> str:
        """构建统一的评估和生成prompt"""
        
        # 分离不同类型的示例
        step_examples = []  # 带有action descriptions的示例，用于理解步骤执行
        style_examples = []  # AndroidWorld示例，用于任务描述风格参考
        
        for example in fewshot_examples:
            source = example.get('source', 'General')
            
            if source == 'AndroidWorld':
                style_examples.append(example)
            else:
                step_examples.append(example)
        
        print(f"    DEBUG: Found {len(style_examples)} AndroidWorld examples, {len(step_examples)} step examples")
        
        # 构建步骤理解示例（带有详细action descriptions）
        step_examples_text = ""
        if step_examples:
            step_examples_text = "\n## Step Execution Understanding Examples\n"
            step_examples_text += "The following examples show how complex tasks are broken down into specific executable steps. Use these to understand the granularity and type of steps needed for task execution:\n"
            
            for i, example in enumerate(step_examples[:3], 1):  # 限制为3个
                instruction = example.get('instruction', '')
                
                if 'action_descriptions' in example:
                    action_descriptions = example.get('action_descriptions', [])
                    step_examples_text += f"""
Step Example {i}:
Task: {instruction}
Execution Steps ({len(action_descriptions)} total):
{chr(10).join([f"  {j+1}. {desc}" for j, desc in enumerate(action_descriptions)])}
"""
                elif 'steps' in example:
                    steps = example.get('steps', [])
                    action_descriptions = [step.get('action_description', '') for step in steps if step.get('action_description')]
                    step_examples_text += f"""
Step Example {i}:
Task: {instruction}
Execution Steps ({len(action_descriptions)} total):
{chr(10).join([f"  {j+1}. {desc}" for j, desc in enumerate(action_descriptions)])}
"""
        
        # 构建风格参考示例（AndroidWorld格式）
        style_examples_text = ""
        if style_examples:
            style_examples_text = f"\n## AndroidWorld Task Style References for {app_name}\n"
            style_examples_text += "The following are actual AndroidWorld tasks for this app. Use these as style references to generate tasks with similar description patterns, clarity, and specificity:\n"
            
            for i, example in enumerate(style_examples, 1):  # 使用所有AndroidWorld示例，不设置上限
                instruction = example.get('instruction', '')
                task_name = example.get('task_name', '')
                main_app = example.get('main_app', '')
                
                style_examples_text += f"""
Style Reference {i}:
Task Description: {instruction}
{f"Task ID: {task_name}" if task_name else ""}
{f"App: {main_app}" if main_app else ""}
"""
        
        # 合并示例文本
        fewshot_text = step_examples_text + style_examples_text
        
        # 构建任务生成原则
        principles_text = "\n".join([f"{i+1}. {principle}" for i, principle in enumerate(task_principles)])
        
        # 构建已生成任务列表
        existing_tasks_text = ""
        if existing_tasks and len(existing_tasks) > 0:
            existing_tasks_text = f"""
## Already Generated Tasks for {app_name}
To avoid redundancy, here are ALL tasks already generated for this app:
{chr(10).join([f"- {task}" for task in existing_tasks])}

IMPORTANT: Do not generate tasks that are too similar to the above. Focus on exploring NEW core functionalities or significantly different parameter variations.
"""
        
        prompt = f"""You are an expert Curriculum Generator - a teacher designing comprehensive learning tasks for GUI agents. Your core purpose is to create a complete curriculum covering all app functionalities with progressive difficulty levels to systematically teach GUI agents how to use the target app.

As a Curriculum Generator, your job is to:
1. EVALUATE the original task for reasonableness and completion  
2. GENERATE new diverse curriculum tasks that comprehensively cover the app's functionality with SPECIAL EMPHASIS on AndroidWorld examination points

## App Information
App Name: {app_name}
Original Task Goal: {original_goal}

## CRITICAL RULE - Application Name Requirement
🚨 **MANDATORY**: Every generated task instruction MUST explicitly contain the exact application name "{app_name}".
- Tasks without "{app_name}" in the instruction will be automatically REJECTED.
- Do NOT use pronouns like "the app" or "this application" - use "{app_name}" explicitly.

## Few-shot Examples
{fewshot_text}

## Task Generation Principles
{principles_text}
{existing_tasks_text}

## Curriculum Design Instructions

### Step 1: Task Evaluation
Based on the app name, original task goal, and the sequence of visualized screenshots:

1. **Reasonableness Assessment**: 
   - Is this a reasonable task that a user might actually want to perform in this app?
   - Are the task requirements clear and achievable?
   - Does the task make sense in the context of the app's functionality?

2. **Step-by-Step Quality Analysis**:
   - Analyze EACH visible step in the screenshot sequence for reasonableness
   - A "reasonable step" means the action logically progresses the task toward completion
   - An "unreasonable step" means the action is unnecessary, wrong, counterproductive, or gets the agent stuck
   - Examples of unreasonable steps: clicking wrong buttons, getting stuck in loops, redundant actions, going backwards unnecessarily
   - Examples of reasonable steps: directly progressing toward goal, correcting previous mistakes, necessary navigation
   - Even failed trajectories may contain reasonable steps that teach good patterns
   - Even successful trajectories may contain some unreasonable detours or inefficiencies
   - Calculate trajectory_quality_score as (reasonable_steps / total_steps)
   - Provide step-by-step analysis for learning purposes

3. **Overall Completion Assessment**:
   - Based on the sequence of screenshots, did the agent correctly complete the stated task?
   - Are all the required steps performed correctly?
   - Did the agent reach the intended goal state?
   - Consider both the final outcome AND the quality of the step sequence

### Step 2: AndroidWorld-Focused Curriculum Task Generation
As a curriculum designer, generate 1-10 new learning tasks with HIGHEST PRIORITY given to AndroidWorld examination points:

**Screenshot Grounding Requirement:**
- Every generated task MUST be fully achievable using ONLY the UI elements, texts, and states that are explicitly visible in the provided screenshot sequence
- Never reference people/items/menus/accounts that do not appear in the screenshots or confirmed metadata; if an object cannot be pointed out in the current evidence, do not create a task about it
- If a desired AndroidWorld capability is unsupported by the observed UI, skip it instead of hallucinating prerequisites or unseen data
- When describing prerequisite states, explicitly confirm they are visible in the screenshots; otherwise treat the task as invalid and omit it

**CRITICAL REQUIREMENT - AndroidWorld Examination Point Coverage:**
- Your generated tasks MUST comprehensively cover all examination points found in the AndroidWorld style references
- If AndroidWorld examples show specific functionality, you MUST generate similar tasks testing the same capabilities
- Prioritize covering every type of interaction pattern shown in AndroidWorld examples
- Follow the Action Space Testing and Environment Adaptation principles defined in Task Generation Principles above

**Task Description Style Requirements:**
- Follow AndroidWorld task style references exactly for {app_name}
- **CRITICAL: Every task instruction MUST include the exact app name "{app_name}"**
- Use identical language patterns, clarity, and specificity as AndroidWorld examples
- Match AndroidWorld naming conventions (e.g., "pro expense", "simple calendar")
- Ensure task descriptions are concise, actionable, and unambiguous

**Task Difficulty Requirements (CRITICAL - Current tasks are too easy):**
- 🚨 AVOID simple single-operation tasks. Average task should require 15-25 steps.
- Prioritize complex compound tasks that combine multiple operations:
  - **Batch operations**: "Delete ALL events containing 'Meeting' in the title" instead of "Delete one event"
  - **Conditional operations**: "Delete all recipes that contain 'chicken' as an ingredient" instead of "Delete one recipe"
  - **Multi-step workflows**: "Create 3 events for the next 3 Mondays with incrementing titles" instead of "Create one event"
  - **Search + action combinations**: "Find all expenses over $100 and change their category to 'Major'" instead of "Change one expense category"
  - **Multi-item management**: "Add 5 different items to a shopping list, then sort them alphabetically" instead of "Add one item"
- **Difficulty distribution target**: 20% easy (8-12 steps), 50% medium (15-25 steps), 30% hard (25-40 steps)
- Simple tasks like "open settings" or "create one event" should be RARE exceptions (less than 20% of generated tasks)
- When AndroidWorld shows a simple task pattern, ENHANCE it: add batch processing, conditions, or multi-step workflows

**Task Content Requirements:**
- Systematically cover ALL core functionalities shown in AndroidWorld examples
- Have varying step lengths (8-40 steps) representing different complexity levels - minimum 8 steps for any task
- Focus on comprehensive AndroidWorld examination point coverage rather than just parameter variations
- Ensure every AndroidWorld functionality type is represented in generated tasks
- Use screenshot content to adapt parameters while maintaining AndroidWorld examination patterns
- Prefer compound tasks that test multiple capabilities in sequence

**Important:** AndroidWorld style references are your PRIMARY guide for both task content (what to test) and task description style (how to write it). The step execution examples help estimate complexity only.

## CRITICAL - Step Count Estimation Rules (golden_steps)

🚨 **Every atomic UI operation counts as ONE step. Do NOT underestimate step counts!**

Step counting rules:
- Opening app launcher / swiping up app drawer = 1 step
- Clicking any button/icon/text field = 1 step
- Typing text into a field = 1 step (regardless of text length)
- Pressing Enter/Submit/Confirm = 1 step
- Scrolling/swiping gesture = 1 step per gesture
- Selecting from dropdown/picker = 2+ steps (open picker + navigate + select)
- Setting date = 3-5 steps (open date picker + navigate months if needed + select day + confirm)
- Setting time = 3-4 steps (open time picker + set hour + set minutes + confirm)
- Navigating to a specific screen = 2-4 steps depending on depth

**Example - "Create a calendar event titled 'Meeting' for tomorrow at 2pm":**
1. Open app (if not already open): 1 step
2. Click + / New Event button: 1 step
3. Click title field: 1 step
4. Type 'Meeting': 1 step
5. Click date field: 1 step
6. Navigate to tomorrow (if not default): 1-2 steps
7. Select/confirm date: 1 step
8. Click time field: 1 step
9. Set hour to 2: 1-2 steps
10. Set PM: 1 step
11. Confirm time: 1 step
12. Click Save/Create button: 1 step
**Total: ~12-15 steps for this SIMPLE single-item task!**

**Example - "Delete all events containing 'Meeting' in the title" (batch operation):**
- Navigate to search: 2 steps
- Search for 'Meeting': 2 steps
- For EACH matching event (assume 3-5 events): long-press + delete + confirm = 3-4 steps each
- **Total: ~15-25 steps for batch operations**

**Minimum step counts by task type:**
- Simple navigation/settings toggle: 8-12 steps
- Single item creation with details: 12-18 steps
- Single item modification: 10-15 steps
- Batch operations (multiple items): 20-35 steps
- Complex workflows (search + filter + action): 25-40 steps

## Output Format
Provide your response in the following JSON format:

```json
{{
    "evaluation": {{
        "task_reasonable": true/false,
        "task_completed": true/false,
        "reasonableness_explanation": "Detailed explanation of why the task is or isn't reasonable",
        "completion_explanation": "Detailed explanation of whether the agent completed the task correctly",
        "confidence_score": 0.0-1.0,
        "step_quality_analysis": {{
            "total_steps": 10,
            "reasonable_steps": 8,
            "unreasonable_steps": 2,
            "step_details": [
                {{
                    "step_index": 1,
                    "is_reasonable": true,
                    "explanation": "This step logically progresses toward the goal by..."
                }},
                {{
                    "step_index": 2,
                    "is_reasonable": false,
                    "explanation": "This step is unnecessary/wrong because..."
                }}
            ],
            "trajectory_quality_score": 0.8,
            "quality_summary": "Overall assessment of the trajectory quality and step progression"
        }}
    }},
    "generated_tasks": [
        {{
            "task_id": "task_1",
            "instruction": "[MUST contain '{app_name}'] Complex task for {app_name} - prefer batch/conditional/multi-step operations",
            "estimated_steps": 18,
            "core_functionality": "Main functionality being taught",
            "variation_type": "simplification/parameter_change/scenario_application/step_progression",
            "prerequisites": "Only specify non-obvious prerequisites; avoid basic assumptions"
        }}
    ]
}}
```

IMPORTANT: 
1. Every task instruction MUST contain the exact app name "{app_name}" - tasks without it will be rejected.
2. The estimated_steps should be between 8-40 steps. Minimum 8 steps for ANY task. Average should be 15-25 steps.
3. Count EVERY atomic UI interaction as one step. A simple "create one event" task already requires 12-15 steps!

Now analyze the provided app, task, and screenshots to generate your curriculum evaluation and new learning tasks.
"""
        
        return prompt
    
    def _prepare_image_data(self, visualized_screenshots: List[Dict[str, Any]]) -> List:
        """准备图像数据用于LLM调用"""
        image_data_list = []
        
        for screenshot_info in visualized_screenshots[:15]:  # 限制图像数量
            try:
                visualized_image = screenshot_info["visualized_image"]
                # 直接传递PIL Image对象，gpt4v_call函数会处理转换
                image_data_list.append(visualized_image)
                
            except Exception as e:
                print(f"    Failed to prepare image data: {e}")
                continue
        
        print(f"    Prepared {len(image_data_list)} images for LLM call")
        return image_data_list
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 查找JSON内容
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                print("    No JSON found in response, using fallback")
                return self._generate_fallback_result("Unknown", "Unknown task")
            
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            
            # 验证结果格式
            if not self._validate_result_format(result):
                print("    Invalid result format, using fallback")
                return self._generate_fallback_result("Unknown", "Unknown task")
            
            print(f"    Successfully parsed result with {len(result.get('generated_tasks', []))} tasks")
            return result
            
        except json.JSONDecodeError as e:
            print(f"    JSON decode error: {e}")
            # 尝试修复常见的JSON错误
            return self._try_fix_json(response_text)
        except Exception as e:
            print(f"    Error parsing response: {e}")
            return self._generate_fallback_result("Unknown", "Unknown task")
    
    def _try_fix_json(self, response_text: str) -> Dict[str, Any]:
        """尝试修复常见的JSON错误"""
        try:
            # 移除可能的markdown标记
            text = response_text.replace('```json', '').replace('```', '')
            
            # 查找JSON内容
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                
                # 尝试修复截断的JSON
                if not json_str.endswith('}'):
                    # 简单的修复策略
                    if '"generated_tasks"' in json_str and not json_str.endswith(']}'):
                        json_str += ']}'
                    elif not json_str.endswith('}'):
                        json_str += '}'
                
                result = json.loads(json_str)
                
                if self._validate_result_format(result):
                    print("    Successfully fixed JSON format")
                    return result
            
        except Exception as e:
            print(f"    JSON fix failed: {e}")
        
        return self._generate_fallback_result("Unknown", "Unknown task")
    
    def _validate_result_format(self, result: Dict[str, Any]) -> bool:
        """验证结果格式"""
        if not isinstance(result, dict):
            return False
        
        # 检查evaluation部分
        evaluation = result.get('evaluation', {})
        if not isinstance(evaluation, dict):
            return False
        
        required_eval_keys = ['task_reasonable', 'task_completed']
        if not all(key in evaluation for key in required_eval_keys):
            return False
        
        # 检查generated_tasks部分
        tasks = result.get('generated_tasks', [])
        if not isinstance(tasks, list):
            return False
        
        # 检查每个任务的格式
        for task in tasks:
            if not isinstance(task, dict):
                return False
            required_task_keys = ['instruction', 'estimated_steps']
            if not all(key in task for key in required_task_keys):
                return False
        
        return True
    
    def _generate_fallback_result(self, app_name: str, original_goal: str) -> Dict[str, Any]:
        """生成fallback结果"""
        print("    Generating fallback result")
        
        return {
            "evaluation": {
                "task_reasonable": True,
                "task_completed": False,
                "reasonableness_explanation": f"Task appears to be a reasonable operation for {app_name}",
                "completion_explanation": "Unable to evaluate completion due to processing error",
                "confidence_score": 0.5
            },
            "generated_tasks": [
                {
                    "task_id": "fallback_task_1",
                    "instruction": f"In {app_name}, navigate to the main feature and perform a complete workflow including creating, modifying and verifying an item",
                    "estimated_steps": 18,
                    "core_functionality": "Basic app workflow",
                    "variation_type": "step_progression",
                    "prerequisites": "None"
                }
            ]
        }
    
    def _filter_tasks_by_app_name(self, tasks: List[Dict[str, Any]], app_name: str) -> Tuple[List[Dict[str, Any]], int]:
        """
        筛选包含完整app_name的任务
        
        Args:
            tasks: 生成的任务列表
            app_name: 应用名称
            
        Returns:
            (有效任务列表, 被过滤的任务数量)
        """
        valid_tasks = []
        filtered_count = 0
        
        for task in tasks:
            instruction = task.get('instruction', '')
            # 大小写不敏感的匹配
            if app_name.lower() in instruction.lower():
                valid_tasks.append(task)
            else:
                filtered_count += 1
                print(f"    [FILTERED] Task missing app_name '{app_name}': {instruction[:80]}...")
        
        return valid_tasks, filtered_count
    
    def _contains_chinese_characters(self, text: str) -> bool:
        """
        检测文本是否包含中文字符
        
        Args:
            text: 要检测的文本
            
        Returns:
            True 如果包含中文字符，否则 False
        """
        # Unicode 范围涵盖：
        # - CJK Unified Ideographs: \u4e00-\u9fff
        # - CJK Unified Ideographs Extension A: \u3400-\u4dbf
        # - CJK Compatibility Ideographs: \uf900-\ufaff
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
        return bool(chinese_pattern.search(text))
    
    def _filter_tasks_with_chinese(self, tasks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """
        过滤掉包含中文字符的任务（包括乱码中文）
        
        Args:
            tasks: 生成的任务列表
            
        Returns:
            (有效任务列表, 被过滤的任务数量)
        """
        valid_tasks = []
        filtered_count = 0
        
        for task in tasks:
            instruction = task.get('instruction', '')
            if self._contains_chinese_characters(instruction):
                filtered_count += 1
                print(f"    [FILTERED-CHINESE] Task contains Chinese characters: {instruction[:80]}...")
            else:
                valid_tasks.append(task)
        
        return valid_tasks, filtered_count
    
    def _update_token_usage(self, response_text: str) -> None:
        """更新token使用统计"""
        # 简单估算token使用
        estimated_input_tokens = 2000  # 基于prompt和图像的估算
        estimated_output_tokens = len(response_text.split()) * 1.3  # 估算输出token
        
        self.token_usage["total_input_tokens"] += estimated_input_tokens
        self.token_usage["total_output_tokens"] += estimated_output_tokens
        self.token_usage["call_count"] += 1
        
        # 估算成本（GPT-4 Vision的大概价格）
        input_cost = estimated_input_tokens * 0.01 / 1000
        output_cost = estimated_output_tokens * 0.03 / 1000
        self.token_usage["total_cost_usd"] += input_cost + output_cost
    
    def get_token_usage(self) -> Dict[str, Any]:
        """获取token使用统计"""
        return self.token_usage.copy()
    
    def get_last_llm_debug(self) -> Optional[Dict[str, Any]]:
        """获取最后一次LLM调用的调试信息"""
        return self.last_llm_debug
    
    def reset_token_usage(self) -> None:
        """重置token使用统计"""
        self.token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "call_count": 0
        }
