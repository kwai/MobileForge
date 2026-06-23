def get_describe_step_prompt(
    task_description: str, log_action: str, log_detail: str
) -> (str, str):
    """
    Returns the system and user prompts for describing a single step/image in JSON format.
    """
    system_prompt = (
        "You are an expert mobile device assistant. Your task is to analyze a two-panel image showing the 'Before Action' and 'After Action' state of a user's workflow. "
        "Your analysis must focus *only* on the 'Before Action' panel (the left side). "
        "You must output your response in a JSON format."
    )
    user_prompt = (
        f"The overall task is: '{task_description}'.\n\n"
        "## Input Analysis\n"
        "The provided image shows a 'Before Action' state on the left and an 'After Action' state on the right. "
        "Your entire analysis should focus on the left 'Before Action' panel.\n\n"
        "**Note:** If the 'After Action' panel is identical to the 'Before Action' panel, it signifies this is the final action in the task.\n\n"
        "On the left panel, a user action is visualized with markers: a red circle shows the click/touch point, surrounded by a green square, with a 'C' label in the corner. "
        "The raw action from the execution log is provided for context:\n"
        f"- Action Type: `{log_action}`\n"
        f"- Action Detail: `{log_detail}`\n\n"
        "## Your Task\n"
        "Based on the visual evidence in the **left panel** and the provided log context, perform the following two tasks:\n"
        "1. **action_description**: In your own words, crisply describe the specific action performed (e.g., 'Clicked the \"Settings\" button', 'Typed \"hello\" into the search bar').\n"
        "2. **ui_description**: List the key UI elements visible *in the left panel* that are relevant to the action and the overall task. Do not mention the panel name (e.g., 'Before Action') in your description.\n\n"
        "Your output MUST be a JSON object with these two keys.\n\n"
        "### Example\n"
        "```json\n"
        "{{\n"
        '  "action_description": "The user clicked on the settings icon at the bottom of the screen.",\n'
        '  "ui_description": "The home screen with various app icons is visible. Key elements include the Phone, Messages, and Settings icons at the bottom."\n'
        "}}\n"
        "```"
    )
    return system_prompt, user_prompt


def get_describe_final_step_prompt(
    task_description: str, log_action: str, log_detail: str
) -> (str, str):
    """
    Returns the system and user prompts for describing the final single-step image.
    """
    system_prompt = (
        "You are an expert mobile device assistant. Your task is to analyze a single image showing the state of a screen after a user's action. "
        "The user's action (e.g., a click) is visualized on the image itself. "
        "Your analysis must focus on the provided image and the action context. You must output your response in a JSON format."
    )
    user_prompt = (
        f"The overall task is: '{task_description}'.\n\n"
        "## Input Analysis\n"
        "The provided image shows the state of the screen after a final user action. "
        "On the image, the user action may be visualized with markers (e.g., a red circle for a click).\n\n"
        "The raw action from the execution log is provided for context:\n"
        f"- Action Type: `{log_action}`\n"
        f"- Action Detail: `{log_detail}`\n\n"
        "## Your Task\n"
        "Based on the visual evidence in the image and the provided log context, perform the following two tasks:\n"
        "1. **action_description**: In your own words, crisply describe the specific action performed (e.g., 'Clicked the final confirmation button').\n"
        "2. **ui_description**: List the key UI elements visible in this final screen state that are relevant to the action and the overall task.\n\n"
        "**Special Instruction for task completion signals:** If the `Action Detail` suggests the task has been concluded (e.g., it includes words like `completed` or `finished`), describe the action from the user's perspective, such as 'The user ended the task.' Do not state that the task *is* complete, as that is for a later evaluation step.\n\n"
        "Your output MUST be a JSON object with these two keys.\n\n"
        "### Example\n"
        "```json\n"
        "{{\n"
        '  "action_description": "The user clicked the \'Done\' button to complete the process.",\n'
        '  "ui_description": "The confirmation screen is visible, showing a success message and a \'Done\' button."\n'
        "}}\n"
        "```"
    )
    return system_prompt, user_prompt


def _get_evaluation_guidelines() -> str:
    """Returns the shared evaluation guidelines text."""
    return """## Evaluation Guidelines
1.  **Final UI State**: The "final UI state" is the conceptual state of the UI after all actions are performed. It must meet all task requirements. This state may be represented by the last screenshot, or a collection of screenshots from the middle and end of the sequence that together prove task completion. **Information Organization**: When tasks require inputting answers/information into note-taking apps, messaging apps, or similar software, the information must be organized in a logical and orderly manner. Mixed or chaotic organization (e.g., Point 1.1, Point 2.1, Point 2.2, Point 1.2) should be considered task failure, as proper information structure is essential for task completion quality.
2.  **Pre-existing Conditions**: If a task requirement was already met before the agent started (e.g., a 'Shopping' note already exists when the task is to create one), the agent does not need to repeat the action. The task is still considered successful if the final state is correct.
3.  **Trust Correct Actions**: If a sequence of actions is logically correct for the task (e.g., 'Click Save'), you can infer the action was successful and the state was achieved, even if the final screenshot shows a different screen (e.g., the agent has navigated back to the home screen).
4.  **Allow Error Correction**: The agent can make and correct mistakes. As long as the final goal is achieved, intermediate errors do not affect the outcome.
5.  **Handle Unreasonable Tasks**: If a task is inherently unreasonable or impossible to complete (e.g., requesting to find 3 reviews for a newly released product that has no reviews yet), the agent can still be considered successful if it correctly identifies the impossibility and provides appropriate feedback. For example, writing "not found", "no reviews available", or any other clear indication that the agent recognized the task's unreasonable nature is acceptable as successful task completion.
"""


def _get_base_final_decision_prompt(
    task_description: str, step_descriptions: list
) -> (str, str):
    """Internal helper to generate the base system prompt and formatted steps."""
    system_prompt = f"""You are an expert in evaluating mobile UI automation tasks.
{_get_evaluation_guidelines()}"""

    formatted_steps = []
    for i, desc_obj in enumerate(step_descriptions):
        if isinstance(desc_obj, dict):
            step_label = f"Step {i + 1}"
            vlm_action = desc_obj.get("action_description", "N/A")
            vlm_ui = desc_obj.get("ui_description", "N/A")
            raw_action = desc_obj.get("_raw_action", {})
            raw_action_type = raw_action.get("type", "N/A")
            raw_action_detail = raw_action.get("detail", "N/A")

            formatted_steps.append(
                f"- {step_label}:\n"
                f"  - Raw Action Log: type=`{raw_action_type}`, detail=`{raw_action_detail}`\n"
                f"  - VLM-Generated UI Description: {vlm_ui}\n"
                f"  - VLM-Generated Action Description: {vlm_action}\n"
            )
        else:
            step_label = f"Step {i + 1}"
            formatted_steps.append(f"- {step_label}: {desc_obj}")
    formatted_steps_str = "\n".join(formatted_steps)

    base_user_prompt = (
        f"Task Description: '{task_description}'\n\n"
        "Here is a step-by-step breakdown of the agent's actions, including both raw logs and descriptions generated by a Vision Language Model (VLM):\n"
        f"{formatted_steps_str}\n\n"
    )
    return system_prompt, base_user_prompt


def get_final_decision_prompt(
    task_description: str,
    step_descriptions: list,
    uncertainty_reason: str = "",
) -> (str, str):
    """
    Creates prompts for the detailed decision, allowing the LLM to request more screenshots.
    This phase includes step-by-step descriptions and the last 3 screenshots.
    """
    system_prompt, base_user_prompt = _get_base_final_decision_prompt(
        task_description, step_descriptions
    )

    if uncertainty_reason:
        system_prompt += f"""
A previous, less-informed evaluation stage was 'Uncertain' for the following reason: '{uncertainty_reason}'.
Please pay special attention to this aspect. You are now provided with more information (detailed step descriptions and the last 3 screenshots).
"""

    user_prompt = base_user_prompt + (
        "You are now provided with a composite image of the last 3 screenshots. Note that this is only a partial view of the execution. You must synthesize this visual information with the full list of text descriptions to understand the complete workflow.\n\n"
        "**TASK FEASIBILITY ASSESSMENT**: BEFORE evaluating agent performance, you must first assess if the task itself is fundamentally feasible and reasonable given the execution context.\n\n"
        "Consider:\n"
        "1. **Resource Availability**: Does the task require resources that are not available? (e.g., asking to share a photo when the gallery is empty)\n"
        "2. **Prerequisites**: Are necessary prerequisites missing? (e.g., asking to edit a specific note that doesn't exist)\n"
        "3. **Environmental Constraints**: Are there factors that make the task impossible?\n\n"
        '**IMPORTANT**: Even if a task is deemed unreasonable/infeasible, you should still evaluate the agent\'s performance. If the agent correctly identifies the impossibility and provides appropriate feedback (e.g., "photo gallery is empty", "contact not found"), this should be considered successful task completion.\n\n'
        "**CRITICAL WARNING ABOUT TEXT DESCRIPTIONS**: The text-based UI descriptions provided above are INCOMPLETE and may be MISSING CRITICAL INFORMATION. They are generated automatically and may omit important details, parameters, values, or UI elements that are essential for evaluating task completion. DO NOT rely solely on these text descriptions for your decision.\n\n"
        "**MANDATORY VERIFICATION**: Before making any decision, you MUST verify that ALL key information required by the task description is present in either:\n"
        "1. The text descriptions, OR\n"
        "2. The provided screenshots\n\n"
        "If critical information is missing, you should make a reasonable judgment based on the available evidence. If the evidence suggests the task was likely completed correctly, decide success; otherwise, decide failure.\n\n"
        "**FINAL DECISION REQUIRED**: Based on all available information, you MUST make a definitive judgment. "
        "Respond with 'decision' 1 (success) or 0 (failure). You cannot defer or request more information - a decision must be made now.\n\n"
        "**FAILURE STEP TRACKING**: If you determine the task failed (decision = 0), you MUST specify exactly which step number caused the failure by including a 'failure_step' field with the step number where the critical error occurred. Additionally, in your 'reason' field, you MUST include a specific explanation of why you identified this particular step as the failure point (e.g., 'Step X was the failure point because it was where the agent should have applied the required filter but failed to do so').\n\n"
        "**STEP REASONABLENESS ANALYSIS**: In addition to the overall task success/failure judgment, you MUST provide a detailed analysis of step reasonableness. For EACH step in the trajectory, analyze:\n"
        "- **reasonable_steps**: Array of step numbers that were reasonable and contributed positively to task completion\n"
        "- **unreasonable_steps**: Array of step numbers that were unreasonable, incorrect, or counterproductive\n"
        "- **step_analysis**: Object with step numbers as keys, containing:\n"
        "  - **reasonableness**: 'reasonable' or 'unreasonable'\n"
        "  - **explanation**: Detailed explanation of why this step was reasonable/unreasonable\n"
        "  - **impact**: How this step affected task progress ('positive', 'negative', 'neutral')\n\n"
        "Note: Even successful tasks may contain some unreasonable steps that were later corrected. Even failed tasks may contain reasonable steps that moved toward the goal.\n\n"
        "**TASK FEASIBILITY OUTPUT**: You must also include task feasibility assessment in your response:\n"
        "- **task_feasible**: boolean (true if task is reasonable and achievable, false if not)\n"
        "- **task_feasible_reason**: string explaining your feasibility assessment\n"
        "- **task_barriers**: array of specific barriers that make the task infeasible (empty if feasible)\n\n"
        "Example (Success with mixed step quality and feasible task):\n"
        "```json\n"
        '{\n  "decision": 1,\n  "reason": "Task completed successfully despite some inefficient steps.",\n  "task_feasible": true,\n  "task_feasible_reason": "Task is reasonable - setting a timer is always feasible with a clock app",\n  "task_barriers": [],\n  "reasonable_steps": [1, 2, 4, 5],\n  "unreasonable_steps": [3],\n  "step_analysis": {\n    "1": {"reasonableness": "reasonable", "explanation": "Correctly opened the target app", "impact": "positive"},\n    "2": {"reasonableness": "reasonable", "explanation": "Successfully navigated to search function", "impact": "positive"},\n    "3": {"reasonableness": "unreasonable", "explanation": "Applied wrong filter initially, wasted time", "impact": "negative"},\n    "4": {"reasonableness": "reasonable", "explanation": "Corrected the filter to the required one", "impact": "positive"},\n    "5": {"reasonableness": "reasonable", "explanation": "Successfully completed the final action", "impact": "positive"}\n  }\n}\n'
        "```\n\n"
        "Example (Agent success on infeasible task):\n"
        "```json\n"
        '{\n  "decision": 1,\n  "reason": "Task was inherently impossible but agent correctly identified the issue and provided appropriate feedback.",\n  "task_feasible": false,\n  "task_feasible_reason": "Photo gallery is completely empty, cannot share photos that don\'t exist",\n  "task_barriers": ["Empty photo gallery", "No photos available to share"],\n  "reasonable_steps": [1, 2, 3],\n  "unreasonable_steps": [],\n  "step_analysis": {\n    "1": {"reasonableness": "reasonable", "explanation": "Correctly opened the gallery app", "impact": "positive"},\n    "2": {"reasonableness": "reasonable", "explanation": "Checked for available photos", "impact": "positive"},\n    "3": {"reasonableness": "reasonable", "explanation": "Correctly identified no photos available and provided feedback", "impact": "positive"}\n  }\n}\n'
        "```\n\n"
        "Example (Failure with step analysis):\n"
        "```json\n"
        '{\n  "decision": 0,\n  "reason": "Task failed at step 4 where wrong product was selected. Step 4 was the failure point because the agent selected an incorrect item despite correct ones being visible.",\n  "failure_step": 4,\n  "task_feasible": true,\n  "task_feasible_reason": "Task is achievable with the available resources",\n  "task_barriers": [],\n  "reasonable_steps": [1, 2, 3],\n  "unreasonable_steps": [4, 5],\n  "step_analysis": {\n    "1": {"reasonableness": "reasonable", "explanation": "Correctly opened shopping app", "impact": "positive"},\n    "2": {"reasonableness": "reasonable", "explanation": "Proper search execution", "impact": "positive"},\n    "3": {"reasonableness": "reasonable", "explanation": "Applied correct filters", "impact": "positive"},\n    "4": {"reasonableness": "unreasonable", "explanation": "Selected wrong product despite correct ones being available", "impact": "negative"},\n    "5": {"reasonableness": "unreasonable", "explanation": "Continued with wrong product instead of correcting", "impact": "negative"}\n  }\n}\n'
        "```"
    )
    return system_prompt, user_prompt


def get_task_feasibility_prompt(
    task_description: str, step_descriptions: list
) -> (str, str):
    """
    Creates prompts for evaluating task feasibility/reasonableness before judging agent performance.
    """
    system_prompt = """You are an expert in evaluating mobile UI task feasibility. Your role is to determine whether a given task is inherently feasible/reasonable based on the execution context and available resources.

**Your Task**: Before evaluating agent performance, you must first assess if the task itself is reasonable and achievable given the constraints and context shown in the execution trajectory.

**Key Considerations**:
1. **Resource Availability**: Does the task require resources that are not available? (e.g., asking to share a photo when the gallery is empty, contacting a person not in contacts)
2. **Prerequisites**: Are necessary prerequisites missing? (e.g., asking to edit a specific note that doesn't exist)
3. **Logical Consistency**: Is the task internally consistent and logically achievable?
4. **Environmental Constraints**: Are there environmental factors that make the task impossible? (e.g., no internet connection when online actions are required)

**Important Notes**:
- Even if a task is deemed unreasonable/infeasible, the agent evaluation should still proceed
- If the agent correctly identifies the impossibility and provides appropriate feedback (e.g., "photo gallery is empty", "contact not found"), this should be considered successful task completion
- Only mark a task as unreasonable if there are fundamental barriers that prevent completion, not if the agent simply performed poorly

**Response Format**: You must respond with a JSON object containing:
- "feasible": boolean (true if task is reasonable and achievable, false if not)
- "reason": string explaining your assessment
- "barriers": array of specific barriers that make the task infeasible (empty if feasible)
"""

    formatted_steps = []
    for i, desc_obj in enumerate(step_descriptions):
        if isinstance(desc_obj, dict):
            step_label = f"Step {i + 1}"
            action = desc_obj.get("action_description", "N/A")
            ui = desc_obj.get("ui_description", "N/A")
            formatted_steps.append(f"- {step_label}:\n  Action: {action}\n  UI: {ui}")
        else:
            step_label = f"Step {i + 1}"
            formatted_steps.append(f"- {step_label}: {desc_obj}")
    formatted_steps_str = "\n".join(formatted_steps)

    user_prompt = (
        f"Task Description: '{task_description}'\n\n"
        "Here is the execution trajectory with step-by-step descriptions:\n"
        f"{formatted_steps_str}\n\n"
        "Based on the task description and the observed execution context, evaluate if this task is fundamentally feasible and reasonable.\n\n"
        "Examples of unreasonable tasks:\n"
        "- 'Send the first photo from gallery to John' when the photo gallery is completely empty\n"
        "- 'Edit the note titled \"Shopping List\"' when no such note exists and cannot be created\n"
        "- 'Call emergency contact' when no emergency contacts are configured\n\n"
        "Examples of reasonable tasks (even if agent fails):\n"
        "- 'Set a timer for 10 seconds' - this is always feasible if the device has a timer app\n"
        "- 'Search for restaurants nearby' - feasible if maps/search apps are available\n"
        "- 'Take a photo' - feasible if camera access is available\n\n"
        "Provide your assessment in JSON format."
    )
    return system_prompt, user_prompt


def get_pre_evaluation_prompt(
    task_description: str, raw_action_logs: list, total_steps: int
) -> (str, str):
    """
    Creates system and user prompts for the pre-evaluation phase.
    """
    system_prompt = f"""You are an expert in evaluating mobile UI automation tasks. Your goal is to determine if a task has DEFINITELY succeeded based on VERY limited information. You must be extremely confident to make a "Success" decision.

{_get_evaluation_guidelines()}

You will be given:
1. The task description.
2. The raw action logs (without semantic descriptions).
3. A single image combining the last 3 screenshots out of a total of {total_steps} screenshots.

**Crucial Instructions:**
- The information provided is INCOMPLETE. You are only seeing the final UI states and raw, low-level actions.
- You must be EXTREMELY conservative. Only conclude "Success" if the provided evidence is undeniable and accounts for ALL conditions in the task description with absolute certainty.
- If there is ANY ambiguity or any task condition that cannot be verified from the final screenshots (e.g., a filter that was applied in an earlier step), you MUST respond with "Uncertain" and provide a reason. You cannot decide "Failure" at this stage.

**MANDATORY VERIFICATION**: Before making any decision, you MUST verify that ALL key information required by the task description is present in either:
1. The raw action logs, OR
2. The provided screenshots

If ANY critical information, parameters, values, or UI elements mentioned in the task description are NOT clearly visible in the provided screenshots and NOT evident from the raw action logs, you MUST respond with "Uncertain". Do not guess or infer missing information. All required information must be explicitly present and verifiable.

Example Scenarios for "Uncertain":
- Task: "In Amazon, search for 'laptop', filter by '4 stars & up', and add the first item to the cart."
- Provided Info: The final screenshot shows an item in the shopping cart.
- Correct Response: "Uncertain". The final screenshot proves an item was added to the cart, but it's impossible to verify if the '4 stars & up' filter was correctly applied. This requires more information.

- Task: "In Amazon, search for three products A, B, and C in sequence, remember their prices and star ratings, then write down the information you just found in a note-taking app."
- Provided Info: The final screenshots show a note-taking app with prices and ratings for products A, B, and C.
- Correct Response: "Uncertain". While the final screenshots show that information was written in the note-taking app, it's impossible to verify if the recorded prices and ratings actually match the real search results from earlier steps, since the search result screenshots are not provided.

Respond with a JSON object containing "reason" and "decision" ("Success" or "Uncertain").
"""
    raw_actions_str = "\n".join([f"- {log}" for log in raw_action_logs])
    user_prompt = f"""Task Description:
{task_description}

Total Steps in Full Trajectory: {total_steps}
Raw Action Sequence:
{raw_actions_str}

Please evaluate the task outcome based on the provided image showing only the final UI states and the raw action logs.
"""

    return system_prompt, user_prompt


def get_eval_hint_prompt(
    task_description: str,
    failure_reason: str,
    step_descriptions: dict,
    step_analysis: dict,
) -> tuple:
    """
    Returns the system and user prompts for generating self-reflection hints.
    This is used to create learning experiences from attempts with issues.

    Args:
        task_description: The task that was attempted
        failure_reason: Why the task failed or what issues were found
        step_descriptions: Description of steps taken
        step_analysis: Reasonableness analysis for each step from evaluator

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are an intelligent agent performing self-reflection on your task execution. "
        "Your role is to analyze your own actions, identify mistakes and inefficiencies, "
        "and extract valuable lessons to improve your future attempts. "
        "You have access to detailed step-by-step analysis including which of your steps were reasonable and which were not. "
        "You must provide honest, constructive self-critique in JSON format."
    )

    # Build step sequence with reasonableness analysis
    sorted_steps = sorted([k for k in step_descriptions.keys() if isinstance(k, int)])
    step_sequence_parts = []

    for i, step in enumerate(sorted_steps):
        step_desc = step_descriptions[step]
        step_num_str = str(step)

        # Format step description
        if isinstance(step_desc, dict):
            action_desc = step_desc.get("action_description", "N/A")
            ui_desc = step_desc.get("ui_description", "N/A")
            step_line = f"Step {step}: Action: {action_desc}\n  UI: {ui_desc}"
        else:
            step_line = f"Step {step}: {step_desc}"

        # Add reasonableness analysis if available
        if step_analysis and step_num_str in step_analysis:
            analysis = step_analysis[step_num_str]
            reasonableness = analysis.get("reasonableness", "unknown")
            explanation = analysis.get("explanation", "")
            impact = analysis.get("impact", "unknown")

            step_line += f"\n  → Analysis: {reasonableness.upper()}"
            if explanation:
                step_line += f"\n  → Reason: {explanation}"
            step_line += f"\n  → Impact: {impact}"

        step_sequence_parts.append(step_line)

    step_sequence = "\n\n".join(step_sequence_parts)

    user_prompt = (
        f"## Task\n{task_description}\n\n"
        "## My Steps (with Reasonableness Analysis)\n"
        f"{step_sequence}\n\n"
        "## Evaluation Result\n"
        f"{failure_reason}\n\n"
        "## Self-Reflection Task\n"
        "Reflect on your execution and analyze what went wrong or could be improved. "
        "You have detailed step-by-step analysis showing which of your actions were reasonable and which were not. "
        "Perform honest self-critique to:\n"
        "1. Identify your key mistakes (especially focus on steps marked as 'unreasonable')\n"
        "2. Understand what you should avoid in future attempts\n"
        "3. Develop concrete strategies to try instead\n"
        "4. Extract key insights about the task requirements\n\n"
        "Output a JSON object with these fields:\n"
        '- "key_mistake": A concise summary of your main mistake (1-2 sentences, focus on unreasonable steps)\n'
        '- "what_to_avoid": Specific actions or approaches you should avoid (bullet points, array of strings)\n'
        '- "suggested_approach": Concrete alternative approach to try (bullet points, array of strings)\n'
        '- "important_insights": Key learnings about the task (bullet points, array of strings)\n'
        '- "hint_summary": A brief self-reminder for the next attempt (2-3 sentences)\n\n'
        "### Example Output\n"
        "```json\n"
        "{\n"
        '  "key_mistake": "I wasted 18 steps scrolling inefficiently through settings instead of using the search function, which was available and would have been much faster.",\n'
        '  "what_to_avoid": [\n'
        '    "Do not scroll repeatedly when a search function is available",\n'
        '    "Avoid dragging app icons when trying to open settings",\n'
        '    "Do not waste time on inefficient navigation methods"\n'
        "  ],\n"
        '  "suggested_approach": [\n'
        '    "Use the search bar immediately for efficiency",\n'
        '    "Type the exact setting name directly into search",\n'
        '    "After finding information, proceed directly to the next requirement"\n'
        "  ],\n"
        '  "important_insights": [\n'
        '    "Search functionality is faster than manual scrolling in settings",\n'
        '    "I eventually found the correct approach but too late",\n'
        '    "Task requires finding TWO pieces of information, not just one"\n'
        "  ],\n"
        '  "hint_summary": "My previous attempt failed because I spent too much time scrolling instead of using the search function immediately. For the next attempt, I should use search from the start for both pieces of information, and make sure to complete BOTH requirements before finishing."\n'
        "}\n"
        "```\n\n"
        "Be specific, honest, and focus on actionable improvements. "
        "Pay special attention to steps marked as 'unreasonable' - these are the main areas for improvement."
    )

    return system_prompt, user_prompt
