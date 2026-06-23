# INPUTS: app_name, package_name, activity_list
TASK_GOAL_GENERATOR = """Given the screenshot of {app_name} and its available activities, generate a comprehensive list of practical user tasks that:

1. Start from the current screen shown in the screenshot
2. Can be completed within 10-30 steps
3. Utilize the app's full feature set based on the activity list
4. Are concrete and specific (like searching for a particular item rather than just "search")
5. Cover different user interaction patterns (viewing, editing, sharing, etc.)
6. Include both basic and advanced features
7. Represent realistic user behaviors and goals
8. Avoid excessive steps on form-filling or scrolling pages

Important context:
- App name: {app_name}
- Package name: {package_name} 
- Available activities (app screens/features):
```{activity_list}```

Format requirements:
1. List only the tasks without explanations or commentary
2. Each task should be a single, clear directive
3. Use specific examples (e.g., concrete search terms, actions, settings)
4. Include the expected outcome where relevant
5. Tasks should follow this pattern: [Starting action] + [Specific steps] + [End goal]

Example tasks from other apps (for reference only):
1. Search for "ocean waves" white noise, then sort results by most played
2. Open the first recommended video, then post "Great content!" as a comment
3. Play the trending video, then add it to your "Watch Later" playlist
4. Navigate to the comments section of a featured video, then like the top comment

Generate diverse tasks that would help a user explore and utilize all major features visible in the screenshot and implied by the activity list."""


# INPUTS: task_description, numeric_tag_of_element, ui_element_attributes, action
KNOWLEDGE_EXTRACTOR = """Objective: Describe the functionality of a specific UI element in a mobile app screenshot.

Input:
- Two screenshots: Before and after interacting with a UI element
- UI element marked with a numeric tag in the top-left corner
- Element number: {numeric_tag_of_element}
- Broader task context: {task_description}
- Action taken: {action}
- UI Element Attributes: 
  ```
  {ui_element_attributes}
  ```

Requirements for Functionality Description:
1. Concise: 1-2 sentences
2. Focus on general function, not specific details
3. Avoid mentioning the numeric tag
4. Use generic terms like "UI element" or appropriate pronouns

Example:
- Incorrect: "Tapping the element #3 displays David's saved recipes in the results panel"
- Correct: "Tapping this element will initiates a search and displays matching results"

Guidance:
- Describe the core action and immediate result of interacting with the UI element
- Prioritize clarity and generality in the description"""


# INPUTS: task_goal, knowledge_a, knowledge_b
RANKER = """Given the user instruction: {task_goal}, determine which of the following two knowledge entries is more useful.
Respond ONLY with a integer value:
1 means Knowledge A is strictly better.
2 means Knowledge B is strictly better.

Knowledge A: {knowledge_a}
Knowledge B: {knowledge_b}

Please provide your response:
"""


# INPUTS: task_goal, history, ui_elements, knowledge
REASONING = """## Role Definition
You are an Android operation AI that fulfills user requests through precise screen interactions.
The current screenshot and the same screenshot with bounding boxes and labels added are also given to you.

## Action Catalog
Available actions (STRICT JSON FORMAT REQUIRED):
1. Status Operations:
   - Task Complete: {{"action_type": "status", "goal_status": "complete"}}
   - Task Infeasible: {{"action_type": "status", "goal_status": "infeasible"}}
2. Information Actions:
   - Answer Question: {{"action_type": "answer", "text": "<answer_text>"}}
3. Screen Interactions:
   - Tap Element: {{"action_type": "click", "index": <visible_index>}}
   - Long Press: {{"action_type": "long_press", "index": <visible_index>}}
   - Scroll: Scroll the screen or a specific scrollable UI element. Use the `index` of the target element if scrolling a specific element, or omit `index` to scroll the whole screen. {{"action_type": "scroll", "direction": <"up"|"down"|"left"|"right">, "index": <optional_target_index>}}
4. Input Operations:
   - Text Entry: {{"action_type": "input_text", "text": "<content>"}}
   - Keyboard Enter: {{"action_type": "keyboard_enter"}}
5. Navigation:
   - Home Screen: {{"action_type": "navigate_home"}}
   - Back Navigation: {{"action_type": "navigate_back"}}
6. System Actions:
   - Launch App: {{"action_type": "open_app", "app_name": "<exact_name>"}}
   - Wait Refresh: {{"action_type": "wait"}}

## Current Objective
User Goal: {task_goal}

## Execution Context
Action History:
{history}

Visible UI Elements (Only interact with *visible=true elements):
{ui_elements}

## Core Strategy
1. Path Optimization:
   - Prefer direct methods (e.g., open_app > app drawer navigation)
   - Always use the `input_text` action for entering text into designated text fields.
   - Before entering text, click the dialog box where you want to enter text, confirm the selection, and then use the "input_text" action.
   - Verify element visibility (`visible=true`) before attempting any interaction (click, long_press, input_text). Do not interact with elements marked `visible=false`.
   - Use `scroll` when necessary to bring off-screen elements into view. Prioritize scrolling specific containers (`index` provided) over full-screen scrolls if possible.

2. Error Handling Protocol:
   - Switch approach after ≥ 2 failed attempts
   - Prioritize scrolling (`scroll` action) over force-acting on invisible elements
   - If an element is not visible, use `scroll` in the likely direction (e.g., 'down' to find elements below the current view).
   - Try opposite scroll direction if initial fails (up/down, left/right)
   - If the `open_app` action fails to correctly open the app, find the corresponding app in the app drawer and open it.

3. Information Tasks:
   - MANDATORY: Use answer action for questions
   - Verify data freshness (e.g., check calendar date)

## Expert Techniques
Here are some tips for you:
{knowledge}

## Response Format
STRICTLY follow:
Reasoning: [Step-by-step analysis covering:
           - Visibility verification
           - History effectiveness evaluation
           - Alternative approach comparison
           - Consideration of scrolling if needed]
Action: [SINGLE JSON action from catalog]

Generate response:
"""

# INPUTS: task_goal, before_ui_elements, after_ui_elements, action, reasoning
SUMMARY="""
Goal: {task_goal}

Before screenshot elements:
{before_ui_elements}

After screenshot elements:
{after_ui_elements}

Action: {action}
Reasoning: {reasoning}

Provide a concise single-line summary (under 50 words) of this step by comparing screenshots and action outcome. Include:
- What was intended
- Whether it succeeded
- Key information for future actions
- Critical analysis if action/reasoning was flawed
- Important data to remember across apps

For actions like 'answer' or 'wait' with no screen change, assume they worked as intended.

Summary:
"""

# INPUTS: task_goal, trajectory_summary, final_ui_elements, final_screenshot_with_som
TASK_EVALUATOR = """You are an AI capable of evaluating whether a given task has been successfully completed in an Android device exploration trajectory.

Given the following information:
- Task goal: {task_goal}
- Exploration trajectory summary:
{trajectory_summary}
- Visible UI elements on the final screenshot:
{final_ui_elements}

Please determine whether the task goal has been achieved based on the provided trajectory summary and the UI elements in the final screenshot.

Evaluation criteria:
1. **Task completed (complete)**: All key requirements of the task goal have been clearly achieved in the trajectory, and the final screen state matches the expected completion state.
2. **Task incomplete (incomplete)**: Although some operations were performed in the trajectory, the key requirements of the task goal have not been fully achieved, or the final screen state indicates the task is still in progress.
3. **Task infeasible (infeasible)**: Based on the trajectory and final screen state, the task goal is fundamentally impossible to achieve in the current environment or has encountered insurmountable obstacles.

Please respond strictly in the following JSON format, without any additional text or explanations:
{{
  "evaluation": "<complete|incomplete|infeasible>",
  "reasoning": "<Brief explanation of your evaluation, explaining why the task is considered complete/incomplete/infeasible.>"
}}

Your response:
"""

# INPUTS: app_name, original_task_goal, package_name, activity_list, difficulty_level, trajectory_screenshots (implicit)
TASK_GENERATOR = """Based on the exploration trajectory screenshots of app '{app_name}' (package name: {package_name}), available activity list (```{activity_list}```), and original task '{original_task_goal}', please generate 5 new user tasks with **{difficulty_level} difficulty**.

## Difficulty Level Definitions
- **Simple tasks (low_level)**: 1-2 steps
- **Medium difficulty tasks (medium_level)**: 3-4 steps
- **Complex tasks (high_level)**: 5-6 steps

## Core Constraints
1. **Scope Limitation**: All tasks must be based solely on interfaces and functions shown in the provided trajectory screenshots
2. **Interface Availability**: Tasks cannot involve pages, functions, or information that do not appear in the trajectory screenshots
3. **Completability**: Ensure tasks are 100% completable within the explored interface scope

## Task Format Requirements

### Standard Structure
Each task description MUST follow this format:
"Open {app_name} app, [prerequisite actions if needed], navigate to [specific page name], then [specific action/operation] to complete [specific objective]"

### Required Components
1. **App Opening**: "Open {app_name} app"
2. **Prerequisites** (if needed): "first [create/configure/setup required data/conditions]"
3. **Navigation**: "navigate to [exact page name visible in screenshots]"
4. **Actions**: Clear description of operations to perform on that page
5. **Objective**: Specific final goal to be achieved

### Self-Contained Requirement
All tasks MUST be executable on a fresh, unused app without pre-existing data or configurations:
- Include ALL prerequisite setup actions in the task description
- Create any necessary data/contacts/settings before using them
- Make tasks completely independent and executable from a clean app state

## Completion Condition Standards

### Quality Requirements
Each completion condition MUST be:
- **Specific**: Clearly define what success looks like
- **Measurable**: Include exact values, states, or visible elements
- **Verifiable**: Confirmable by checking UI state or displayed information
- **Achievable**: Completable in a reasonable number of steps
- **Logically Related**: Directly verify the task actions performed

### Logical Consistency Rule
The completion condition MUST directly validate the actions performed in the task description. AVOID:
- Unrelated visual elements or content not affected by task actions
- Geographic content, maps, or data that appears regardless of task actions
- Generic UI states that don't specifically validate task completion
- Side effects or secondary changes not directly caused by task operations

## Prohibited Patterns

### Task Description Patterns to AVOID
❌ "Switch between options multiple times to test stability"
❌ "Systematically examine all available settings"
❌ "Compare different configurations and analyze differences"
❌ "Explore various features to understand functionality"
❌ "Test the stability of different modes"
❌ "Perform comprehensive evaluation of options"

### Completion Condition Patterns to AVOID
❌ "Map view displays West African region" (when task is about storage settings)
❌ "Homepage shows recommended content" (when task is about account settings)
❌ "Gallery displays photos" (when task is about camera configuration)
❌ "List shows default items" (when task is about creating specific content)

## Required Patterns

### Task Description Patterns to GENERATE
✅ "Change setting X from value A to value B and save the configuration"
✅ "Create item Y with specific properties Z and add it to category W"
✅ "Navigate to page X, enable feature Y, and verify the confirmation message appears"
✅ "Configure option X to specific value Y and confirm the change is applied"

### Completion Condition Patterns to GENERATE
✅ "Storage setting shows 'Multiuser storage 1' as selected option"
✅ "Contact 'John Smith' appears in contacts list with email field populated"
✅ "Camera settings page displays 'Resolution: 4K' in the video section"
✅ "New note titled 'Meeting Notes' is visible in notes list with checklist icon"

## Example Tasks with Prerequisites

**Contact Management:**
"Open Contacts app, first create a new contact named 'John Smith' with phone number 123-456-7890, then navigate to John Smith's contact details page and add email address john@example.com to complete contact information update"

**Navigation Setup:**
"Open OsmAnd app, first download the Egypt map through the Downloads page, then navigate to the Map page and enable offline mode to complete offline navigation setup"

**Photo Editing:**
"Open Camera app, first take a test photo in the Camera mode, then navigate to the Gallery page and apply a sepia filter to the test photo to complete photo editing"

**Note Enhancement:**
"Open Notes app, first create a new note titled 'Meeting Notes' with some sample content, then navigate to the note editing page and add a checklist item to complete note enhancement"

## Consistency Examples

### Good Example
- **Task**: "Change storage from External storage 1 to Multiuser storage 1"
- **Completion**: "Data storage folder page shows 'Multiuser storage 1' as the selected option"

### Bad Example
- **Task**: "Change storage from External storage 1 to Multiuser storage 1"
- **Completion**: "Map view displays West African region with country boundaries visible"
- **Problem**: Map content is unrelated to storage setting changes

## Output Format
Please respond strictly in the following JSON array format:

```json
[
  {{"task_description": "<{difficulty_level} difficulty task description 1 following the required format with prerequisites>", "completion_condition": "<Specific, measurable, and logically related completion condition for task 1>", "difficulty_level": "<{difficulty_level}>"}},
  {{"task_description": "<{difficulty_level} difficulty task description 2 following the required format with prerequisites>", "completion_condition": "<Specific, measurable, and logically related completion condition for task 2>", "difficulty_level": "<{difficulty_level}>"}},
  {{"task_description": "<{difficulty_level} difficulty task description 3 following the required format with prerequisites>", "completion_condition": "<Specific, measurable, and logically related completion condition for task 3>", "difficulty_level": "<{difficulty_level}>"}},
  {{"task_description": "<{difficulty_level} difficulty task description 4 following the required format with prerequisites>", "completion_condition": "<Specific, measurable, and logically related completion condition for task 4>", "difficulty_level": "<{difficulty_level}>"}},
  {{"task_description": "<{difficulty_level} difficulty task description 5 following the required format with prerequisites>", "completion_condition": "<Specific, measurable, and logically related completion condition for task 5>", "difficulty_level": "<{difficulty_level}>"}}
]
```

"""

# INPUTS: new_task_description, difficulty_level, trajectory_screenshots (implicit)
TASK_FEASIBILITY_EVALUATOR = """Given task '{new_task_description}' (original difficulty level: {difficulty_level}) and a series of exploration trajectory screenshots, please evaluate whether this task can be completed within the provided trajectory scope AND reassess the actual difficulty level of the task.

## Evaluation Framework

### 1. Task Format Validation
Verify the task follows the required structure:
- Contains "Open [App Name] app" instruction
- Includes all necessary prerequisite setup actions (if applicable)
- Specifies "navigate to [specific page name]" with exact page name
- Describes detailed actions to perform on that specific page
- Includes clear completion objective

### 2. Self-Contained Task Verification
Assess task independence and completeness:
- Task is completely self-contained and executable on a fresh app
- All prerequisite data/contacts/settings are created within the task itself
- No external dependencies or pre-existing data assumptions
- Task includes setup steps for any required data before using it

### 3. Complete Workflow Feasibility
Verify technical execution capability:
- ALL steps (including prerequisites) can be completed using interfaces shown in screenshots
- Prerequisite creation steps are visible and achievable in the trajectory
- Main task actions can be performed after prerequisites are established
- Complete end-to-end workflow is supported by available UI elements

### 4. Fresh App Execution Validation
Confirm clean-state compatibility:
- Task can be successfully executed starting from an empty, unused app state
- Task does not rely on any pre-existing user data or configurations
- All necessary setup actions are included before main operations

### 5. Completion Condition Quality Assessment
Evaluate completion criteria effectiveness:
- Completion condition is specific, measurable, and verifiable
- Can be confirmed by checking specific UI states or displayed information
- Avoids vague conditions requiring subjective judgment
- Avoids repeated testing without clear endpoints

### 6. Task-Completion Logical Consistency
Verify logical alignment between task and success criteria:
- Completion condition directly relates to and validates the task actions performed
- Completion condition verifies specific changes/actions described in the task
- Avoids completion conditions that would be true regardless of task execution
- Rejects unrelated visual content (e.g., map regions when changing settings)

## Prohibited Patterns to Reject

### Task Quality Issues
❌ Tasks involving "multiple times," "systematically," or "comprehensive" testing without specific endpoints
❌ Tasks requiring "comparison" or "analysis" without concrete actionable outcomes
❌ Tasks with completion conditions like "test stability" or "explore functionality"
❌ Tasks requiring subjective evaluation or repeated operations without clear success criteria

### Logical Consistency Issues
❌ Tasks where completion conditions include unrelated visual content
❌ Tasks with completion conditions that would be true regardless of task execution
❌ Tasks where success criteria don't specifically validate the performed actions

## Required Patterns to Accept

### High-Quality Task Characteristics
✅ Tasks with specific, measurable completion conditions
✅ Tasks that create specific items/configurations and verify their creation
✅ Tasks that change settings to specific values and confirm the changes
✅ Tasks with concrete success indicators visible in the UI

### Logical Consistency Requirements
✅ Tasks where completion conditions directly validate the actions performed
✅ Tasks with completion conditions that can only be true if the task was successfully executed
✅ Tasks where success criteria specifically verify the task's intended changes

## Evaluation Guidelines

### Critical Assessment Areas
1. **Interface Availability**: Examine all trajectory screenshots to understand available functionalities
2. **Workflow Completeness**: Verify complete task workflow (prerequisites + main actions) is achievable
3. **UI Element Coverage**: Confirm all UI elements needed for setup and main actions are present
4. **Independence Verification**: Assess task's self-contained nature and fresh app executability
5. **Success Criteria Quality**: Critically evaluate completion condition specificity and verifiability
6. **Logical Alignment**: Critically evaluate completion condition relevance to task actions

### Decision Framework
- **Accept**: Only tasks that meet ALL evaluation criteria
- **Reject**: Any task failing one or more critical requirements
- **Priority**: Logical consistency issues are grounds for immediate rejection

## Response Requirements

Provide detailed analysis covering:
1. Task format and self-contained nature assessment
2. Complete workflow feasibility including prerequisites
3. Fresh app execution capability evaluation
4. UI element availability for entire task flow
5. Completion condition clarity and verifiability analysis
6. Task-completion logical consistency verification
7. **Difficulty level reassessment based on actual task complexity**

### Difficulty Level Assessment Guidelines
Reassess the task difficulty based on the following criteria:
- **Simple tasks (low_level)**: 1-2 steps
- **Medium difficulty tasks (medium_level)**: 3-4 steps
- **Complex tasks (high_level)**: 5+ steps

If rejecting, clearly specify whether due to:
- Unclear completion conditions
- Vague requirements
- Prohibited patterns
- Completion conditions that don't logically relate to task actions

## Output Format
Please respond strictly in the following JSON format:

```json
{{
  "can_complete": <true|false>,
  "confidence": <0.0-1.0>,
  "reasoning": "<Detailed analysis based on trajectory screenshots, addressing all six evaluation criteria. If rejecting, clearly state the primary reason: unclear completion conditions, vague requirements, prohibited patterns, OR completion conditions that don't logically relate to the task actions.>",
  "reassessed_difficulty": "<low|medium|high>",
  "difficulty_change_reason": "<Explanation if the reassessed difficulty differs from the original {difficulty_level} level. If same, explain why the original assessment was correct.>"
}}
```

"""

# INPUTS: screenshot, ui_elements_desc, activity_name
PAGE_TYPE_CLASSIFIER = """Analyze this Android app page and determine its page type.

Page Information:
- Activity Name: {activity_name}
- Visible UI Elements: {ui_elements_desc}

Please select the most matching page type from the following:
- main_page: App main page/homepage
- settings_page: Settings page
- profile_page: Personal info/account page
- login_page: Login/registration page
- search_page: Search page
- list_page: List/catalog page
- detail_page: Detail page
- edit_page: Edit/modify page
- help_page: Help/support page
- about_page: About/version page
- custom_page: Other special pages

Please respond strictly in the following JSON format:
{{
  "page_type": "<page_type>",
  "confidence": <confidence_0.0-1.0>,
  "key_features": ["feature1", "feature2", "feature3"],
  "reasoning": "<reasoning>"
}}
"""

# INPUTS: screenshot, page_features
PAGE_SIMILARITY_CALCULATOR = """Compare the similarity of two Android app pages.

Please analyze the two page screenshots provided and determine whether they are different states of the same page or completely different pages.

Consider the following factors:
1. Page layout structure
2. Main UI element positions
3. Page title and navigation
4. Function button distribution
5. Content area organization

Please respond strictly in the following JSON format:
{{
  "similarity_score": <similarity_score_0.0-1.0>,
  "is_same_page": <true|false>,
  "reasoning": "<reasoning>",
  "key_differences": ["difference1", "difference2"]
}}
"""
