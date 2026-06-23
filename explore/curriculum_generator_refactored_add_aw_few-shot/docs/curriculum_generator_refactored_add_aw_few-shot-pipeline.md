## 重构版 Curriculum Generator 运行流水线（main.py 全流程详解）

本文档详解 `curriculum_generator_refactored_add_aw_few-shot/main.py` 的运行流程、输入输出、并行策略、调试产物与目录结构，便于理解整体 pipeline 并在实践中高效使用与排错。

### 目标概述
- 基于探索阶段产出的可视化轨迹数据（screenshots + steps 元信息），对每条轨迹进行：
  - 合理性/完成度评估
  - 结合 AndroidWorld 风格与通用 few-shot 示例生成一批高质量训练任务
  - 估计任务所需步骤数
- 在应用级别并行处理多个应用；在应用内按照轨迹顺序串行处理（支持重试）
- 完整保存可视化图片、评估/生成结果、few-shot 使用、LLM 调用与令牌统计等调试信息，最终汇总输出

---

### 关键依赖模块
- `TrajectoryParser`（来自 `curriculum_generator_refactored/trajectory_parser.py`）
  - 解析应用基本信息、读取应用下所有轨迹文件、解析单条轨迹 JSON、获取轨迹截图路径等。
- `ActionVisualizer`（来自 `curriculum_generator_refactored/action_visualizer.py`）
  - 将每步动作叠加到对应截图上，输出“动作可视化截图”（带点击/滑动标注等）。
- `UnifiedTaskProcessor`（来自 `curriculum_generator_refactored/unified_task_processor.py`）
  - 输入：应用名、原始目标、可视化截图序列、few-shot 示例、任务生成原则、已存在任务列表
  - 输出：evaluation（任务合理性、是否完成等）与 generated_tasks（新任务清单）
  - 提供 `get_token_usage()` 与 `get_last_llm_debug()`，用于保存调用/令牌统计与 LLM 调试信息。
- `RefactoredResultSaver`（来自 `curriculum_generator_refactored/result_saver.py`）
  - 保存应用级结果与生成主汇总报告。
- `AndroidWorldTaskLoader`（本目录 `android_world_loader.py`）
  - 从 `251103-android-world-tasks-to-app.xlsx` 读取 AndroidWorld 任务，按包名/应用名匹配 few-shot。

---

### 命令行参数
- `--vis_data_dir` 可视化数据目录（默认：`./exploration_output_vis`）
- `--app_package` 处理的应用列表（可为多个；默认 `all` 表示处理目录下的所有应用）
- `--output_dir` 输出目录名（最终会加上运行时 timestamp，并放入 `generated_tasks/` 下）
- `--limit` 限制每个应用内处理的轨迹数量
- `--parallel_workers` 应用级并行的最大工作线程数（默认 10；`all` 模式会自动使用应用数）
- `--max_retries` 单条轨迹失败后的最大重试次数（默认 1000）

示例：

```bash
python curriculum_generator_refactored_add_aw_few-shot/main.py \
  --vis_data_dir /path/to/exploration_output_vis_25102901 \
  --app_package com.example.app1 com.example.app2 \
  --output_dir generated_tasks_refactored \
  --limit 20 \
  --parallel_workers 4 \
  --max_retries 5
```

---

### 顶层执行流程
1. 初始化 `RefactoredCurriculumGenerator`
   - 记录 `session_timestamp`
   - 加载“通用 few-shot 示例”（3 个内置样例）
   - 初始化 `AndroidWorldTaskLoader`（加载 Excel：`251103-android-world-tasks-to-app.xlsx`）
   - 加载“任务生成原则”（AndroidWorld 风格优先、滚动/长按/问答能力等覆盖）
   - 初始化线程安全的统计结构 `progress_stats`
2. 解析命令行参数并检查 `--vis_data_dir`
3. 计算输出目录：`generated_tasks/<output_dir>_<session_timestamp>`
4. 解析应用列表
   - `all` 模式：扫描 `vis_data_dir` 下的所有子目录作为应用包名
   - 指定模式：直接使用传入包名
5. 配置应用级并行度
   - `all` 模式：worker 数 = 应用数
   - 指定模式：worker 数 = min(应用数, `--parallel_workers`)
6. 为每个应用提交并行任务（线程池）
   - 每个任务调用 `process_app_trajectories(app_package, timestamped_output_dir, limit, max_retries)`
7. 等待全部应用处理完成后，调用 `result_saver.create_master_summary()` 生成主汇总报告

---

### 应用内处理流程（顺序执行）
入口：`process_app_trajectories(app_package, output_dir, limit, max_retries)`

1. 解析应用信息：`TrajectoryParser.parse_app_info(app_package)`
2. 获取该应用下所有轨迹：`TrajectoryParser.get_all_trajectories(app_package)`；若配置了 `limit` 则截断
3. 初始化进度统计 `progress_stats`
4. 初始化去重用列表 `generated_instructions = []`
5. 遍历每条轨迹（顺序执行），每条轨迹带重试：
   - 调用 `_process_single_trajectory(...)`
   - 若成功：
     - 累加结果到 `all_results`
     - 将生成任务里的 `instruction` 追加到 `generated_instructions`（用于后续去重/防重复）
     - `progress_stats` 中 `processed += 1`
   - 若失败：
     - 若未超重试次数：等待片刻重试
     - 最终失败：`progress_stats` 中 `failed += 1`
   - 每条轨迹结束打印阶段性进度（完成数、成功率、ETA）
6. 打印最终统计（总数、成功/失败、成功率、总用时、平均用时）
7. 调用 `result_saver.save_app_results(...)` 将应用级结果落盘

---

### 单条轨迹处理细节
入口：`_process_single_trajectory(app_package, app_name, trajectory_file, generated_instructions)`

1. 解析轨迹 JSON
   - 获取 `trajectory_id / goal / depth / steps`
2. 准备调试目录
   - 路径：`generated_tasks/debug_output_<session_timestamp>/<app_package>/<trajectory_id>/`
3. 获取截图信息并生成“动作可视化截图”
   - `TrajectoryParser.get_trajectory_screenshots(...)`
   - `ActionVisualizer.create_visualized_screenshots(...)`
   - 保存到 `visualized_screenshots/`
     - `step_XX_visualized.png`
     - `step_XX_info.json`（记录 step 索引、动作类型、摘要、坐标、目标元素等）
4. 构造 few-shot 示例（按“AndroidWorld 优先 + 通用示例补齐”的策略）
   - AndroidWorld app 相关示例：`AndroidWorldTaskLoader.get_app_fewshot_examples(app_package, app_name, max_examples=None)`
   - 若不足 3 条，则从内置的“通用 few-shot 示例”补足到 3 条
5. 统一处理器评估与生成
   - `UnifiedTaskProcessor.process_task(...)`
     - 输入：`app_name / original_goal / visualized_screenshots / fewshot_examples / task_principles / existing_tasks`
     - 输出：
       - `evaluation`：如 `task_reasonable`（是否合理）、`task_completed`（是否完成）等
       - `generated_tasks`：新任务列表（含 `instruction`、`estimated_steps` 等）
6. 保存调试信息
   - `unified_processing/`
     - `unified_result.json`：完整输入输出
     - `evaluation_result.json`：评估子集
     - `generated_tasks.json`：生成任务清单
     - `token_usage.json`：令牌使用统计（来自 `UnifiedTaskProcessor.get_token_usage()`）
     - `fewshot_examples.json`：实际用到的 few-shot 示例（若存在）
     - `llm_call_debug.json`：最近一次 LLM 调用的详细调试数据（如由处理器提供）
   - `android_world/`
     - `app_specific_tasks.json`：AndroidWorld 匹配到的 app 相关任务
     - `app_match_info.json`：匹配统计（包名、应用名、条数等）
7. 返回该轨迹的聚合结果字典（供应用级保存）

备注：文件中还提供了 `_process_single_trajectory_parallel(...)` 的线程安全版本，逻辑基本一致，主要用于需要在轨迹级并行时的替代方案（当前主流程未启用）。

---

### 任务生成原则（核心要点）
代码中加载的原则强调“AndroidWorld 风格优先”，关注以下考察面向：
- 覆盖 AndroidWorld 展示的核心功能形态（如通用动作空间测试、问答类读取信息、滚动/拖拽、长按）
- 参数要贴合截图环境（如日期/联系人等需按实际可见数据自适应）
- 无参数的通用功能（如开关蓝牙）可直接复制
- 仅生成截图确实可达/可见的功能任务（相信 AndroidWorld 能力描述，但同时受限于当前轨迹数据）
- 注重功能覆盖而非参数变化
- 明确指定目标应用
- 用目标导向方式描述任务，不写具体操作路径
- 任务复杂度涵盖 1~40 步，参考 AndroidWorld 步长模式进行估计

---

### LLM 调用 Prompt 全文
`UnifiedTaskProcessor._build_unified_prompt` 会依据应用名、原始目标、few-shot 示例、任务原则与已有任务列表构造以下完整 Prompt（`existing_tasks_text` 部分仅在存在历史任务时追加）：

````text
You are an expert Curriculum Generator - a teacher designing comprehensive learning tasks for GUI agents. Your core purpose is to create a complete curriculum covering all app functionalities with progressive difficulty levels to systematically teach GUI agents how to use the target app.

As a Curriculum Generator, your job is to:
1. EVALUATE the original task for reasonableness and completion  
2. GENERATE new diverse curriculum tasks that comprehensively cover the app's functionality

## App Information
App Name: {app_name}
Original Task Goal: {original_goal}

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

### Step 2: Curriculum Task Generation
As a curriculum designer, generate 3-8 new learning tasks that:
- Systematically cover different core functionalities of {app_name}
- Have varying step lengths (1-40 steps) representing different complexity levels
- Are pedagogically sound for teaching GUI agents
- Avoid redundancy with existing tasks (if any listed above)
- Focus on core functionality variations rather than just parameter changes
- For the same functionality, limit to at most 3 different parameter variations
- Ensure comprehensive coverage of the app's feature set as much as possible

## Output Format
IMPORTANT: Keep step_quality_analysis concise. Only analyze 3-5 key representative steps, not every single step.
Provide your response in the following JSON format:

```json
{
    "evaluation": {
        "task_reasonable": true/false,
        "task_completed": true/false,
        "reasonableness_explanation": "Brief explanation (1-2 sentences)",
        "completion_explanation": "Brief explanation (1-2 sentences)",
        "confidence_score": 0.0-1.0,
        "step_quality_analysis": {
            "total_steps": 10,
            "reasonable_steps": 8,
            "trajectory_quality_score": 0.8,
            "quality_summary": "Brief assessment (1-2 sentences)"
        }
    },
    "generated_tasks": [
        {
            "task_id": "task_1",
            "instruction": "Self-contained learning task for {app_name} that teaches specific functionality",
            "estimated_steps": 5,
            "core_functionality": "Main functionality being taught",
            "variation_type": "simplification/parameter_change/scenario_application/step_progression",
            "prerequisites": "Only specify non-obvious prerequisites; avoid basic assumptions"
        }
    ]
}
```

IMPORTANT: The estimated_steps should be between 1-40 steps, where the step count represents the task complexity level.

Now analyze the provided app, task, and screenshots to generate your curriculum evaluation and new learning tasks.
````

说明：
- `{fewshot_text}` 会拼接最多 100 个 few-shot 示例；当示例缺少步骤信息时，仅包含任务指令。
- `{existing_tasks_text}` 在已有任务时列出全部历史指令，并强调避免重复生成。
- Prompt 将在 `gpt4v_call` 中搭配最多 15 张动作可视化截图发送给多模态模型。
- `{principles_text}` 为下面列出的 11 条任务生成原则（带编号）：

```
1. AndroidWorld Style Priority: The generated tasks MUST comprehensively cover all examination points and similar tasks found in the AndroidWorld few-shot examples. If AndroidWorld examples show certain functionality, prioritize generating similar tasks that test the same capabilities.
2. Generic Action Space Testing: Follow AndroidWorld patterns for generic actions. If AndroidWorld examples include tasks like 'take a photo' or 'set a timer' without specifying exact app details, generate similar generic tasks that test basic app functionality in a straightforward manner.
3. Agent Answer Space Testing: When AndroidWorld examples include question-answering tasks (e.g., 'What's my schedule for next Saturday afternoon in Simple Calendar app?'), prioritize generating similar tasks that require the agent to read information and provide answers based on current app state.
4. Scroll/Drag Action Space Testing: When AndroidWorld examples involve scrolling or dragging operations, prioritize generating tasks that test the agent's ability to navigate through interfaces using scroll and drag gestures. Examples include: expense tracking tasks requiring scrolling through long lists to find specific categories or tags, calendar apps requiring swiping to navigate between months or weeks, music apps requiring scrolling through playlists, settings apps requiring scrolling to find specific configuration options.
5. Long-press Action Space Testing: When AndroidWorld examples require long-press operations, prioritize generating tasks that test the agent's long-press interaction capabilities. Examples include: audio recording apps requiring long-press to select and rename default file names, expense apps requiring long-press to select and delete multiple expense entries, gallery apps requiring long-press to select multiple photos for batch operations, contacts apps requiring long-press to access contact editing or deletion options.
6. Environment-Adaptive Parameter Usage: For tasks with parameters, adapt to the actual environment visible in screenshots. If AndroidWorld shows 'check yesterday's running duration in OpenTracks' but screenshots only show data from the day before yesterday, generate 'check the day before yesterday's running duration in OpenTracks'. If AndroidWorld shows 'change Li Hua's phone to 177-8888-9999' but screenshots show 'Zhang San' instead, generate 'change Zhang San's phone to 177-8888-9999'.
7. Parameter-Free Task Replication: For AndroidWorld tasks without environment-dependent parameters (e.g., 'turn on Bluetooth'), directly replicate these tasks as they are universally executable functionality.
8. Screenshot-Based Real Functionality: Only generate tasks for functionality actually visible in trajectory screenshots. However, trust that AndroidWorld examples reflect real app capabilities - if AndroidWorld shows certain functionality, the app definitely supports it, but current data might be limited.
9. Core Functionality Comprehensive Coverage: Based on AndroidWorld examples, ensure comprehensive coverage of all core functionalities demonstrated. Don't just create parameter variations - focus on covering different functional areas shown in AndroidWorld tasks.
10. Application Specification: Each task must clearly specify the target application, following AndroidWorld naming conventions (e.g., 'pro expense', 'simple calendar', etc.).
11. Goal-Oriented Task Description: Describe what to achieve, not how to achieve it. Follow AndroidWorld's descriptive style focusing on end goals rather than step-by-step instructions.
12. Step Length Progression: Create tasks with varying complexity (1-40 steps) representing different difficulty levels, with step counts estimated based on AndroidWorld examples and step execution patterns.
```

---

### 输出目录结构与产物
假设参数 `--output_dir=generated_tasks_refactored` 且本次 `session_timestamp=20250101_120000`，则：

- 应用级结果目录（由 `save_app_results` 统一输出）：
  - `generated_tasks/generated_tasks_refactored_20250101_120000/<app_package>/...`
    - 包含应用级汇总 JSON/CSV 等（具体以 `RefactoredResultSaver` 实现为准）
- 调试目录（按轨迹组织）：
  - `generated_tasks/debug_output_20250101_120000/<app_package>/<trajectory_id>/`
    - `visualized_screenshots/`
      - `step_00_visualized.png`
      - `step_00_info.json`
      - ...
    - `unified_processing/`
      - `unified_result.json`
      - `evaluation_result.json`
      - `generated_tasks.json`
      - `token_usage.json`
      - `fewshot_examples.json`（若有）
      - `llm_call_debug.json`（若有）
    - `android_world/`
      - `app_specific_tasks.json`
      - `app_match_info.json`
- 主汇总报告：
  - `generated_tasks/generated_tasks_refactored_20250101_120000/` 下由 `create_master_summary(...)` 生成的汇总文件（例如总体统计、索引等）

---

### 并行与重试策略
- 并行粒度：应用级并行（ThreadPoolExecutor）；应用内轨迹顺序执行
- 并行 worker：
  - `all` 模式：与应用数一致
  - 指定模式：不超过 `--parallel_workers`
- 重试：单条轨迹最多 `--max_retries` 次；失败会短暂 sleep 再试；最终失败计入统计
- 进度：实时输出完成/失败数、成功率、ETA；结束输出总览（总耗时、平均耗时）

---

### 运行前准备
1. 确保 `--vis_data_dir` 指向合法目录，目录结构满足 `TrajectoryParser` 的读取约定：
   - `<vis_data_dir>/<app_package>/...` 下包含轨迹 JSON 与对应截图
2. 确保 Excel `251103-android-world-tasks-to-app.xlsx` 存在于 `curriculum_generator_refactored_add_aw_few-shot/` 目录
3. Python 依赖安装齐全（参考项目 `requirements.txt`）

---

### 常见问题排查
- 找不到应用目录：确认 `--vis_data_dir` 下是否存在对应 `<app_package>` 子目录
- 解析轨迹失败：检查轨迹 JSON 格式、关键字段（`trajectory_id/goal/steps`）是否完整
- 截图缺失：`TrajectoryParser.get_trajectory_screenshots(...)` 要能找到每步对应的图片
- AndroidWorld few-shot 为空：可能匹配不到该应用；会自动使用“通用示例”补齐，非致命
- 无输出/无生成任务：检查 `unified_processing/unified_result.json` 和 `llm_call_debug.json` 了解 LLM 侧的返回与上下文

---

### 最佳实践建议
- 初次运行建议设置较小 `--limit` 与较少应用，验证产物结构与质量
- 打开 `debug_output_<timestamp>` 目录配合 `unified_processing` 与 `visualized_screenshots` 定位问题
- 若需扩充 few-shot，可：
  - 增强 AndroidWorld Excel 的映射内容
  - 扩写 `_load_general_fewshot_examples()` 的通用示例
- 如需轨迹级并行，可参考 `_process_single_trajectory_parallel(...)` 的实现思路进行扩展，但注意 I/O 与互斥写入

---

### 一句话总结
该流水线将探索轨迹转化为“评估 + 任务生成”的标准化流程，应用级并行、轨迹内顺序、强调 AndroidWorld 风格覆盖与可视化/调试产物完备，为后续训练数据构建提供高质量、可追溯的任务集合。 


