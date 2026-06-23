# MobileGym-Critic Pipeline 说明

本文档详细介绍 MobileGym-Critic 的整体流水线、输入输出规范、核心模块以及常见使用方式，帮助研发与评测同学快速理解并扩展该系统。

## 1. 总体概览

MobileGym-Critic 用于对移动 GUI 任务的执行轨迹进行自动化质量评估。其核心目标是：

1. 解析代理执行日志与截图，构建结构化的可视化证据。
2. 通过视觉语言模型（VLM）和大模型评判任务是否完成，以及每一步的合理性。
3. 汇总评估结果，并回填到结果目录与汇总 CSV 中，产出辅助分析的多种工件。

整体流程可概括为：

```
结果目录 (log.json + screenshots)
        ↓
动作可视化与拼图生成
        ↓
并行 VLM 步骤描述
        ↓
最终判定
        ↓
结果写回 + JSON 摘要 + 日志
```

## 2. 目录结构与输入约定

MobileGym-Critic 需要一个符合 MobileForge 结果格式的目录作为输入。默认入口函数为 `mobilegym_critic/evaluator.py` 中的 `mobilegym_critic_evaluator`。关键约定如下：

- **结果根目录 (`result_dir`)**  
  每个任务以 `task_identifier/` 命名，内部包含执行尝试。若启用了 `agent` 或 `attempt_num` 参数，则路径会拼接对应子目录。
- **任务目录内容**  
  - `log.json`：逐步动作日志，记录 `step` 编号、`action`、`detail` 等信息。
  - 连续命名的截图 `0.png`, `1.png`, …：每一步动作前后的屏幕状态。
  - （可选）预生成的 `visualize_actions/`、`single_actions/` 等目录；若不存在 MobileGym-Critic 会自动创建。
- **结果 CSV (`utils.get_results_csv_path`)**  
  通过 `mobilegym_critic/utils/data.get_dataset` 读取，要求首列为 `task_identifier` 索引。评估结果会回写该 CSV。

确保结果目录具备上述文件后，即可运行评估流程。

## 3. 评估流水线阶段

### 3.1 动作可视化与拼图构建

调用 `visualize_and_save_actions`（见 `mobilegym_critic/utils/visualize_actions.py`）完成：

1. 读取 `log.json`，并匹配对应的截图序列。
2. 为每一步生成两类图片：
   - `visualize_actions/step_{n}.png`：拼接“前/后”截图并覆盖动作标记、文字说明。
   - `single_actions/step_{n}.png`：仅使用动作发生前的截图，叠加红圈/绿框标记与动作详情文字，方便后续做个别步骤的精细复盘或衍生可视化。
3. 生成 `puzzle/puzzle.png`：以网格方式组合所有步骤截图，标题包含任务描述，方便模型快速回顾整个流程。

若后续生成了 LLM 步骤描述，还可调用 `create_llm_puzzle` 构建 `puzzle_llm.png`，将模型描述直接渲染到拼图图片中。

### 3.2 并行生成步骤描述

核心函数 `_generate_description_for_step` 会对每个步骤调用 VLM，将原始动作日志转化为结构化语义描述。详细过程如下：

1. **Prompt 选择**  
   - 若当前步骤不是最后一步，调用 `get_describe_step_prompt`；若是最后一步，则使用 `get_describe_final_step_prompt`，避免在终态画面中过度依赖“After Action”子图。
2. **模型调用**  
   - 通过 `mobilegym_critic/utils/vivo/vivo_api.inference_chat_gemini_1_image` 访问默认配置的 Gemini 图像模型（在 `utils/vivo/llm_config.py` 中定义），每次只传入对应的动作截图。
3. **结果解析**  
   - 模型输出为 JSON 字符串或包含 `content` 字段的字典。解析后提取 `action_description`、`ui_description`，并附带 `_usage_info`（token、费用）与 `_model_info`（模型名称、提供方）。
   - 同时记录 `_raw_action`，保留原始日志中的动作类型与详细参数，便于后续决策阶段引用。
4. **并行加速**  
   - 采用 `ThreadPoolExecutor(max_workers=8)` 并行触发多个步骤的描述生成，在高阶任务下显著缩短耗时。
5. **审计与落盘**  
   - 所有 Prompt、模型原始响应均写入 `prompt_logs.json`，确保评估可追溯；每个步骤的结果则写入 `step_descriptions`（字典结构，以步骤号为 key）。

典型返回对象示例：

```json
{
  "action_description": "点击了页面底部的“保存”按钮。",
  "ui_description": "屏幕显示笔记编辑界面，下方有保存按钮。",
  "_usage_info": {"prompt_tokens": 512, "completion_tokens": 168, "total_tokens": 680},
  "_model_info": {"model": "gemini-pro-vision", "provider": "google", "api_cost": 0.024},
  "_raw_action": {"type": "click", "detail": "按钮坐标 (812, 1914)"}
}
```

### 3.3 最终判定

最终判定通过一次大模型调用完成。`mobilegym_critic_evaluator` 会使用 `get_final_decision_prompt` 组合任务描述、步骤描述清单以及末尾 3 张截图（来自 `puzzle/puzzle.png`），并通过 `inference_chat_gemini_1_image` 请求模型给出综合评审。模型需输出：
- `decision`: 1（成功）、0（失败）
- `reason`: 判定理由
- `failure_step`:（若失败）关键错误步骤
- `reasonable_steps` / `unreasonable_steps` / `step_analysis`: 对每一步的合理性标签与影响说明
- `task_feasible` / `task_feasible_reason` / `task_barriers`: 任务可行性评估

评估器会对响应进行如下处理：

1. **JSON 解析与校验**  
   - 使用 `parse_json_from_response` 从模型回复中提取 JSON，若解析失败则判为异常流程并写入错误日志。
2. **字段落盘**  
   - 将 `final_decision_response`、`failure_step`、`step_analysis` 等写入 `evaluation_detail`。
   - 统计判定阶段的 token/费用信息（如返回体带 `usage` 字段），方便预算核对。
3. **结果回写**  
   - 调用 `utils.save_result__completed_evaluation` 更新结果 CSV 中对应任务行，记录决策、理由、步骤分析与可行性结论。
   - 将原始决策 JSON 保存到 `final_decision.json`，并生成 `evaluation_summary.json` 作为整体摘要。

所有判定交互同样记录到 `prompt_logs.json`，并将模型响应保存为 `final_decision.json`。

### 3.4 结果汇总与统计

判定完成后，`mobilegym_critic_evaluator` 会：

1. 将最终决策信息写回 `evaluation_detail`，并整理 `final_decision_data`。
2. 分别统计步骤描述与最终判定阶段的 token 使用与 API 成本，写入 MobileForge 结果 CSV（调用 `utils.save_result__completed_evaluation`）。
3. 生成 `evaluation_summary.json`，包含任务描述、最终结果、理由、步骤详情与合理性分析。
4. 若评估过程中出现异常（缺失目录、无法解析 JSON、模型错误等），会在对应阶段捕获并保存错误信息，确保流程不中断。

## 4. 运行方式与参数

### 4.1 命令行入口

`evaluator.py` 支持命令行调用：

```bash
python mobilegym_critic/evaluator.py \
  --task_identifier 251021115900.177490_977DF8DF3ACA4E36-002 \
  --result_dir /path/to/session-debug \
  --mode eval \
  --agent GUIOWL \
  --attempt_num 1 \
  --reasoning_mode direct \
  --action_mode with_action
```

关键参数说明：

- `task_identifier`：目标任务 ID（默认指向示例任务）。
- `result_dir`：MobileForge 结果根目录。
- `mode`：路径拼接策略，`eval` 忽略 `agent`，`full` 会将 `agent` 插入路径。
- `agent`：代理名称。
- `attempt_num`：尝试次数目录命名，如 `attempt_1`。
- `reasoning_mode` / `action_mode`：留作评估控制参数。

### 4.2 批量评估建议

在批处理场景下，可自行编写脚本：

1. 读取待评估的任务列表（例如从 CSV 中筛选特定 batch）。
2. 逐个调用 `mobilegym_critic_evaluator`，或使用线程池控制并行度。
3. 监控返回值，若为 -1 说明出现异常，需要人工检查结果目录与日志。

## 5. 模型与服务配置

- **模型配置**  
  默认使用 `utils/vivo/llm_config.py` 中的 `DEFAULT_DESC_MODEL` / `DEFAULT_DECISION_MODEL` 等设定，可根据需要替换为企业内部 VLM。所有推理接口封装在 `utils/vivo/vivo_api.py`。

- **鉴权与环境变量**  
  模型服务的凭证通常读取自环境变量或 `auth_util.py` 的封装，部署前需确保访问凭证可用。

- **Token & 成本监控**  
  MobileGym-Critic 会在 `step_descriptions` 与最终判定阶段分别记录 token 消耗与 API 成本，方便后续做预算或性能分析。

## 6. 输出与工件

每个任务评估完成后，目标目录会出现以下关键文件与目录：

- `visualize_actions/`：带有前后对比、动作标记的拼接图。
- `single_actions/`：突出单步动作的截图。
- `puzzle/puzzle.png`：完整步骤拼图。
- `llm_described_actions/` 与 `puzzle/puzzle_llm.png`（可选）：嵌入步骤描述的图像。
- `prompt_logs.json`：所有模型调用的请求与响应。
- `final_decision.json`：最终判定详情。
- `evaluation_summary.json`：整合的评估摘要。

结果 CSV 中会新增（或覆盖）当前任务的评估字段，包括最终结论、失败步骤、合理性分析、token 使用等。

## 7. 常见问题与排查

| 问题现象 | 排查步骤 |
| --- | --- |
| `Target directory not found` | 确认 `result_dir/task_identifier[/agent]/[attempt_n]` 路径存在。 |
| 缺少 `log.json` 或截图 | 核对代理执行流程是否产出完整日志；必要时重新生成结果。 |
| 模型响应无法解析 JSON | 查看 `prompt_logs.json`，定位具体阶段，检查响应格式是否符合预期；必要时调整提示或添加更严格的正则解析策略。 |
| VLM 请求失败 | 检查 `utils/vivo/vivo_api.py` 的日志输出与鉴权配置。 |
| Token 统计缺失 | 确认模型返回体中是否包含 `usage` 字段，或在 `llm_config.py` 中开启相应支持。 |

## 8. 扩展建议

- **定制提示词**：可在 `utils/prompts.py` 中修改模板细节，以适配新的评估标准。
- **替换模型**：在 `utils/vivo/llm_config.py` 中更新默认模型名与提供方；必要时调整返回字段解析逻辑。
- **新增指标**：在 `mobilegym_critic_evaluator` 中扩展统计字段，并在 `utils.save_result__completed_evaluation` 处写入 CSV。
- **可视化/报表**：结合 `evaluation_summary.json` 与 `prompt_logs.json`，可以生成更丰富的分析看板。

## 9. Prompt 模板详解

核心 Prompt 均定义在 `mobilegym_critic/utils/prompts.py`，所有调用都会在任务目录生成 `prompt_logs.json`，记录完整的系统/用户 Prompt 及模型回复，便于调试与审计。下列模板均可按需调整，其中花括号字段在运行时动态替换。

### 9.1 `get_describe_step_prompt`（非最终步骤）

```text
SYSTEM PROMPT:
You are an expert mobile device assistant. Your task is to analyze a two-panel image showing the 'Before Action' and 'After Action' state of a user's workflow. Your analysis must focus *only* on the 'Before Action' panel (the left side). You must output your response in a JSON format.

USER PROMPT:
The overall task is: '{task_description}'.

## Input Analysis
The provided image shows a 'Before Action' state on the left and an 'After Action' state on the right. Your entire analysis should focus on the left 'Before Action' panel.

**Note:** If the 'After Action' panel is identical to the 'Before Action' panel, it signifies this is the final action in the task.

On the left panel, a user action is visualized with markers: a red circle shows the click/touch point, surrounded by a green square, with a 'C' label in the corner. The raw action from the execution log is provided for context:
- Action Type: `{log_action}`
- Action Detail: `{log_detail}`

## Your Task
Based on the visual evidence in the **left panel** and the provided log context, perform the following two tasks:
1. **action_description**: ...
2. **ui_description**: ...

Your output MUST be a JSON object with these two keys.
```

占位符说明：
- `{task_description}`：来自结果 CSV 的任务说明。
- `{log_action}` / `{log_detail}`：对应步骤的原始动作类型与细节。

### 9.2 `get_describe_final_step_prompt`（最终步骤）

与 9.1 类似，但面向单张终态截图。系统 Prompt 指示模型只关注“最终动作后的屏幕状态”，并提醒若 `Action Detail` 已表明任务结束，需要描述“用户执行了怎样的动作”而非直接宣判完成。

### 9.3 `get_final_decision_prompt`（最终判定）

该模板由系统 Prompt 与用户 Prompt 组成。系统部分会附带 `_get_evaluation_guidelines()` 输出的准则；用户部分注入任务描述、步骤描述与组合截图。核心片段如下：

```text
SYSTEM PROMPT:
You are an expert in evaluating mobile UI automation tasks.
## Evaluation Guidelines
1. **Final UI State**: ...
...

USER PROMPT:
Task Description: '{task_description}'

Here is a step-by-step breakdown of the agent's actions, including both raw logs and descriptions generated by a Vision Language Model (VLM):
{formatted_steps}

You are now provided with a composite image of the last 3 screenshots. Note that this is only a partial view of the execution. ...
**TASK FEASIBILITY ASSESSMENT**: ...
**CRITICAL WARNING ABOUT TEXT DESCRIPTIONS**: ...
**MANDATORY VERIFICATION**: ...
**FAILURE STEP TRACKING**: ...
**STEP REASONABLENESS ANALYSIS**: ...
**TASK FEASIBILITY OUTPUT**: ...
```

模型需要输出包含 `decision`、`reason`、`failure_step`、`reasonable_steps`、`unreasonable_steps`、`step_analysis` 以及任务可行性字段的 JSON，解析方式见第 3.3 节。

### 9.4 `get_task_feasibility_prompt`

用于单独判断任务是否具备完成条件，输出格式为：

```json
{
  "feasible": true,
  "reason": "...",
  "barriers": []
}
```

默认流程会在最终判定的提示词中嵌入可行性说明，如需拆分评估可直接调用该模板获得更细粒度的结论。

---

通过以上各阶段的协作，MobileGym-Critic 能够在保证解释性的同时，实现对移动 GUI 代理任务的高质量自动化评估。若在使用或二次开发过程中遇到更多问题，欢迎继续完善本文档。欢迎提 PR！***

