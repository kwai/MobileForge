# M3A_vivo_gemini: 基于Vivo Gemini的多模态Android自主代理

## 概述

M3A_vivo_gemini是基于M3A（Multimodal Autonomous Agent for Android）架构的变体，专门使用Vivo Gemini作为底层大语言模型。该代理能够通过视觉和文本输入来理解和执行Android设备上的复杂任务。

## Pipeline流程图

为了更好地理解M3A_vivo_gemini的工作流程，我们提供了一个详细的流程图：

![M3A_vivo_gemini Pipeline](m3a_vivo_gemini_pipeline.drawio)

该流程图展示了以下关键组件：

### 主流程（中央垂直流程）
1. **环境初始化** → **状态获取与预处理** → **动作选择** → **LLM推理** → **动作解析与执行** → **状态更新与总结** → **任务完成判断**

### 关键技术组件（左侧）
- **坐标转换系统**: 处理不同屏幕方向和分辨率
- **UI元素验证**: 确保元素有效性和可见性  
- **安全机制**: 多层安全检查和处理
- **历史管理**: 详细的状态记录和上下文维护

### 支持的动作类型（右上）
- click, long_press, input_text, scroll, open_app, navigate, status, answer

### 错误处理机制（右下）
- 动作解析错误、索引越界错误、执行错误

### 性能指标（左下）
- 成功率、执行时间、Token使用、错误率

## 核心架构

### 1. 代理初始化

M3A_vivo_gemini代理在`benchmark_run.py`中通过以下方式初始化：

```python
elif args.agent == "M3A_vivo_gemini":
    agent = m3a.M3A(env, infer.VivoGeminiWrapper(args.vivo_gemini_api_model))
    screenshot_key = "raw_screenshot"
    grounded_action_key = "action_output_json"
    log_keys = ["action_output", "summary"]
    raw_response_key = ["action_raw_response", "summary_raw_response"]
```

### 2. 核心组件

- **环境接口**: 通过`interface.AsyncEnv`与Android设备交互
- **多模态LLM**: 使用`VivoGeminiWrapper`处理视觉和文本输入
- **动作执行器**: 将高级动作转换为具体的设备操作
- **状态管理器**: 维护代理的历史状态和上下文

## Pipeline详细流程

### 阶段1: 环境初始化

```python
def setup_agent(env):
    """根据参数配置适当的代理"""
    if args.agent == "M3A_vivo_gemini":
        agent = m3a.M3A(env, infer.VivoGeminiWrapper(args.vivo_gemini_api_model))
        # ... 配置相关键值
    return agent, screenshot_key, grounded_action_key, log_keys, raw_response_key
```

### 阶段2: 状态获取与预处理

在每一步中，代理首先获取当前屏幕状态：

1. **获取原始截图**: `state.pixels.copy()`
2. **获取UI元素列表**: `state.ui_elements`
3. **生成UI元素描述**: 通过`_generate_ui_elements_description_list()`函数
4. **添加视觉标记**: 在截图上添加边界框和数字索引

```python
# 获取当前状态
state = self.get_post_transition_state()
logical_screen_size = self.env.logical_screen_size
orientation = self.env.orientation
physical_frame_boundary = self.env.physical_frame_boundary

# 处理UI元素
before_ui_elements = state.ui_elements
before_ui_elements_list = _generate_ui_elements_description_list(
    before_ui_elements, logical_screen_size
)

# 添加视觉标记
for index, ui_element in enumerate(before_ui_elements):
    if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
        m3a_utils.add_ui_element_mark(
            before_screenshot,
            ui_element,
            index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
```

### 阶段3: 动作选择

代理使用以下提示模板生成动作选择提示：

```python
ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}\n\n'
    'Here is a history of what you have done so far:\n{history}\n\n'
    'The current screenshot and the same screenshot with bounding boxes'
    ' and labels added are also given to you.\n'
    'Here is a list of detailed'
    ' information for some of the UI elements...\n{ui_elements}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\nNow output an action from the above list in the correct JSON format,'
    ' following the reason why you do that. Your answer should look like:\n'
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    'Your Answer:\n'
)
```

#### 支持的动作类型

1. **点击操作**: `{"action_type": "click", "index": <target_index>}`
2. **长按操作**: `{"action_type": "long_press", "index": <target_index>}`
3. **文本输入**: `{"action_type": "input_text", "text": <text_input>, "index": <target_index>}`
4. **滚动操作**: `{"action_type": "scroll", "direction": <up, down, left, right>, "index": <optional_target_index>}`
5. **应用打开**: `{"action_type": "open_app", "app_name": <name>}`
6. **导航操作**: `{"action_type": "navigate_home"}` 或 `{"action_type": "navigate_back"}`
7. **状态更新**: `{"action_type": "status", "goal_status": "complete"}` 或 `{"action_type": "status", "goal_status": "infeasible"}`
8. **回答问题**: `{"action_type": "answer", "text": "<answer_text>"}`

### 阶段4: LLM推理

代理调用Vivo Gemini进行多模态推理：

```python
action_output, is_safe, raw_response = self.llm.predict_mm(
    action_prompt,
    [
        step_data['raw_screenshot'],
        before_screenshot,
    ],
)
```

### 阶段5: 动作解析与执行

1. **解析LLM输出**: 使用`parse_reason_action_output()`函数提取原因和动作
2. **转换为JSON动作**: 使用`JSONAction`类解析动作参数
3. **验证索引范围**: 确保UI元素索引在有效范围内
4. **执行动作**: 通过环境接口执行具体动作

```python
# 解析输出
reason, action = m3a_utils.parse_reason_action_output(action_output)

# 转换为JSON动作
converted_action = json_action.JSONAction(
    **agent_utils.extract_json(action),
)

# 执行动作
actual_action_coordinates = self.env.execute_action(converted_action)
```

### 阶段6: 状态更新与总结

动作执行后，代理：

1. **等待屏幕稳定**: `time.sleep(self.wait_after_action_seconds)`
2. **获取新状态**: 重新获取截图和UI元素
3. **生成总结**: 使用LLM比较执行前后的状态

```python
# 获取执行后的状态
state = self.env.get_state(wait_to_stabilize=False)
after_ui_elements = state.ui_elements
after_screenshot = state.pixels.copy()

# 生成总结提示
summary_prompt = _summarize_prompt(
    action,
    reason,
    goal,
    before_ui_elements_list,
    after_ui_elements_list,
)

# 调用LLM生成总结
summary, is_safe, raw_response = self.llm.predict_mm(
    summary_prompt,
    [
        before_screenshot,
        after_screenshot,
    ],
)
```

## 关键技术细节

### 1. 坐标转换系统

M3A_vivo_gemini使用复杂的坐标转换系统来处理不同屏幕方向和分辨率：

```python
def _logical_to_physical(
    logical_coordinates: tuple[int, int],
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
) -> tuple[int, int]:
    """将逻辑坐标转换为物理坐标"""
    x, y = logical_coordinates
    px0, py0, px1, py1 = physical_frame_boundary
    px, py = px1 - px0, py1 - py0
    lx, ly = logical_screen_size
    
    if orientation == 0:  # 竖屏
        return (int(x * px / lx) + px0, int(y * py / ly) + py0)
    elif orientation == 1:  # 横屏
        return (px - int(y * px / ly) + px0, int(x * py / lx) + py0)
    # ... 其他方向的处理
```

### 2. UI元素验证

系统包含严格的UI元素验证机制：

```python
def validate_ui_element(
    ui_element: representation_utils.UIElement,
    screen_width_height_px: tuple[int, int],
) -> bool:
    """验证UI元素的有效性"""
    screen_width, screen_height = screen_width_height_px
    
    # 过滤不可见元素
    if not ui_element.is_visible:
        return False
    
    # 验证边界框
    if ui_element.bbox_pixels:
        x_min = ui_element.bbox_pixels.x_min
        x_max = ui_element.bbox_pixels.x_max
        y_min = ui_element.bbox_pixels.y_min
        y_max = ui_element.bbox_pixels.y_max
        
        if (x_min >= x_max or x_min >= screen_width or 
            x_max <= 0 or y_min >= y_max or 
            y_min >= screen_height or y_max <= 0):
            return False
    
    return True
```

### 3. 安全机制

代理包含多层安全机制：

1. **LLM安全检查**: 检测并处理不安全的输出
2. **动作验证**: 确保动作参数的有效性
3. **异常处理**: 优雅处理执行过程中的错误

```python
if is_safe == False:
    action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""
```

### 4. 历史管理

代理维护详细的历史记录，包括：

- 每个步骤的截图
- 动作选择和原因
- 执行结果和坐标
- LLM原始响应
- 步骤总结

## 性能优化

### 1. 视觉标记优化

- 使用缩放因子确保标记在不同分辨率下的一致性
- 优化边界框绘制性能

### 2. 内存管理

- 及时释放截图内存
- 使用浅拷贝避免不必要的数据复制

### 3. 并发处理

- 支持异步环境接口
- 优化等待时间设置

## 错误处理机制

### 1. 动作解析错误

```python
try:
    converted_action = json_action.JSONAction(
        **agent_utils.extract_json(action),
    )
    step_data['action_output_json'] = converted_action
except Exception as e:
    print('Failed to convert the output to a valid action.')
    step_data['action_output_json'] = None
    step_data['summary'] = 'Can not parse the output to a valid action.'
```

### 2. 索引越界处理

```python
if action_index >= num_ui_elements:
    print(f'Index out of range, prediction index is {action_index}')
    step_data['summary'] = 'The parameter index is out of range.'
    return base_agent.AgentInteractionResult(False, step_data)
```

### 3. 执行错误处理

```python
try:
    actual_action_coordinates = self.env.execute_action(converted_action)
    step_data['actual_action_coordinates'] = actual_action_coordinates
except Exception as e:
    print('Failed to execute action.')
    step_data['summary'] = 'Can not execute the action.'
    return base_agent.AgentInteractionResult(False, step_data)
```

## 配置参数

### 主要配置选项

- `--agent`: 代理类型选择
- `--vivo_gemini_api_model`: Gemini模型版本
- `--max_rounds`: 最大执行轮数
- `--wait_after_action_seconds`: 动作后等待时间
- `--device_console_port`: 设备控制台端口
- `--device_grpc_port`: 设备gRPC端口

### 环境配置

- ADB路径配置
- 设备序列号设置
- 屏幕分辨率配置
- 方向检测设置

## 使用示例

### 基本使用

```bash
python benchmark_run.py \
    --agent M3A_vivo_gemini \
    --task "打开时钟应用并查看伦敦时间" \
    --vivo_gemini_api_model gemini-2.5-pro \
    --max_rounds 10 \
    --output_dir ./results/m3a_vivo_gemini_test
```

### 高级配置

```bash
python benchmark_run.py \
    --agent M3A_vivo_gemini \
    --task "在日历应用中创建一个新事件" \
    --vivo_gemini_api_model gemini-2.5-pro \
    --max_rounds 15 \
    --wait_after_action_seconds 3.0 \
    --device_console_port 5554 \
    --device_grpc_port 8554 \
    --output_dir ./results/calendar_task
```

## 评估与监控

### 1. 性能指标

- **成功率**: 任务完成率统计
- **执行时间**: 总执行时间和每步耗时
- **Token使用**: 提示和完成token统计
- **错误率**: 各种错误类型的统计

### 2. 日志记录

系统记录详细的执行日志：

```python
benchmark_log.append({
    "step": action_cnt,
    "action_output": response.data["action_output"],
    "summary": response.data["summary"],
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "action": action_log,
})
```

### 3. 可视化报告

支持生成HTML格式的执行报告，包含：

- 执行前后的截图对比
- 动作选择和原因分析
- 步骤总结和状态变化
- 错误信息和调试数据

## 扩展性

### 1. 新动作类型支持

可以通过扩展`JSONAction`类来支持新的动作类型：

```python
class CustomAction(json_action.JSONAction):
    action_type: str = "custom_action"
    custom_param: str = ""
```

### 2. 新模型集成

可以通过实现`MultimodalLlmWrapper`接口来集成新的LLM模型：

```python
class CustomLLMWrapper(infer.MultimodalLlmWrapper):
    def predict_mm(self, prompt: str, images: list[np.ndarray]) -> tuple[str, bool, Any]:
        # 实现自定义LLM调用逻辑
        pass
```

### 3. 新验证器支持

可以通过扩展验证器来支持新的任务类型和评估方法。

## 总结

M3A_vivo_gemini是一个功能强大的多模态Android自主代理，它结合了先进的视觉理解能力和精确的动作执行机制。通过使用Vivo Gemini作为底层模型，该代理能够理解复杂的视觉场景并执行精确的设备操作。其模块化设计使得系统具有良好的扩展性和可维护性，为Android自动化任务提供了一个强大而灵活的解决方案。 