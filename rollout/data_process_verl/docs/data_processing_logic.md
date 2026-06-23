# MobileForge 数据处理管道详解

本文档详细介绍了 `mobileforge_data_processor.py` 脚本的数据处理流程、核心逻辑和关键功能。

## 1. 概述

`mobileforge_data_processor.py` 是一个用于处理 MobileForge Rollout 数据的命令行工具。其主要目标是将原始的 `rollout` 结果（包含模型日志、截图、评估文件等）转换为可用于 MobileForge GRPO 训练的结构化数据格式。

### 核心功能

- **自动化数据格式检测**：脚本能自动识别两种日志格式，并采取不同的处理策略。
  - **新版格式**：`detailed_model_logs.json` 中直接内嵌了完整的 Base64 图像数据。
  - **占位符图像格式**：`detailed_model_logs.json` 中包含图像占位符，需要从本地 PNG 文件中重建。
- **图像错位问题修复**：针对占位符图像格式中因 UI-TARS 日志机制导致的图像与对话历史不匹配问题，脚本实现了核心的修复逻辑。
- **步骤级正负样本划分**：利用 `evaluation_summary.json` 中的步骤级合理性分析，精确地将每个操作步骤划分为正样本（合理操作）或负样本（不合理操作），为 GRPO 训练提供高质量输入。
- **多目录合并处理**：支持一次性输入多个 `rollout` 目录，脚本会自动合并所有任务并进行统一处理。
- **结构化输出**：所有处理结果都保存在一个带时间戳的会话目录中，包含训练文件、统计数据和使用说明，方便追溯和使用。

## 2. 脚本入口与参数

通过命令行运行 `python mobileforge_data_processor.py` 来启动数据处理。

### 主要命令行参数

- `--rollout_dir` (必需): 一个或多个 `rollout` 结果的输入目录路径。
- `--output_dir`: 输出目录，默认为 `processed_data`。
- `--format`: 输出格式，支持 `grpo` (默认) 和 `r1v`。
- `--max_tasks`: 限制处理的最大任务数量，用于快速测试或处理部分数据。
- `--test`: 测试模式，仅处理前3个任务。

**注意**：脚本虽然定义了并行处理的参数（`--parallel`, `--max_workers` 等），但在当前版本的 `main` 函数中并未启用，所有处理均在单线程下执行。并行处理逻辑已在 `core/parallel_processor.py` 中实现，但需要修改主脚本才能调用。

## 3. 核心处理流程

数据处理流程可以分为三个主要阶段：**数据加载与预处理** -> **核心逻辑处理** -> **数据保存**。

![Data Process Flow](https://i.imgur.com/your_image_link.png)  <!-- 你可以替换成流程图链接 -->

### 阶段一：数据加载与预处理 (在 `MobileForgeDataProcessor.process_all_tasks` 中)

1.  **扫描任务目录**：脚本首先会遍历所有指定的 `--rollout_dir`，收集其中所有的任务子目录。
2.  **任务迭代**：对每个任务目录，脚本会调用 `process_single_task` 方法进行独立处理。

### 阶段二：核心逻辑处理 (在 `MobileForgeDataProcessor.process_single_task` 及子方法中)

这是数据处理的核心，每个任务的处理都遵循以下步骤：

1.  **遍历 `Attempt`**：一个任务可能包含多次尝试 (`attempt_1`, `attempt_2`, ...)。脚本会遍历位于 `UITARS/` 子目录下的所有 `attempt_*` 目录。

2.  **加载关键文件**：在每个 `attempt` 目录中，脚本会加载两个核心文件：
    - `evaluation_summary.json`：包含对该次尝试的评估结果，最关键的是 `step_reasonableness_analysis` 字段，它定义了每一步操作的合理性。
    - `detailed_model_logs.json`：记录了模型与环境交互的详细日志，包括多轮对话历史、模型响应和截屏信息。

3.  **处理轨迹数据 (`process_detailed_model_logs`)**：
    a. **版本检测 (`_check_has_base64_images`)**：首先检查 `detailed_model_logs.json` 中的图像数据格式。如果日志中包含完整的 `data:image/...;base64,...` 数据，则识别为 **新版格式**；如果包含占位符 `[BASE64_IMAGE_DATA_REMOVED_FOR_LOGGING]`，则识别为 **占位符图像格式**。

    b. **图像修复 (占位符图像格式专属, `replace_image_placeholders_with_png`)**：
        - **核心问题**：UI-TARS 在记录多轮对话历史时，为了节省空间，只保留最近的5张截图，并且截图文件名（如 `0.png`, `1.png`）与对话中的步骤并非一一对应，导致历史图像错位。
        - **修复逻辑 (`create_correct_png_mapping`)**：脚本通过分析对话结构和当前步骤，精确地计算出每个图像占位符应该对应哪一个 PNG 文件。它重建了正确的时间序列，确保对话历史中的每张图片都与当时真实的设备状态一致。
        - **占位符替换**：将占位符替换为从对应 PNG 文件中读取并编码为 Base64 的图像数据。

4.  **转换为训练格式 (`convert_to_training_format`)**：
    a. **构建完整对话**：将修复后的对话历史和模型的响应组合成完整的对话流。
    b. **处理重试步骤**：在一次任务中，模型可能会对同一步骤进行多次重试。此方法会对这些重试步骤进行去重，只保留一次有代表性的执行（优先保留成功的，其次保留最后一次的），避免数据冗余。
    c. **整合步骤级标注**：将从 `evaluation_summary.json` 中解析出的步骤级标签（`positive`/`negative`）和解释，附加到每一步的元数据中。

### 阶段三：数据保存 (在 `MobileForgeDataSaver` 中)

1.  **创建会话目录**：在指定的 `output_dir`下，创建一个以时间戳命名的子目录（例如 `session_20251106_100000`），用于存放本次处理的所有产出。

2.  **保存为 GRPO 格式 (`_save_grpo_format`)**：
    - **逻辑**：该方法遍历所有处理过的任务的每一个步骤。根据步骤元数据中的**步骤级标注** (`step_label`) 来决定样本的类别。
    - **正样本**：如果一个步骤的 `step_label` 是 `positive`，它对应的对话（历史+响应）被视为一个正样本，保存到 `mobileforge_positive_{timestamp}.json` 文件中。
    - **负样本**：如果 `step_label` 是 `negative`，则被视为一个负样本，保存到 `mobileforge_negative_{timestamp}.json` 文件中。
    - **忽略未知**：`step_label` 为 `unknown` 的步骤将被忽略，以保证训练数据质量。

3.  **生成辅助文件**：
    - `mobileforge_grpo_stats_{timestamp}.json`: 包含详细的统计信息，如正负样本数量、重试去重统计、图像修复详情等。
    - `session_summary.json`: 本次处理会话的总体摘要。
    - `README.md`: 自动生成的使用说明，包含了如何使用产出文件进行 MobileForge 训练的示例命令。

## 4. 输出文件结构

处理完成后，输出目录的结构如下：

```
processed_data/
└── session_YYYYMMDD_HHMMSS/       # 带时间戳的会话目录
    ├── mobileforge_positive_*.json  # GRPO 格式正样本
    ├── mobileforge_negative_*.json  # GRPO 格式负样本
    ├── mobileforge_grpo_stats_*.json# 详细统计信息
    ├── session_summary.json        # 会话汇总
    └── README.md                   # 使用说明，包含训练命令
```

这份详细的文档和结构化的输出，确保了数据处理流程的透明性、可复现性和易用性。
