# 探索输出可视化工具使用指南

## 问题修复说明

您遇到的可视化网页没有生成的问题已经修复！原因是main.py中的HTML报告生成被意外跳过了。

## 修复内容

1. **恢复HTML报告生成**：重新启用了visualizer模块的HTML报告生成功能
2. **修复导入路径**：解决了模块导入的兼容性问题
3. **保持内存优化**：维持原有的内存优化特性

## 快速开始

### 智能跳过机制（新功能）

现在脚本会自动检测已解析的数据，避免重复解析！

**首次运行**（完整解析）：
```bash
cd /path/to/MobileForge-Opensource/explore
python data_process/vis-exploration-output/main.py \
    --input_dir ./exploration_output/ \
    --output_dir ./exploration_output_final_test_02 \
    --memory_limit 16 \
    --batch_size 2 \
    --verbose
```

**后续运行**（只生成HTML，秒级完成）：
```bash
# 相同命令会自动跳过解析，直接生成HTML
python data_process/vis-exploration-output/main.py \
    --input_dir ./exploration_output/ \
    --output_dir ./exploration_output_final_test_01 \
    --verbose
```

### 预期输出

**首次运行**（完整解析）：
```
📊 Parsing exploration data...
✅ Successfully parsed data for X app(s)
🎨 Generating HTML reports...
✅ HTML reports generated successfully
📄 Open ./exploration_output_final_test_01/overview.html to view the main report
```

**后续运行**（智能跳过）：
```
🔍 Found existing parsed data!
📥 Loading from existing JSON files instead of re-parsing...
✅ Successfully loaded data for X app(s)
🎨 Generating HTML reports...
✅ HTML reports generated successfully
📄 Open ./exploration_output_final_test_01/overview.html to view the main report
```

## 生成的文件结构

```
exploration_output_final_test_01/
├── overview.html                 # 📊 主要概览报告（从这里开始）
├── app_package_name_1/
│   ├── report.html               # 📱 单个应用详细报告
│   ├── app_info.json            # 应用信息
│   ├── statistics.json          # 统计数据
│   ├── trajectories_summary.json # 轨迹摘要
│   ├── trajectories/            # 单个轨迹文件
│   └── screenshots/             # 截图文件（如果启用）
└── app_package_name_2/
    └── ...
```

## 可视化报告功能

### 📊 概览报告 (overview.html)

- **全局统计**：总应用数、轨迹数、成功率等
- **应用对比图表**：轨迹数量和成功率对比
- **应用列表表格**：详细的应用统计信息
- **跳转链接**：直接访问单个应用的详细报告

### 📱 应用详细报告 (report.html)

- **应用统计卡片**：轨迹数、成功率、平均步数等
- **深度分析图表**：按探索深度的轨迹分布和成功率
- **深度分析卡片**：每个深度的详细统计
- **轨迹详情**：前20个轨迹的详细信息
- **步骤预览**：每个轨迹的步骤摘要
- **完整轨迹查看**：可展开查看完整的步骤详情
- **截图展示**：包含操作前后的截图（如果有）

## 高级使用选项

### 内存优化选项

```bash
# 轻量模式（推荐用于快速预览）
python data_process/vis-exploration-output/main.py \
    --input_dir ./exploration_output/ \
    --output_dir ./output_lightweight \
    --lightweight

# 无截图模式（节省内存和时间）
python data_process/vis-exploration-output/main.py \
    --input_dir ./exploration_output/ \
    --output_dir ./output_no_screenshots \
    --no_screenshots \
    --memory_limit 8 \
    --batch_size 1

# 限制轨迹数量
python data_process/vis-exploration-output/main.py \
    --input_dir ./exploration_output/ \
    --output_dir ./output_limited \
    --max_trajectories 50 \
    --memory_limit 16
```

### 特定应用处理

```bash
# 只处理特定应用
python data_process/vis-exploration-output/main.py \
    --input_dir ./exploration_output/ \
    --output_dir ./output_specific_app \
    --app_package com.example.app
```

### 仅生成JSON数据（不生成HTML）

```bash
python data_process/vis-exploration-output/main.py \
    --input_dir ./exploration_output/ \
    --output_dir ./output_json_only \
    --no_html
```

### 智能跳过控制选项

```bash
# 强制重新解析（忽略已有数据）
python data_process/vis-exploration-output/main.py \
    --input_dir ./exploration_output/ \
    --output_dir ./exploration_output_final_test_01 \
    --force_reparse

# 仅生成HTML（要求已有解析数据）
python data_process/vis-exploration-output/main.py \
    --input_dir ./exploration_output/ \
    --output_dir ./exploration_output_final_test_01 \
    --html_only
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | `./exploration_output` | 输入目录路径 |
| `--output_dir` | `./exploration_output_uncompress` | 输出目录路径 |
| `--memory_limit` | `16.0` | 内存限制（GB） |
| `--batch_size` | `3` | 批处理大小 |
| `--max_trajectories` | `0` | 每个应用最大轨迹数（0=无限制） |
| `--no_screenshots` | `False` | 跳过截图提取 |
| `--no_html` | `False` | 跳过HTML报告生成 |
| `--lightweight` | `False` | 轻量模式 |
| `--verbose` | `False` | 详细输出 |
| `--force_reparse` | `False` | **强制重新解析（忽略已有数据）** |
| `--html_only` | `False` | **仅生成HTML报告（需要已有解析数据）** |

## 故障排除

### 1. 内存不足错误

```bash
# 解决方案：使用更严格的内存限制
python data_process/vis-exploration-output/main.py \
    --input_dir ./exploration_output/ \
    --output_dir ./output_safe \
    --no_screenshots \
    --memory_limit 8 \
    --batch_size 1 \
    --max_trajectories 20
```

### 2. 处理速度慢

```bash
# 解决方案：使用轻量模式
python data_process/vis-exploration-output/main.py \
    --input_dir ./exploration_output/ \
    --output_dir ./output_fast \
    --lightweight \
    --aggressive_cleanup
```

### 3. 某些应用处理失败

```bash
# 解决方案：逐个处理应用
for app in $(ls exploration_output/); do
    echo "Processing $app..."
    python data_process/vis-exploration-output/main.py \
        --input_dir ./exploration_output/ \
        --output_dir ./output_individual \
        --app_package $app \
        --verbose
done
```

## 查看结果

1. **打开主报告**：
   ```bash
   # 在浏览器中打开概览报告
   xdg-open exploration_output_final_test_01/overview.html
   # 或者
   firefox exploration_output_final_test_01/overview.html
   ```

2. **浏览单个应用**：
   - 在概览报告中点击应用名称或"View Details"链接
   - 或直接打开：`exploration_output_final_test_01/[app_package]/report.html`

## 报告特性

### 🎨 可视化图表
- 使用Chart.js生成交互式图表
- 支持悬停显示详细信息
- 响应式设计，支持移动设备

### 📸 截图功能
- 支持点击放大查看
- 自动压缩优化加载速度
- 显示操作前后对比

### 🔍 详细信息
- 可折叠的轨迹详情
- 步骤级别的操作记录
- UI元素和活动信息

### 📊 统计分析
- 成功率分析
- 深度分布统计
- 平均步数计算

## 性能建议

1. **首次运行**：建议使用 `--lightweight` 模式快速预览
2. **大数据集**：使用 `--no_screenshots` 和较小的 `--batch_size`
3. **详细分析**：在确认数据正常后，使用完整模式重新生成
4. **内存限制**：根据系统内存调整 `--memory_limit` 参数

现在您可以重新运行之前的命令，应该能看到完整的HTML可视化报告了！
