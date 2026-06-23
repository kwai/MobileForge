# 多设备并行探索系统 - 使用说明

## 📁 文件结构

新的并行探索系统已经重组到 `parallel_exploration/` 文件夹中：

```
MobileForge Explore/
├── run_parallel_exploration.py          # 项目根目录入口脚本
└── parallel_exploration/                # 并行探索模块
    ├── __init__.py                      # 模块初始化
    ├── main.py                          # 主程序
    ├── device_manager.py                # 设备管理器
    ├── parallel_explorer.py             # 并行探索执行器
    ├── run_parallel_exploration.sh      # 便捷启动脚本
    ├── test_parallel_system.py          # 系统测试脚本
    ├── README.md                        # 详细文档
    └── test_apps.txt                    # 测试应用列表
```

## 🚀 启动方式

### 方式一：从项目根目录运行
```bash
# 单应用探索
python run_parallel_exploration.py -package_name com.android.camera2

# 多设备并行探索
python run_parallel_exploration.py -batch_mode -num_devices 2

# 使用应用列表文件
python run_parallel_exploration.py -batch_mode -app_list_file parallel_exploration/test_apps.txt
```

### 方式二：从parallel_exploration目录内运行
```bash
cd parallel_exploration

# 单应用探索
python main.py -package_name com.android.camera2

# 多设备并行探索
python main.py -batch_mode -num_devices 2

# 使用便捷脚本
./run_parallel_exploration.sh -b -n 2
```

### 方式三：作为Python模块导入
```python
from parallel_exploration import DeviceManager, ParallelExplorer, run_batch_exploration

# 创建设备管理器
device_manager = DeviceManager(num_devices=2)

# 运行批量探索
usage_stats = run_batch_exploration(
    device_manager=device_manager,
    app_package_list=['com.android.camera2', 'com.android.settings']
)
```

## ⚙️ 主要参数

### 基础参数
- `-package_name`: 单个应用包名
- `-output_dir`: 输出目录（默认：`./exploration_output_parallel`）
- `-batch_mode`: 批处理模式（探索多个应用）

### 并行设备参数
- `-num_devices`: 使用的设备数量（默认：1）
- `-use_emulator`: 使用模拟器
- `-emulator_exe`: 模拟器可执行文件路径
- `-source_avd_name`: 源AVD名称
- `-source_avd_home`: 源AVD主目录

### 探索参数
- `-max_branching_factor`: 每个节点最大探索任务数（默认：3）
- `-max_exploration_steps`: 每个任务最大探索步数（默认：30）
- `-max_exploration_depth`: 最大探索深度（默认：5）

## 📊 功能特性

### ✅ 已验证功能
- ✅ 模块导入和包结构
- ✅ 命令行参数解析
- ✅ 双重执行方式（根目录/模块目录）
- ✅ 便捷脚本功能
- ✅ 错误处理和回退机制
- ✅ 设备管理和AVD克隆逻辑
- ✅ 并行任务队列管理
- ✅ 知识提取集成

### 🔄 核心流程
1. **设备初始化** → 检测或创建多个设备实例
2. **任务分配** → 将应用列表分配到任务队列
3. **并行执行** → 每个设备独立执行探索任务
4. **故障恢复** → 自动处理设备故障和重启
5. **结果汇总** → 收集所有设备的探索结果
6. **知识提取** → 自动提取和整合知识库

## 📈 性能提升

相比原始的单设备顺序执行：
- **2设备**: 理论上2倍速度提升
- **3设备**: 理论上3倍速度提升
- **4设备**: 理论上4倍速度提升

实际性能取决于设备性能、网络状况和应用复杂度。

## 🛠️ 开发集成

现在可以轻松地将并行探索功能集成到其他项目中：

```python
# 简单集成示例
from parallel_exploration import run_batch_exploration, DeviceManager

def my_exploration_pipeline():
    device_manager = DeviceManager(num_devices=3, use_emulator=True)
    apps = ['com.example.app1', 'com.example.app2', 'com.example.app3']

    results = run_batch_exploration(
        device_manager=device_manager,
        app_package_list=apps,
        exploration_output_root_dir='./my_output'
    )

    return results
```

系统重组完成！新的文件结构更加模块化，使用更加便捷。