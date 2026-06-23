"""
多设备并行探索主程序
改进的exploration_and_mining.py，支持多设备并行探索不同应用

增强功能：
- 支持设备初始化（时间设置、应用安装）
- 支持按需数据注入
"""
import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict
from glob import glob
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 尝试相对导入（当作为模块运行时）
    from .device_manager import DeviceManager
    from .parallel_explorer import run_batch_exploration
except ImportError:
    # 回退到绝对导入（当直接运行时）
    from device_manager import DeviceManager
    from parallel_explorer import run_batch_exploration

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.utils import load_object_from_disk, save_object_to_disk
from utils.memory import load_knowledge_raw_data, load_memory
from utils.knowledge_generation import update_trajectory_to_knowledge
from exploration_and_mining import auto_exploration


def get_default_app_packages() -> List[str]:
    """获取默认的应用包名列表"""
    return [
        "com.android.camera2",
        "com.google.android.deskclock",
        "com.google.android.contacts",
        "com.arduia.expense",
        "net.gsantner.markor",
        "net.osmand",
        "com.simplemobiletools.calendar.pro",
        "com.simplemobiletools.smsmessenger",
        "com.android.settings"
    ]


def read_app_packages_from_file(file_path: str) -> List[str]:
    """从文件读取应用包名列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return packages
    except Exception as e:
        print(f"Error reading app packages from {file_path}: {e}")
        return []


def setup_device_manager(args) -> DeviceManager:
    """设置设备管理器"""
    if args.use_emulator:
        if not all([args.emulator_exe, args.source_avd_name, args.source_avd_home]):
            raise ValueError("Using emulator requires: emulator_exe, source_avd_name, source_avd_home")

        return DeviceManager(
            num_devices=args.num_devices,
            use_emulator=True,
            emulator_exe=args.emulator_exe,
            source_avd_name=args.source_avd_name,
            source_avd_home=args.source_avd_home,
            target_avd_home=args.target_avd_home or args.source_avd_home,
            android_sdk_path=args.android_sdk_path
        )
    else:
        return DeviceManager(
            num_devices=args.num_devices,
            use_emulator=False
        )


def process_knowledge_extraction(output_dir: str, usage_stats: dict):
    """处理知识提取逻辑"""
    print("\n=== Starting Knowledge Extraction ===")

    # 配置参数
    EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_UPGRADE = (
        os.getenv("EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_UPGRADE", "False").lower() == "true"
    )
    EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_DOWNGRADE = (
        os.getenv("EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_DOWNGRADE", "True").lower() == "true"
    )

    # 查找所有app_info.json文件
    app_info_json_paths = glob(
        os.path.join(output_dir, "**", "app_info.json"), recursive=True
    )

    if not app_info_json_paths:
        print("No app_info.json files found. Skipping knowledge extraction.")
        return

    # 加载应用信息
    app_infos = {}
    for app_info_json_path in app_info_json_paths:
        try:
            with open(app_info_json_path, "r", encoding="utf-8") as f:
                app_info = json.load(f)
            app_infos[app_info["app_pkg"]] = app_info
        except Exception as e:
            print(f"Error loading {app_info_json_path}: {e}")

    print(f"Found {len(app_infos)} apps for knowledge extraction")

    # 加载知识库
    print("Loading knowledge raw data...")
    knowledge_raw_data = load_knowledge_raw_data()
    fusion_knowledge = []
    locations = []

    # 处理版本变化
    for pkg, app_info in app_infos.items():
        if pkg in knowledge_raw_data:
            old_info = knowledge_raw_data[pkg]
            new_version = app_info["app_version"]
            old_version = old_info["app_version"]

            if new_version > old_version and EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_UPGRADE:
                print(f"App {pkg} upgraded from {old_version} to {new_version}, clearing knowledge base")
                old_info["knowledge"] = []
            elif new_version < old_version and EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_DOWNGRADE:
                print(f"App {pkg} downgraded from {old_version} to {new_version}, clearing knowledge base")
                old_info["knowledge"] = []

    # 构建融合知识库
    for pkg, info in knowledge_raw_data.items():
        fusion_knowledge.extend(info["knowledge"])
        locations.extend([(pkg, i) for i in range(len(info["knowledge"]))])

    # 添加新应用
    for pkg, app_info in app_infos.items():
        if pkg not in knowledge_raw_data:
            print(f"Adding new app {pkg} to knowledge base")
            knowledge_raw_data[pkg] = app_info
            knowledge_raw_data[pkg]["knowledge"] = []

    # 加载融合记忆
    print("Loading fusion memory...")
    fusion_memory = load_memory(fusion_knowledge)

    # 提取知识
    print("Extracting knowledge from trajectories...")
    for app_info_json_path in tqdm(app_info_json_paths, desc="Processing apps"):
        app_info_json_dir = os.path.dirname(app_info_json_path)
        pkl_paths = glob(
            os.path.join(app_info_json_dir, "**", "*.pkl.zst"), recursive=True
        )

        for pkl_path in tqdm(pkl_paths, desc=f"Processing {os.path.basename(app_info_json_dir)}", leave=False):
            try:
                trajectory_data = load_object_from_disk(pkl_path)
                update_trajectory_to_knowledge(
                    trajectory_data=trajectory_data,
                    locations=locations,
                    fusion_memory=fusion_memory,
                    knowledge_data=knowledge_raw_data,
                    usage=usage_stats,
                )
            except Exception as e:
                print(f"Error processing {pkl_path}: {e}")

    # 保存知识库
    knowledge_base_root_path = os.path.abspath(
        os.getenv("KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH", "./knowledge_base")
    )
    os.makedirs(knowledge_base_root_path, exist_ok=True)
    fp = os.path.join(knowledge_base_root_path, "knowledge_data.pkl")

    print(f"Saving knowledge base to {fp}")
    save_object_to_disk(knowledge_raw_data, fp, compress_level=20)
    print("Knowledge extraction completed!")


def main():
    parser = argparse.ArgumentParser(
        description="多设备并行App探索程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 单设备探索指定应用:
   python parallel_exploration.py -package_name com.android.camera2

2. 多设备并行探索默认应用列表:
   python parallel_exploration.py -batch_mode -num_devices 3 -use_emulator \\
       -emulator_exe /path/to/emulator -source_avd_name MyAVD -source_avd_home /path/to/avd

3. 从文件读取应用列表并探索:
   python parallel_exploration.py -batch_mode -app_list_file apps.txt -num_devices 2

4. 物理设备并行探索:
   python parallel_exploration.py -batch_mode -num_devices 2
        """
    )

    # 基础参数
    parser.add_argument(
        "-package_name",
        help="单个应用的包名，例如 com.android.settings"
    )
    parser.add_argument(
        "-device_serial",
        help="设备序列号，使用 `adb devices` 查看"
    )
    parser.add_argument(
        "-output_dir",
        help="输出目录",
        default="./exploration_output_parallel"
    )

    # 探索参数
    parser.add_argument(
        "-max_branching_factor",
        help="每个节点最大探索任务数",
        type=int,
        default=3
    )
    parser.add_argument(
        "-max_exploration_steps",
        help="每个任务最大探索步数",
        type=int,
        default=30
    )
    parser.add_argument(
        "-max_exploration_depth",
        help="最大探索深度",
        type=int,
        default=5
    )

    # 并行设备参数
    parser.add_argument(
        "-num_devices",
        help="使用的设备数量",
        type=int,
        default=1
    )
    parser.add_argument(
        "-use_emulator",
        help="是否使用模拟器",
        action="store_true"
    )
    parser.add_argument(
        "-emulator_exe",
        help="模拟器可执行文件路径"
    )
    parser.add_argument(
        "-source_avd_name",
        help="源AVD名称"
    )
    parser.add_argument(
        "-source_avd_home",
        help="源AVD主目录"
    )
    parser.add_argument(
        "-target_avd_home",
        help="目标AVD主目录（默认与源AVD主目录相同）"
    )
    parser.add_argument(
        "-android_sdk_path",
        help="Android SDK路径"
    )

    # 批处理参数
    parser.add_argument(
        "-batch_mode",
        help="批处理模式，探索多个应用",
        action="store_true"
    )
    parser.add_argument(
        "-app_list_file",
        help="包含应用包名列表的文件路径"
    )
    parser.add_argument(
        "-skip_knowledge_extraction",
        help="跳过知识提取步骤",
        action="store_true"
    )

    # 设备初始化参数
    parser.add_argument(
        "-perform_initialization",
        help="启用设备初始化（设置时间、安装应用等，仅模拟器）",
        action="store_true"
    )
    parser.add_argument(
        "-skip_app_install",
        help="跳过应用安装（与 -perform_initialization 配合使用）",
        action="store_true"
    )
    
    # 数据注入参数
    parser.add_argument(
        "-inject_app_data",
        help="启用按需数据注入（在探索前注入应用所需数据）",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "-no_inject_app_data",
        help="禁用按需数据注入",
        action="store_true"
    )

    args = parser.parse_args()
    
    # 处理数据注入参数（-no_inject_app_data 覆盖 -inject_app_data）
    if args.no_inject_app_data:
        args.inject_app_data = False

    # 参数验证
    if not args.batch_mode and not args.package_name:
        parser.error("必须指定 -package_name 或使用 -batch_mode")

    if args.use_emulator and not all([args.emulator_exe, args.source_avd_name, args.source_avd_home]):
        parser.error("使用模拟器需要指定: emulator_exe, source_avd_name, source_avd_home")

    print("=== 多设备并行App探索程序 ===")
    print(f"配置参数: {vars(args)}")

    # 设置输出目录
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 初始化使用统计
    usage_stats = {"prompt_tokens": 0, "completion_tokens": 0}

    try:
        if args.batch_mode:
            # 批处理模式
            print("\n=== 批处理模式 ===")

            # 获取应用列表
            if args.app_list_file:
                app_packages = read_app_packages_from_file(args.app_list_file)
                if not app_packages:
                    print("从文件读取应用列表失败，使用默认列表")
                    app_packages = get_default_app_packages()
            else:
                app_packages = get_default_app_packages()

            print(f"计划探索 {len(app_packages)} 个应用: {app_packages}")

            # 设置设备管理器
            device_manager = setup_device_manager(args)
            print(f"设置 {args.num_devices} 个设备...")
            
            # 设置设备，可选初始化
            devices = device_manager.setup_devices(
                perform_initialization=args.perform_initialization,
                skip_app_install=args.skip_app_install
            )
            print(f"成功设置 {len(devices)} 个设备")
            
            # 显示初始化和数据注入状态
            if args.perform_initialization:
                print(f"✅ 设备初始化: 已启用 (跳过应用安装: {'是' if args.skip_app_install else '否'})")
            else:
                print(f"⏭️ 设备初始化: 已跳过")
            
            if args.inject_app_data:
                print(f"✅ 按需数据注入: 已启用")
            else:
                print(f"⏭️ 按需数据注入: 已禁用")

            try:
                # 执行批量探索
                batch_usage = run_batch_exploration(
                    device_manager=device_manager,
                    app_package_list=app_packages,
                    exploration_output_root_dir=output_dir,
                    max_exploration_tasks=args.max_branching_factor,
                    max_exploration_steps=args.max_exploration_steps,
                    max_exploration_depth=args.max_exploration_depth,
                    inject_app_data_before_exploration=args.inject_app_data
                )

                # 累计使用统计
                usage_stats["prompt_tokens"] += batch_usage["prompt_tokens"]
                usage_stats["completion_tokens"] += batch_usage["completion_tokens"]

            finally:
                # 清理设备
                if args.use_emulator:
                    print("正在终止模拟器...")
                    device_manager.terminate_emulators()

        else:
            # 单应用模式
            print(f"\n=== 单应用探索模式: {args.package_name} ===")

            auto_exploration(
                package_name=args.package_name,
                exploration_output_root_dir=output_dir,
                device_serial=args.device_serial,
                max_exploration_tasks=args.max_branching_factor,
                max_exploration_steps=args.max_exploration_steps,
                max_exploration_depth=args.max_exploration_depth,
                usage=usage_stats,
            )

        print(f"\n=== 探索完成 ===")
        print(f"Token使用统计: {usage_stats}")

        # 知识提取
        if not args.skip_knowledge_extraction:
            process_knowledge_extraction(output_dir, usage_stats)
        else:
            print("跳过知识提取步骤")

        print(f"最终Token使用统计: {usage_stats}")
        print("所有任务完成!")

    except KeyboardInterrupt:
        print("\n用户中断程序")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()