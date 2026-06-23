#!/usr/bin/env python3
"""
多设备并行探索交互式启动脚本
在项目根目录运行: python interactive_parallel_exploration.py

增强功能：
- 设备初始化选项（时间设置、应用安装）
- 按需数据注入（在探索前注入应用所需数据）
"""
import os
import sys
import subprocess
import time
from typing import List, Dict

# 添加parallel_exploration模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'parallel_exploration'))

def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("🚀 多设备并行App探索系统 - 交互式启动")
    print("=" * 60)
    print()

def check_adb():
    """检查ADB是否可用"""
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
            devices = [line.split()[0] for line in lines if line.strip() and 'device' in line]
            return True, devices
        else:
            return False, []
    except FileNotFoundError:
        return False, []

def get_device_count() -> int:
    """获取用户想要使用的设备数量"""
    while True:
        try:
            count = input("请输入要使用的设备数量: ").strip()
            count = int(count)
            return count
        except ValueError:
            print("❌ 请输入有效的数字")

def choose_device_mode() -> str:
    """选择设备模式"""
    print("\n📱 设备模式选择:")
    print("1. 物理设备")
    print("2. 模拟器")

    while True:
        choice = input("请选择设备模式 (1/2): ").strip()
        if choice == "1":
            return "physical"
        elif choice == "2":
            return "emulator"
        else:
            print("❌ 请选择 1 或 2")

def get_emulator_config() -> Dict[str, str]:
    """获取模拟器配置"""
    config = {}
    
    # 默认值配置
    DEFAULT_EMULATOR_EXE = "/path/to/android-sdk/emulator/emulator"
    DEFAULT_AVD_NAME = "AndroidWorldAvd"
    DEFAULT_AVD_HOME = "/path/to/android/avd"

    print("\n⚙️ 模拟器配置:")

    # 模拟器可执行文件路径
    while True:
        emulator_exe = input(f"请输入模拟器可执行文件路径 (默认: {DEFAULT_EMULATOR_EXE}): ").strip()
        if not emulator_exe:
            # 使用默认值
            emulator_exe = DEFAULT_EMULATOR_EXE
        if os.path.exists(emulator_exe):
            config['emulator_exe'] = emulator_exe
            break
        else:
            # 尝试在PATH中查找
            try:
                subprocess.run(['which', emulator_exe], check=True, capture_output=True)
                config['emulator_exe'] = emulator_exe
                break
            except subprocess.CalledProcessError:
                print(f"❌ 找不到模拟器: {emulator_exe}")

    # AVD名称
    while True:
        avd_name = input(f"请输入源AVD名称 (默认: {DEFAULT_AVD_NAME}): ").strip()
        if not avd_name:
            # 使用默认值
            avd_name = DEFAULT_AVD_NAME
        config['source_avd_name'] = avd_name
        break

    # AVD主目录
    while True:
        avd_home = input(f"请输入AVD主目录路径 (默认: {DEFAULT_AVD_HOME}): ").strip()
        if not avd_home:
            # 使用默认值
            avd_home = DEFAULT_AVD_HOME
        # 展开用户目录
        avd_home = os.path.expanduser(avd_home)
        if os.path.exists(avd_home):
            config['source_avd_home'] = avd_home
            break
        else:
            print(f"❌ 目录不存在: {avd_home}")

    # Android SDK路径 (可选)
    sdk_path = input("请输入Android SDK路径 (可选，直接回车跳过): ").strip()
    if sdk_path:
        sdk_path = os.path.expanduser(sdk_path)
        if os.path.exists(sdk_path):
            config['android_sdk_path'] = sdk_path
        else:
            print(f"⚠️ SDK路径不存在，将跳过: {sdk_path}")

    return config

def get_app_assignments(device_count: int) -> List[str]:
    """获取每个设备要探索的应用"""
    apps = []

    print(f"\n📱 为 {device_count} 个设备分配应用:")
    print("请为每个设备输入要探索的应用包名")
    print("提示: 常见应用包名如 com.android.camera2, com.android.settings, net.osmand 等")
    print()

    for i in range(device_count):
        while True:
            app = input(f"设备 {i+1} 要探索的应用包名: ").strip()
            if app:
                apps.append(app)
                print(f"✅ 设备 {i+1}: {app}")
                break
            else:
                print("❌ 请输入有效的应用包名")

    return apps

def get_exploration_config() -> Dict[str, int]:
    """获取探索配置参数"""
    config = {}

    print("\n⚙️ 探索参数配置 (直接回车使用默认值):")

    # 最大探索步数
    steps = input("每个任务最大探索步数 (默认: 30): ").strip()
    config['max_exploration_steps'] = int(steps) if steps else 30

    # 最大探索深度
    depth = input("最大探索深度 (默认: 5): ").strip()
    config['max_exploration_depth'] = int(depth) if depth else 5

    # 分支因子
    branching = input("每个节点最大探索任务数 (默认: 3): ").strip()
    config['max_branching_factor'] = int(branching) if branching else 3

    return config


def get_initialization_config(device_mode: str) -> Dict[str, bool]:
    """获取设备初始化配置
    
    Args:
        device_mode: 设备模式 ('emulator' 或 'physical')
        
    Returns:
        初始化配置字典
    """
    config = {
        'perform_initialization': False,
        'skip_app_install': False,
        'inject_app_data': True,
    }
    
    print("\n🔧 设备初始化与数据注入配置:")
    
    # 只有模拟器才支持初始化
    if device_mode == 'emulator':
        print("\n📱 设备初始化（仅模拟器）:")
        print("   初始化将执行以下操作:")
        print("   - 设置系统时间为 2023-10-15（与AndroidWorld一致）")
        print("   - 安装24个必需的应用程序")
        print("   - 配置系统参数（时区、亮度等）")
        
        while True:
            init_choice = input("\n是否启用设备初始化? (y/n, 默认: n): ").strip().lower()
            if init_choice in ['', 'n', 'no', '否']:
                config['perform_initialization'] = False
                break
            elif init_choice in ['y', 'yes', '是']:
                config['perform_initialization'] = True
                
                # 询问是否跳过应用安装
                skip_apps = input("是否跳过应用安装（假设应用已安装）? (y/n, 默认: n): ").strip().lower()
                config['skip_app_install'] = skip_apps in ['y', 'yes', '是']
                break
            else:
                print("❌ 请输入 y 或 n")
    else:
        print("\n📱 物理设备不支持自动初始化")
        print("   请确保物理设备已预先配置好所需应用和数据")
    
    # 数据注入配置
    print("\n📥 按需数据注入:")
    print("   在探索每个App前，自动注入该App所需的确定性数据")
    print("   支持的应用包括: Contacts, SMS, Dialer, Gallery, Files,")
    print("   Recipe, Tasks, Joplin, OpenTracks, Calendar, Markor,")
    print("   Expense, Retro Music, VLC")
    
    while True:
        inject_choice = input("\n是否启用按需数据注入? (y/n, 默认: y): ").strip().lower()
        if inject_choice in ['', 'y', 'yes', '是']:
            config['inject_app_data'] = True
            break
        elif inject_choice in ['n', 'no', '否']:
            config['inject_app_data'] = False
            break
        else:
            print("❌ 请输入 y 或 n")
    
    return config

def get_output_directory() -> str:
    """获取输出目录"""
    default_dir = "./exploration_output_parallel"
    output_dir = input(f"输出目录 (默认: {default_dir}): ").strip()
    return output_dir if output_dir else default_dir

def confirm_settings(device_count: int, device_mode: str, apps: List[str],
                    output_dir: str, exploration_config: Dict[str, int],
                    init_config: Dict[str, bool] = None,
                    emulator_config: Dict[str, str] = None) -> bool:
    """确认设置"""
    print("\n" + "=" * 50)
    print("📋 配置确认")
    print("=" * 50)
    print(f"设备数量: {device_count}")
    print(f"设备模式: {'模拟器' if device_mode == 'emulator' else '物理设备'}")

    if emulator_config:
        print("模拟器配置:")
        for key, value in emulator_config.items():
            print(f"  {key}: {value}")

    if init_config:
        print("初始化与数据注入配置:")
        print(f"  设备初始化: {'启用' if init_config.get('perform_initialization') else '禁用'}")
        if init_config.get('perform_initialization'):
            print(f"  跳过应用安装: {'是' if init_config.get('skip_app_install') else '否'}")
        print(f"  按需数据注入: {'启用' if init_config.get('inject_app_data') else '禁用'}")

    print("应用分配:")
    for i, app in enumerate(apps):
        print(f"  设备 {i+1}: {app}")

    print(f"输出目录: {output_dir}")
    print("探索参数:")
    for key, value in exploration_config.items():
        print(f"  {key}: {value}")

    print("=" * 50)

    while True:
        confirm = input("确认开始探索? (y/n): ").strip().lower()
        if confirm in ['y', 'yes', '是']:
            return True
        elif confirm in ['n', 'no', '否']:
            return False
        else:
            print("❌ 请输入 y 或 n")

def build_command(device_count: int, device_mode: str, apps: List[str],
                 output_dir: str, exploration_config: Dict[str, int],
                 init_config: Dict[str, bool] = None,
                 emulator_config: Dict[str, str] = None) -> List[str]:
    """构建执行命令"""
    cmd = [
        "python", "parallel_exploration/main.py",
        "-batch_mode",
        "-num_devices", str(device_count),
        "-output_dir", output_dir,
        "-max_exploration_steps", str(exploration_config['max_exploration_steps']),
        "-max_exploration_depth", str(exploration_config['max_exploration_depth']),
        "-max_branching_factor", str(exploration_config['max_branching_factor'])
    ]

    # 创建临时应用列表文件
    app_file = "temp_app_list.txt"
    with open(app_file, 'w', encoding='utf-8') as f:
        for app in apps:
            f.write(f"{app}\n")

    cmd.extend(["-app_list_file", app_file])

    # 添加模拟器配置
    if device_mode == "emulator" and emulator_config:
        cmd.append("-use_emulator")
        cmd.extend(["-emulator_exe", emulator_config['emulator_exe']])
        cmd.extend(["-source_avd_name", emulator_config['source_avd_name']])
        cmd.extend(["-source_avd_home", emulator_config['source_avd_home']])

        if 'android_sdk_path' in emulator_config:
            cmd.extend(["-android_sdk_path", emulator_config['android_sdk_path']])

    # 添加初始化配置
    if init_config:
        if init_config.get('perform_initialization'):
            cmd.append("-perform_initialization")
            if init_config.get('skip_app_install'):
                cmd.append("-skip_app_install")
        
        if init_config.get('inject_app_data'):
            cmd.append("-inject_app_data")
        else:
            cmd.append("-no_inject_app_data")

    return cmd, app_file

def main():
    """主函数"""
    print_banner()

    # 检查ADB
    print("🔍 检查系统环境...")
    adb_available, physical_devices = check_adb()
    if not adb_available:
        print("❌ 错误: ADB不可用，请确保Android SDK已正确安装")
        return

    print(f"✅ ADB可用，发现 {len(physical_devices)} 个物理设备")
    if physical_devices:
        print("物理设备列表:")
        for device in physical_devices:
            print(f"  - {device}")
    print()

    try:
        # 获取配置
        device_count = get_device_count()
        device_mode = choose_device_mode()

        emulator_config = None
        if device_mode == "emulator":
            emulator_config = get_emulator_config()
        elif device_mode == "physical" and len(physical_devices) < device_count:
            print(f"⚠️ 警告: 你想使用 {device_count} 个设备，但只发现 {len(physical_devices)} 个物理设备")
            confirm = input("是否继续? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes', '是']:
                print("操作已取消")
                return

        apps = get_app_assignments(device_count)
        exploration_config = get_exploration_config()
        
        # 获取初始化和数据注入配置
        init_config = get_initialization_config(device_mode)
        
        output_dir = get_output_directory()

        # 确认配置
        if not confirm_settings(device_count, device_mode, apps, output_dir,
                              exploration_config, init_config, emulator_config):
            print("操作已取消")
            return

        # 构建命令
        print("\n🚀 正在启动并行探索...")
        cmd, app_file = build_command(device_count, device_mode, apps, output_dir,
                                    exploration_config, init_config, emulator_config)

        print(f"执行命令: {' '.join(cmd)}")
        print()

        # 执行命令
        try:
            result = subprocess.run(cmd, check=True)
            print("\n🎉 探索完成!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 探索过程中出现错误: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断操作")
        finally:
            # 清理临时文件
            if os.path.exists(app_file):
                os.remove(app_file)
                print(f"清理临时文件: {app_file}")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    main()