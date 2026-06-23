#!/usr/bin/env python3
"""
APK导出工具

从已安装应用的Android设备导出APK文件到本地 apks/ 目录，
用于构建本地APK缓存，加速后续设备初始化。

使用方法:
    python scripts/export_apks.py                    # 导出所有支持的应用
    python scripts/export_apks.py -s emulator-5554   # 指定设备
    python scripts/export_apks.py -p com.arduia.expense  # 只导出指定应用
    python scripts/export_apks.py --list             # 列出设备上已安装的目标应用
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_APK_DIR = os.path.join(PROJECT_ROOT, "apks")
DEFAULT_MANIFEST_FILE = os.path.join(DEFAULT_APK_DIR, "apk_manifest.json")

# 需要导出的第三方应用包名列表
TARGET_PACKAGES = [
    # 第三方应用
    "com.arduia.expense",
    "net.gsantner.markor",
    "net.osmand",
    "com.simplemobiletools.calendar.pro",
    "com.simplemobiletools.smsmessenger",
    "com.simplemobiletools.gallery.pro",
    "com.flauschcode.broccoli",
    "org.tasks",
    "net.cozic.joplin",
    "de.dennisguse.opentracks",
    "code.name.monkey.retromusic",
    "org.videolan.vlc",
]


def get_adb_devices() -> List[str]:
    """获取所有已连接的ADB设备"""
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        devices = []
        for line in result.stdout.strip().split("\n")[1:]:
            if line.strip() and "\tdevice" in line:
                device_serial = line.split("\t")[0]
                devices.append(device_serial)
        
        return devices
    except Exception as e:
        print(f"获取设备列表失败: {e}")
        return []


def get_installed_packages(device_serial: str) -> Dict[str, str]:
    """
    获取设备上已安装应用的包名和APK路径
    
    Returns:
        {包名: APK路径} 的字典
    """
    try:
        result = subprocess.run(
            ["adb", "-s", device_serial, "shell", "pm", "list", "packages", "-f"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        packages = {}
        for line in result.stdout.strip().split("\n"):
            if line.startswith("package:"):
                # 格式: package:/path/to/app.apk=com.example.package
                parts = line[8:].rsplit("=", 1)
                if len(parts) == 2:
                    apk_path, package_name = parts
                    packages[package_name] = apk_path
        
        return packages
    except Exception as e:
        print(f"获取已安装应用列表失败: {e}")
        return {}


def get_app_version(device_serial: str, package_name: str) -> str:
    """获取应用版本号"""
    try:
        result = subprocess.run(
            ["adb", "-s", device_serial, "shell", "dumpsys", "package", package_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        for line in result.stdout.split("\n"):
            if "versionName=" in line:
                return line.split("versionName=")[1].strip()
        
        return ""
    except Exception:
        return ""


def export_apk(
    device_serial: str,
    package_name: str,
    remote_path: str,
    output_dir: str
) -> Tuple[bool, str, str]:
    """
    从设备导出APK文件
    
    Args:
        device_serial: 设备序列号
        package_name: 包名
        remote_path: 设备上的APK路径
        output_dir: 本地输出目录
        
    Returns:
        (是否成功, 本地文件路径, 错误消息)
    """
    local_filename = f"{package_name}.apk"
    local_path = os.path.join(output_dir, local_filename)
    
    try:
        # 使用 adb pull 导出 APK
        result = subprocess.run(
            ["adb", "-s", device_serial, "pull", remote_path, local_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0 and os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            return True, local_path, f"成功 ({file_size / 1024 / 1024:.1f} MB)"
        else:
            error_msg = result.stderr or result.stdout
            return False, "", f"导出失败: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, "", "导出超时"
    except Exception as e:
        return False, "", f"导出错误: {e}"


def update_manifest(
    manifest_file: str,
    package_name: str,
    version: str,
    filename: str
) -> None:
    """更新APK清单配置文件"""
    manifest = {"apps": {}}
    
    # 读取现有清单
    if os.path.exists(manifest_file):
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass
    
    # 确保有 apps 字段
    if "apps" not in manifest:
        manifest["apps"] = {}
    
    # 更新或添加应用信息
    manifest["apps"][package_name] = {
        "filename": filename,
        "version": version,
        "source": f"从设备导出 ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        "notes": ""
    }
    
    # 写回文件
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def list_target_apps(device_serial: str) -> None:
    """列出设备上已安装的目标应用"""
    print(f"\n📱 设备 {device_serial} 上的目标应用:\n")
    
    installed = get_installed_packages(device_serial)
    
    found = []
    missing = []
    
    for package in TARGET_PACKAGES:
        if package in installed:
            version = get_app_version(device_serial, package)
            found.append((package, version))
        else:
            missing.append(package)
    
    if found:
        print("✅ 已安装:")
        for package, version in found:
            version_str = f" (v{version})" if version else ""
            print(f"   - {package}{version_str}")
    
    if missing:
        print("\n❌ 未安装:")
        for package in missing:
            print(f"   - {package}")
    
    print(f"\n总计: {len(found)} 已安装, {len(missing)} 未安装")


def export_apps(
    device_serial: str,
    packages: Optional[List[str]] = None,
    output_dir: str = DEFAULT_APK_DIR,
    manifest_file: str = DEFAULT_MANIFEST_FILE
) -> Dict[str, Tuple[bool, str]]:
    """
    导出应用APK
    
    Args:
        device_serial: 设备序列号
        packages: 要导出的包名列表，None表示导出所有目标应用
        output_dir: 输出目录
        manifest_file: 清单文件路径
        
    Returns:
        {包名: (是否成功, 消息)} 的字典
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取设备上已安装的应用
    installed = get_installed_packages(device_serial)
    
    # 确定要导出的应用
    target_packages = packages or TARGET_PACKAGES
    
    results = {}
    
    print(f"\n📦 开始从设备 {device_serial} 导出APK...\n")
    
    for i, package_name in enumerate(target_packages, 1):
        print(f"  [{i}/{len(target_packages)}] {package_name}...", end=" ", flush=True)
        
        if package_name not in installed:
            print("⏭️ 未安装，跳过")
            results[package_name] = (False, "未安装")
            continue
        
        remote_path = installed[package_name]
        success, local_path, message = export_apk(
            device_serial,
            package_name,
            remote_path,
            output_dir
        )
        
        if success:
            print(f"✅ {message}")
            
            # 获取版本号并更新清单
            version = get_app_version(device_serial, package_name)
            update_manifest(
                manifest_file,
                package_name,
                version,
                f"{package_name}.apk"
            )
        else:
            print(f"❌ {message}")
        
        results[package_name] = (success, message)
    
    # 打印统计
    success_count = sum(1 for s, _ in results.values() if s)
    print(f"\n📊 导出完成: {success_count}/{len(target_packages)} 成功")
    print(f"📁 输出目录: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="从Android设备导出APK文件到本地缓存目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/export_apks.py                        # 导出所有支持的应用
  python scripts/export_apks.py -s emulator-5554       # 指定设备
  python scripts/export_apks.py -p com.arduia.expense  # 只导出指定应用
  python scripts/export_apks.py --list                 # 列出设备上已安装的目标应用
        """
    )
    
    parser.add_argument(
        "-s", "--serial",
        help="设备序列号（不指定则自动选择第一个设备）"
    )
    parser.add_argument(
        "-p", "--packages",
        nargs="+",
        help="要导出的包名列表（不指定则导出所有目标应用）"
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_APK_DIR,
        help=f"输出目录（默认: {DEFAULT_APK_DIR}）"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只列出设备上已安装的目标应用，不导出"
    )
    parser.add_argument(
        "--all-installed",
        action="store_true",
        help="导出设备上所有已安装的第三方应用（不只是目标列表）"
    )
    
    args = parser.parse_args()
    
    # 获取设备
    if args.serial:
        device_serial = args.serial
    else:
        devices = get_adb_devices()
        if not devices:
            print("❌ 没有找到已连接的设备")
            print("请确保设备已连接并启用USB调试")
            sys.exit(1)
        
        device_serial = devices[0]
        if len(devices) > 1:
            print(f"ℹ️ 发现多个设备，使用第一个: {device_serial}")
            print(f"   可用设备: {devices}")
    
    # 检查设备连接
    try:
        result = subprocess.run(
            ["adb", "-s", device_serial, "shell", "echo", "ok"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"❌ 无法连接到设备 {device_serial}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 设备连接错误: {e}")
        sys.exit(1)
    
    print(f"🔗 已连接设备: {device_serial}")
    
    # 执行操作
    if args.list:
        list_target_apps(device_serial)
    else:
        packages = None
        if args.packages:
            packages = args.packages
        elif args.all_installed:
            # 获取所有已安装的第三方应用
            installed = get_installed_packages(device_serial)
            packages = [
                pkg for pkg in installed.keys()
                if not pkg.startswith("com.android.") and not pkg.startswith("com.google.")
            ]
            print(f"ℹ️ 将导出 {len(packages)} 个第三方应用")
        
        results = export_apps(
            device_serial,
            packages=packages,
            output_dir=args.output
        )
        
        # 检查是否有失败
        failed = [pkg for pkg, (success, _) in results.items() if not success]
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
