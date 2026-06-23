"""
本地APK安装模块

从项目的 apks/ 目录安装预下载的APK文件，
避免每次初始化时从网络下载，显著加快设备初始化速度。
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_APK_DIR = os.path.join(PROJECT_ROOT, "apks")
DEFAULT_MANIFEST_FILE = os.path.join(DEFAULT_APK_DIR, "apk_manifest.json")


@dataclass
class ApkInfo:
    """APK信息数据类"""
    package_name: str
    filename: str
    version: str
    source: str
    notes: str
    local_path: Optional[str] = None
    exists: bool = False


class LocalApkInstaller:
    """本地APK安装器
    
    管理和安装本地缓存的APK文件。
    """
    
    def __init__(
        self,
        device_serial: str,
        apk_dir: str = DEFAULT_APK_DIR,
        manifest_file: str = DEFAULT_MANIFEST_FILE,
        grant_permissions: bool = True
    ):
        """
        初始化本地APK安装器
        
        Args:
            device_serial: 设备序列号（如 emulator-5554）
            apk_dir: APK文件存放目录
            manifest_file: APK清单配置文件路径
            grant_permissions: 安装时是否自动授予权限
        """
        self.device_serial = device_serial
        self.apk_dir = apk_dir
        self.manifest_file = manifest_file
        self.grant_permissions = grant_permissions
        self.logger = logging.getLogger(__name__)
        
        # 加载APK清单
        self.manifest: Dict = {}
        self.apk_info_cache: Dict[str, ApkInfo] = {}
        self._load_manifest()
    
    def _load_manifest(self) -> None:
        """加载APK清单配置文件"""
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, 'r', encoding='utf-8') as f:
                    self.manifest = json.load(f)
                self.logger.info(f"Loaded APK manifest from {self.manifest_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load APK manifest: {e}")
                self.manifest = {"apps": {}}
        else:
            self.logger.warning(f"APK manifest not found: {self.manifest_file}")
            self.manifest = {"apps": {}}
        
        # 构建APK信息缓存
        self._build_apk_info_cache()
    
    def _build_apk_info_cache(self) -> None:
        """构建APK信息缓存，检查文件是否存在"""
        apps = self.manifest.get("apps", {})
        
        for package_name, info in apps.items():
            filename = info.get("filename", f"{package_name}.apk")
            local_path = os.path.join(self.apk_dir, filename)
            exists = os.path.isfile(local_path)
            
            self.apk_info_cache[package_name] = ApkInfo(
                package_name=package_name,
                filename=filename,
                version=info.get("version", ""),
                source=info.get("source", ""),
                notes=info.get("notes", ""),
                local_path=local_path if exists else None,
                exists=exists
            )
        
        # 同时扫描目录中的APK文件（可能不在清单中）
        self._scan_apk_directory()
    
    def _scan_apk_directory(self) -> None:
        """扫描APK目录，发现清单中未记录的APK文件"""
        if not os.path.isdir(self.apk_dir):
            return
        
        for filename in os.listdir(self.apk_dir):
            if not filename.endswith('.apk'):
                continue
            
            # 从文件名推断包名
            package_name = filename[:-4]  # 移除 .apk 后缀
            
            if package_name not in self.apk_info_cache:
                local_path = os.path.join(self.apk_dir, filename)
                self.apk_info_cache[package_name] = ApkInfo(
                    package_name=package_name,
                    filename=filename,
                    version="",
                    source="目录扫描发现",
                    notes="",
                    local_path=local_path,
                    exists=True
                )
                self.logger.info(f"Found unlisted APK: {filename}")
    
    def has_local_apk(self, package_name: str) -> bool:
        """
        检查是否存在指定包名的本地APK
        
        Args:
            package_name: 应用包名
            
        Returns:
            是否存在本地APK文件
        """
        info = self.apk_info_cache.get(package_name)
        return info is not None and info.exists
    
    def get_apk_path(self, package_name: str) -> Optional[str]:
        """
        获取指定包名的APK文件路径
        
        Args:
            package_name: 应用包名
            
        Returns:
            APK文件路径，不存在则返回None
        """
        info = self.apk_info_cache.get(package_name)
        if info and info.exists:
            return info.local_path
        return None
    
    def get_available_packages(self) -> List[str]:
        """
        获取所有可用的本地APK包名列表
        
        Returns:
            可用的包名列表
        """
        return [
            pkg for pkg, info in self.apk_info_cache.items()
            if info.exists
        ]
    
    def is_app_installed(self, package_name: str) -> bool:
        """
        检查应用是否已安装在设备上
        
        Args:
            package_name: 应用包名
            
        Returns:
            是否已安装
        """
        try:
            result = subprocess.run(
                ['adb', '-s', self.device_serial, 'shell', 'pm', 'list', 'packages', package_name],
                capture_output=True, text=True, timeout=30
            )
            return f"package:{package_name}" in result.stdout
        except Exception as e:
            self.logger.warning(f"Failed to check if {package_name} is installed: {e}")
            return False
    
    def install_app(
        self,
        package_name: str,
        reinstall: bool = False,
        timeout: int = 120
    ) -> Tuple[bool, str]:
        """
        安装单个应用
        
        Args:
            package_name: 应用包名
            reinstall: 是否重新安装（即使已安装）
            timeout: 安装超时时间（秒）
            
        Returns:
            (是否成功, 消息)
        """
        # 检查本地APK是否存在
        apk_path = self.get_apk_path(package_name)
        if not apk_path:
            return False, f"本地APK不存在: {package_name}"
        
        # 检查是否已安装
        if not reinstall and self.is_app_installed(package_name):
            self.logger.info(f"App already installed: {package_name}")
            return True, f"应用已安装: {package_name}"
        
        # 构建安装命令
        install_cmd = ['adb', '-s', self.device_serial, 'install']
        
        if reinstall:
            install_cmd.append('-r')  # 允许重新安装
        
        if self.grant_permissions:
            install_cmd.append('-g')  # 自动授予权限
        
        install_cmd.append(apk_path)
        
        self.logger.info(f"Installing {package_name} from {apk_path}")
        
        try:
            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0 and 'Success' in result.stdout:
                self.logger.info(f"Successfully installed: {package_name}")
                return True, f"安装成功: {package_name}"
            else:
                error_msg = result.stderr or result.stdout
                self.logger.error(f"Failed to install {package_name}: {error_msg}")
                return False, f"安装失败: {package_name}, 错误: {error_msg}"
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Installation timeout for {package_name}")
            return False, f"安装超时: {package_name}"
        except Exception as e:
            self.logger.error(f"Installation error for {package_name}: {e}")
            return False, f"安装错误: {package_name}, 错误: {e}"
    
    def install_all_available(
        self,
        skip_installed: bool = True,
        progress_callback=None
    ) -> Dict[str, Tuple[bool, str]]:
        """
        安装所有可用的本地APK
        
        Args:
            skip_installed: 是否跳过已安装的应用
            progress_callback: 进度回调函数，签名为 callback(current, total, package_name)
            
        Returns:
            {包名: (是否成功, 消息)} 的字典
        """
        available_packages = self.get_available_packages()
        total = len(available_packages)
        results = {}
        
        self.logger.info(f"Installing {total} apps from local cache...")
        
        for i, package_name in enumerate(available_packages):
            if progress_callback:
                progress_callback(i + 1, total, package_name)
            
            success, message = self.install_app(
                package_name,
                reinstall=not skip_installed
            )
            results[package_name] = (success, message)
            
            # 安装间隔，避免设备过载
            if success:
                time.sleep(1)
        
        # 统计结果
        success_count = sum(1 for s, _ in results.values() if s)
        self.logger.info(f"Installation complete: {success_count}/{total} successful")
        
        return results
    
    def install_packages(
        self,
        package_names: List[str],
        skip_installed: bool = True,
        fallback_to_network: bool = True,
        progress_callback=None
    ) -> Dict[str, Tuple[bool, str]]:
        """
        安装指定的应用列表
        
        Args:
            package_names: 要安装的包名列表
            skip_installed: 是否跳过已安装的应用
            fallback_to_network: 本地没有时是否标记为需要网络下载
            progress_callback: 进度回调函数
            
        Returns:
            {包名: (是否成功, 消息)} 的字典
        """
        total = len(package_names)
        results = {}
        needs_network_download = []
        
        self.logger.info(f"Installing {total} specified apps...")
        
        for i, package_name in enumerate(package_names):
            if progress_callback:
                progress_callback(i + 1, total, package_name)
            
            # 检查是否有本地APK
            if self.has_local_apk(package_name):
                success, message = self.install_app(
                    package_name,
                    reinstall=not skip_installed
                )
                results[package_name] = (success, message)
                
                if success:
                    time.sleep(1)
            else:
                # 没有本地APK
                if fallback_to_network:
                    needs_network_download.append(package_name)
                    results[package_name] = (False, f"需要网络下载: {package_name}")
                else:
                    results[package_name] = (False, f"本地APK不存在: {package_name}")
        
        # 报告需要网络下载的应用
        if needs_network_download:
            self.logger.info(
                f"{len(needs_network_download)} apps need network download: "
                f"{needs_network_download}"
            )
        
        return results
    
    def get_summary(self) -> Dict:
        """
        获取本地APK缓存的统计摘要
        
        Returns:
            统计信息字典
        """
        available = self.get_available_packages()
        all_packages = list(self.apk_info_cache.keys())
        
        return {
            "apk_directory": self.apk_dir,
            "manifest_file": self.manifest_file,
            "total_in_manifest": len(self.manifest.get("apps", {})),
            "total_discovered": len(all_packages),
            "available_count": len(available),
            "available_packages": available,
            "missing_packages": [
                pkg for pkg in all_packages
                if not self.apk_info_cache[pkg].exists
            ]
        }
    
    def print_summary(self) -> None:
        """打印本地APK缓存的统计摘要"""
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print("📦 本地APK缓存摘要")
        print("=" * 50)
        print(f"APK目录: {summary['apk_directory']}")
        print(f"清单文件: {summary['manifest_file']}")
        print(f"清单中的应用数: {summary['total_in_manifest']}")
        print(f"发现的APK文件数: {summary['available_count']}")
        
        if summary['available_packages']:
            print("\n✅ 可用的APK:")
            for pkg in summary['available_packages']:
                print(f"   - {pkg}")
        
        if summary['missing_packages']:
            print("\n❌ 缺失的APK:")
            for pkg in summary['missing_packages']:
                print(f"   - {pkg}")
        
        print("=" * 50 + "\n")


def get_local_installer(device_serial: str) -> LocalApkInstaller:
    """
    获取本地APK安装器实例的便捷函数
    
    Args:
        device_serial: 设备序列号
        
    Returns:
        LocalApkInstaller实例
    """
    return LocalApkInstaller(device_serial)


def install_local_apks(
    device_serial: str,
    package_names: Optional[List[str]] = None,
    skip_installed: bool = True
) -> Dict[str, Tuple[bool, str]]:
    """
    安装本地APK的便捷函数
    
    Args:
        device_serial: 设备序列号
        package_names: 要安装的包名列表，None表示安装所有可用的
        skip_installed: 是否跳过已安装的应用
        
    Returns:
        安装结果字典
    """
    installer = LocalApkInstaller(device_serial)
    
    if package_names:
        return installer.install_packages(
            package_names,
            skip_installed=skip_installed
        )
    else:
        return installer.install_all_available(
            skip_installed=skip_installed
        )


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="本地APK安装器")
    parser.add_argument(
        "-s", "--serial",
        default="emulator-5554",
        help="设备序列号"
    )
    parser.add_argument(
        "-p", "--packages",
        nargs="+",
        help="要安装的包名列表"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="只显示摘要，不安装"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    installer = LocalApkInstaller(args.serial)
    
    if args.summary:
        installer.print_summary()
    else:
        def progress(current, total, package):
            print(f"  [{current}/{total}] Installing {package}...")
        
        if args.packages:
            results = installer.install_packages(
                args.packages,
                progress_callback=progress
            )
        else:
            results = installer.install_all_available(
                progress_callback=progress
            )
        
        # 打印结果
        print("\n安装结果:")
        for pkg, (success, msg) in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {pkg}: {msg}")
