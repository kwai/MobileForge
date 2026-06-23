"""
设备初始化模块

在并行探索开始前，对模拟器设备进行初始化配置：
- 设置系统时间为 2023-10-15（与 AndroidWorld 一致）
- 安装24个必需的应用程序（支持本地APK缓存优先）
- 配置系统参数（时区、亮度等）

此模块基于 comprehensive_setup 的逻辑，简化并适配并行探索场景。
"""

import logging
import subprocess
import time
import os
import sys
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入本地APK安装器
try:
    from parallel_exploration.local_apk_installer import LocalApkInstaller
    LOCAL_APK_INSTALLER_AVAILABLE = True
except ImportError:
    LOCAL_APK_INSTALLER_AVAILABLE = False

# 尝试导入 android_world 模块
try:
    from android_world.env import env_launcher
    from android_world.env import interface
    from android_world.env import adb_utils
    from android_world.env.setup_device import setup
    from android_world.env import device_constants
    from android_world.utils import datetime_utils
    ANDROID_WORLD_AVAILABLE = True
except ImportError:
    ANDROID_WORLD_AVAILABLE = False
    print("Warning: android_world package not available. Device initialization will be limited.")


@dataclass
class DeviceInitConfig:
    """设备初始化配置"""
    console_port: int = 5554
    grpc_port: int = 8554
    timezone: str = "UTC"
    datetime_base: str = "2023-10-15 10:00:00"
    brightness: str = "max"
    orientation: str = "portrait"
    install_apps: bool = True
    random_seed: int = 42
    # 本地APK缓存相关配置
    use_local_apks: bool = True  # 优先使用本地APK缓存
    local_apk_dir: str = ""  # 本地APK目录，空字符串表示使用默认目录
    fallback_to_network: bool = True  # 本地没有时回退到网络下载


class DeviceInitializer:
    """设备初始化器
    
    负责在探索开始前对设备进行完整的初始化配置。
    """
    
    def __init__(self, device_serial: str, config: Optional[DeviceInitConfig] = None):
        """
        初始化设备初始化器
        
        Args:
            device_serial: 设备序列号（如 emulator-5554）
            config: 初始化配置，如果为None则使用默认配置
        """
        self.device_serial = device_serial
        self.config = config or DeviceInitConfig()
        self.logger = logging.getLogger(__name__)
        self.env: Optional[interface.AsyncEnv] = None
        
        # 从设备序列号解析端口
        if device_serial.startswith("emulator-"):
            port = int(device_serial.split("-")[1])
            self.config.console_port = port
            # gRPC 端口通常是 console_port + 3000
            self.config.grpc_port = port + 3000
    
    def initialize(self, skip_app_install: bool = False) -> bool:
        """
        执行完整的设备初始化
        
        Args:
            skip_app_install: 是否跳过应用安装（用于已安装应用的设备）
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info(f"Starting device initialization for {self.device_serial}...")
            print(f"🚀 Initializing device {self.device_serial}...")
            
            # 步骤1：连接设备
            print(f"  [1/4] Connecting to device...")
            if not self._connect_device():
                self.logger.error("Failed to connect to device")
                return False
            
            # 步骤2：设置系统参数（时间、时区等）
            print(f"  [2/4] Configuring system settings...")
            self._setup_system_settings()
            
            # 步骤3：安装应用（如果需要）
            if not skip_app_install and self.config.install_apps:
                print(f"  [3/4] Installing 24 apps (this may take several minutes)...")
                self._install_apps()
            else:
                print(f"  [3/4] Skipping app installation...")
            
            # 步骤4：基础配置完成
            print(f"  [4/4] Finalizing setup...")
            self._finalize_setup()
            
            print(f"✅ Device {self.device_serial} initialized successfully!")
            self.logger.info(f"Device {self.device_serial} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Device initialization failed: {e}")
            print(f"❌ Device initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self._cleanup()
    
    def _connect_device(self) -> bool:
        """连接到设备并创建环境"""
        if not ANDROID_WORLD_AVAILABLE:
            self.logger.warning("android_world not available, using ADB fallback")
            return self._connect_device_adb_fallback()
        
        try:
            # 找到 ADB 路径
            adb_path = self._find_adb_path()
            self.logger.info(f"Using ADB at: {adb_path}")
            
            # 创建 AndroidWorld 环境
            self.env = env_launcher.load_and_setup_env(
                console_port=self.config.console_port,
                emulator_setup=False,  # 我们会手动设置
                freeze_datetime=False,  # 我们会手动设置时间
                adb_path=adb_path,
                grpc_port=self.config.grpc_port
            )
            
            # 验证连接
            screen_size = self.env.controller.device_screen_size
            self.logger.info(f"Connected to device with screen size: {screen_size}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
    
    def _connect_device_adb_fallback(self) -> bool:
        """ADB 回退连接方式"""
        try:
            result = subprocess.run(
                ['adb', '-s', self.device_serial, 'shell', 'echo', 'connected'],
                capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0 and 'connected' in result.stdout
        except Exception as e:
            self.logger.error(f"ADB fallback connection failed: {e}")
            return False
    
    def _find_adb_path(self) -> str:
        """查找 ADB 路径"""
        potential_paths = [
            os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
            '/usr/local/bin/adb',
            '/usr/bin/adb',
            'adb'
        ]
        
        for path in potential_paths:
            if path == 'adb':
                try:
                    subprocess.run(['which', 'adb'], check=True, capture_output=True)
                    return 'adb'
                except subprocess.CalledProcessError:
                    continue
            elif os.path.isfile(path):
                return path
        
        return 'adb'
    
    def _setup_system_settings(self) -> None:
        """设置系统参数"""
        if not ANDROID_WORLD_AVAILABLE or self.env is None:
            self._setup_system_settings_adb_fallback()
            return
        
        import datetime
        import random
        
        # 设置随机种子
        random.seed(self.config.random_seed)
        self.logger.info(f"Set random seed to {self.config.random_seed}")
        
        try:
            # 确保有root权限
            adb_utils.set_root_if_needed(self.env.controller.env)
            
            # 设置时间和时区
            datetime_utils.setup_datetime(self.env.controller.env)
            self.logger.info("Disabled auto time/timezone, enabled 24-hour format, set timezone to UTC")
            
            # 设置具体时间
            dt = datetime.datetime.strptime(self.config.datetime_base, "%Y-%m-%d %H:%M:%S")
            datetime_utils.set_datetime(self.env.controller.env, dt)
            self.logger.info(f"Set device datetime to: {self.config.datetime_base}")
            
            # 设置亮度
            if self.config.brightness in ['max', 'min']:
                adb_utils.set_brightness(self.config.brightness, self.env.controller.env)
            
            # 设置屏幕方向
            if self.config.orientation in ['portrait', 'landscape']:
                adb_utils.change_orientation(self.config.orientation, self.env.controller.env)
            
            # 返回主屏幕
            adb_utils.press_home_button(self.env.controller)
            
        except Exception as e:
            self.logger.warning(f"Error during system setup: {e}")
            self._setup_system_settings_adb_fallback()
    
    def _setup_system_settings_adb_fallback(self) -> None:
        """使用 ADB 命令设置系统参数（回退方式）"""
        try:
            # 设置时区为 UTC
            self._adb_shell("settings put global auto_time 0")
            self._adb_shell("settings put global auto_time_zone 0")
            self._adb_shell("setprop persist.sys.timezone UTC")
            
            # 设置24小时制
            self._adb_shell("settings put system time_12_24 24")
            
            # 设置时间（需要 root）
            self._adb_shell("date -s '2023-10-15 10:00:00'")
            
            # 设置最大亮度
            self._adb_shell("settings put system screen_brightness 255")
            
            # 返回主屏幕
            self._adb_shell("input keyevent KEYCODE_HOME")
            
            self.logger.info("System settings configured via ADB fallback")
            
        except Exception as e:
            self.logger.warning(f"ADB fallback system setup failed: {e}")
    
    def _adb_shell(self, command: str) -> str:
        """执行 ADB shell 命令"""
        try:
            result = subprocess.run(
                ['adb', '-s', self.device_serial, 'shell', command],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout.strip()
        except Exception as e:
            self.logger.debug(f"ADB command failed: {command}, error: {e}")
            return ""
    
    def _install_apps(self) -> None:
        """安装24个必需的应用
        
        安装策略：
        1. 如果启用本地APK缓存，优先从本地安装
        2. 本地没有的应用，使用 android_world 从网络下载安装
        """
        local_installed = []
        needs_network = []
        
        # 步骤1：尝试从本地APK缓存安装
        if self.config.use_local_apks and LOCAL_APK_INSTALLER_AVAILABLE:
            local_installed, needs_network = self._install_from_local_cache()
        
        # 步骤2：处理需要网络下载的应用
        if needs_network or not self.config.use_local_apks:
            if self.config.fallback_to_network:
                self._install_from_network(needs_network)
            else:
                if needs_network:
                    self.logger.warning(
                        f"以下应用需要网络下载但已禁用回退: {needs_network}"
                    )
        
        # 统计安装结果
        total_local = len(local_installed)
        total_network = len(needs_network) if self.config.fallback_to_network else 0
        self.logger.info(
            f"App installation complete: {total_local} from local, "
            f"{total_network} from network"
        )
    
    def _install_from_local_cache(self) -> Tuple[List[str], List[str]]:
        """从本地APK缓存安装应用
        
        Returns:
            (已安装的包名列表, 需要网络下载的包名列表)
        """
        installed = []
        needs_network = []
        
        try:
            # 创建本地安装器
            apk_dir = self.config.local_apk_dir or None  # 空字符串转为None使用默认目录
            installer = LocalApkInstaller(
                device_serial=self.device_serial,
                apk_dir=apk_dir
            ) if apk_dir else LocalApkInstaller(device_serial=self.device_serial)
            
            # 获取可用的本地APK列表
            available_packages = installer.get_available_packages()
            
            if not available_packages:
                self.logger.info("No local APKs available, will use network download")
                print("    ℹ️ 本地APK缓存为空，将使用网络下载")
                return [], []
            
            print(f"    📦 发现 {len(available_packages)} 个本地APK缓存")
            self.logger.info(f"Found {len(available_packages)} local APKs: {available_packages}")
            
            # 定义进度回调
            def progress_callback(current, total, package_name):
                print(f"      [{current}/{total}] 安装 {package_name}...")
            
            # 批量安装本地APK
            results = installer.install_all_available(
                skip_installed=True,
                progress_callback=progress_callback
            )
            
            # 统计结果
            for package_name, (success, message) in results.items():
                if success:
                    installed.append(package_name)
                else:
                    # 安装失败的可能需要网络下载
                    if "需要网络下载" in message or "本地APK不存在" in message:
                        needs_network.append(package_name)
                    else:
                        self.logger.warning(f"Local install failed for {package_name}: {message}")
            
            print(f"    ✅ 从本地缓存安装了 {len(installed)} 个应用")
            
        except Exception as e:
            self.logger.error(f"Local APK installation error: {e}")
            print(f"    ⚠️ 本地APK安装出错: {e}")
        
        return installed, needs_network
    
    def _install_from_network(self, specific_packages: List[str] = None) -> None:
        """使用 android_world 从网络安装应用
        
        Args:
            specific_packages: 指定要安装的包名列表，None表示安装所有
        """
        if not ANDROID_WORLD_AVAILABLE or self.env is None:
            self.logger.warning("Cannot install apps: android_world not available")
            print("    ⚠️ 跳过网络安装 (android_world 不可用)")
            return
        
        try:
            if specific_packages:
                print(f"    🌐 从网络下载并安装 {len(specific_packages)} 个应用...")
                self.logger.info(f"Installing from network: {specific_packages}")
            else:
                print("    🌐 从网络下载并安装所有应用（可能需要较长时间）...")
                self.logger.info("Installing all apps from network...")
            
            # 返回主屏幕，避免快速设置菜单阻挡
            adb_utils.press_home_button(self.env.controller)
            
            # 使用 AndroidWorld 的 setup 系统安装应用
            # 注意：android_world 的 setup_apps 不支持指定包名，会安装所有应用
            setup.setup_apps(self.env)
            
            print("    ✅ 网络安装完成")
            self.logger.info("Network installation completed")
            
        except Exception as e:
            self.logger.error(f"Network app installation failed: {e}")
            print(f"    ❌ 网络安装失败: {e}")
            raise
    
    def _finalize_setup(self) -> None:
        """完成设置的最终步骤"""
        try:
            # 返回主屏幕
            if self.env:
                adb_utils.press_home_button(self.env.controller)
            else:
                self._adb_shell("input keyevent KEYCODE_HOME")
            
            # 等待一小段时间确保设置生效
            time.sleep(2)
            
        except Exception as e:
            self.logger.debug(f"Finalize setup warning: {e}")
    
    def _cleanup(self) -> None:
        """清理资源"""
        if self.env:
            try:
                self.env.close()
            except Exception as e:
                self.logger.debug(f"Cleanup warning: {e}")
            finally:
                self.env = None


def initialize_device(device_serial: str, 
                     config: Optional[DeviceInitConfig] = None,
                     skip_app_install: bool = False) -> bool:
    """
    便捷函数：初始化单个设备
    
    Args:
        device_serial: 设备序列号
        config: 初始化配置
        skip_app_install: 是否跳过应用安装
        
    Returns:
        bool: 是否成功
    """
    initializer = DeviceInitializer(device_serial, config)
    return initializer.initialize(skip_app_install=skip_app_install)


def initialize_devices_parallel(device_serials: List[str],
                               config: Optional[DeviceInitConfig] = None,
                               skip_app_install: bool = False) -> Dict[str, bool]:
    """
    并行初始化多个设备
    
    Args:
        device_serials: 设备序列号列表
        config: 初始化配置（所有设备共用基础配置）
        skip_app_install: 是否跳过应用安装
        
    Returns:
        Dict[str, bool]: 每个设备的初始化结果
    """
    import concurrent.futures
    
    results = {}
    
    def init_single_device(serial: str) -> tuple:
        # 为每个设备创建独立的配置
        device_config = DeviceInitConfig() if config is None else DeviceInitConfig(
            timezone=config.timezone,
            datetime_base=config.datetime_base,
            brightness=config.brightness,
            orientation=config.orientation,
            install_apps=config.install_apps,
            random_seed=config.random_seed,
            use_local_apks=config.use_local_apks,
            local_apk_dir=config.local_apk_dir,
            fallback_to_network=config.fallback_to_network,
        )
        
        # 从设备序列号解析端口
        if serial.startswith("emulator-"):
            port = int(serial.split("-")[1])
            device_config.console_port = port
            device_config.grpc_port = port + 3000
        
        success = initialize_device(serial, device_config, skip_app_install)
        return (serial, success)
    
    print(f"🔄 Initializing {len(device_serials)} devices in parallel...")
    
    # 使用线程池并行初始化
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(device_serials)) as executor:
        futures = [executor.submit(init_single_device, serial) for serial in device_serials]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                serial, success = future.result()
                results[serial] = success
            except Exception as e:
                logging.error(f"Device initialization error: {e}")
    
    # 统计结果
    success_count = sum(1 for v in results.values() if v)
    print(f"📊 Initialization complete: {success_count}/{len(device_serials)} devices successful")
    
    return results


if __name__ == "__main__":
    # 测试单设备初始化
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Android device for exploration")
    parser.add_argument("-s", "--serial", default="emulator-5554", help="Device serial")
    parser.add_argument("--skip-apps", action="store_true", help="Skip app installation")
    parser.add_argument(
        "--no-local-apks",
        action="store_true",
        help="Disable local APK cache (always use network download)"
    )
    parser.add_argument(
        "--local-apk-dir",
        default="",
        help="Custom local APK directory path"
    )
    parser.add_argument(
        "--no-network-fallback",
        action="store_true",
        help="Disable fallback to network download when local APK not found"
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置
    config = DeviceInitConfig(
        use_local_apks=not args.no_local_apks,
        local_apk_dir=args.local_apk_dir,
        fallback_to_network=not args.no_network_fallback
    )
    
    success = initialize_device(
        args.serial,
        config=config,
        skip_app_install=args.skip_apps
    )
    sys.exit(0 if success else 1)

