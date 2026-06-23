"""
设备管理器模块，用于管理多个Android设备的并行探索
改编自MobileForge parallel exploration的设备管理机制

增强功能：
- 支持设备初始化（时间设置、应用安装）
- 与 device_initializer 模块集成
"""
import os
import sys
import subprocess
import time
import queue
import threading
import concurrent.futures
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.device import Device

# 导入设备初始化模块
try:
    from parallel_exploration.device_initializer import (
        DeviceInitializer,
        DeviceInitConfig,
        initialize_devices_parallel,
    )
    INITIALIZER_AVAILABLE = True
except ImportError:
    INITIALIZER_AVAILABLE = False


@dataclass
class DeviceInfo:
    """设备信息数据类"""
    serial: str
    console_port: Optional[int] = None
    grpc_port: Optional[int] = None
    is_emulator: bool = False


class DeviceManager:
    """设备管理器类"""

    def __init__(self,
                 num_devices: int = 1,
                 use_emulator: bool = False,
                 emulator_exe: Optional[str] = None,
                 source_avd_name: Optional[str] = None,
                 source_avd_home: Optional[str] = None,
                 target_avd_home: Optional[str] = None,
                 android_sdk_path: Optional[str] = None):
        """
        初始化设备管理器

        Args:
            num_devices: 需要的设备数量
            use_emulator: 是否使用模拟器
            emulator_exe: 模拟器可执行文件路径
            source_avd_name: 源AVD名称
            source_avd_home: 源AVD主目录
            target_avd_home: 目标AVD主目录
            android_sdk_path: Android SDK路径
        """
        self.num_devices = num_devices
        self.use_emulator = use_emulator
        self.emulator_exe = emulator_exe
        self.source_avd_name = source_avd_name
        self.source_avd_home = source_avd_home
        self.target_avd_home = target_avd_home
        self.android_sdk_path = android_sdk_path
        self.devices: List[DeviceInfo] = []

    def execute_adb(self, command: str, verbose: bool = True) -> str:
        """执行ADB命令"""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            if verbose:
                print(f"Command execution failed: {command}")
                print(result.stderr)
            return "ERROR"
        except Exception as e:
            if verbose:
                print(f"Command execution failed with exception: {command}")
                print(f"Exception: {e}")
            return "ERROR"

    def get_all_devices(self) -> List[str]:
        """获取所有连接的设备"""
        command = "adb devices"
        result = self.execute_adb(command)
        device_list = []
        if result != "ERROR":
            devices = result.split("\n")[1:]
            for device in devices:
                if device.strip():
                    device_list.append(device.split()[0])
        return device_list

    def parse_adb_devices(self, result: str) -> Dict[str, str]:
        """解析adb devices输出"""
        devices = {}
        for line in result.split("\n")[1:]:
            if line.strip():
                parts = line.split("\t")
                if len(parts) >= 2:
                    serial, status = parts[0], parts[1]
                    devices[serial] = status
        return devices

    def clone_avd(self, idx: int):
        """克隆AVD，改编自utils_clone_avd"""
        if not all([self.source_avd_home, self.source_avd_name, self.target_avd_home]):
            raise ValueError("AVD paths not properly configured")

        source_avd_dir = os.path.join(self.source_avd_home, f"{self.source_avd_name}.avd")
        source_ini_file = os.path.join(self.source_avd_home, f"{self.source_avd_name}.ini")
        target_avd_name = f"{self.source_avd_name}_{idx}"
        target_avd_dir = os.path.join(self.target_avd_home, f"{target_avd_name}.avd")
        target_ini_file = os.path.join(self.target_avd_home, f"{target_avd_name}.ini")

        # 复制AVD目录
        if os.path.exists(target_avd_dir):
            import shutil
            shutil.rmtree(target_avd_dir)

        import shutil
        shutil.copytree(source_avd_dir, target_avd_dir)

        # 复制并修改INI文件
        if os.path.exists(source_ini_file):
            with open(source_ini_file, 'r', encoding='utf-8') as f:
                ini_content = f.read()

            # 更新路径
            ini_content = ini_content.replace(source_avd_dir, target_avd_dir)
            ini_content = ini_content.replace(self.source_avd_name, target_avd_name)

            with open(target_ini_file, 'w', encoding='utf-8') as f:
                f.write(ini_content)

        print(f"Cloned AVD: {target_avd_name}")

    def setup_emulators(self):
        """设置模拟器"""
        if not self.use_emulator:
            return

        # 首先克隆所需的AVD
        for idx in range(self.num_devices):
            self.clone_avd(idx)

        # 设置环境变量
        if self.android_sdk_path:
            adb_path = os.path.join(self.android_sdk_path, "platform-tools")
            os.environ["PATH"] = f"{adb_path}{os.pathsep}{os.environ['PATH']}"

        # 创建设备信息
        self.devices = [
            DeviceInfo(
                serial=f"emulator-{5554 + (idx * 2)}",
                console_port=5554 + (idx * 2),
                grpc_port=8554 + (idx * 2),
                is_emulator=True
            )
            for idx in range(self.num_devices)
        ]

        # 启动模拟器
        for idx, device in enumerate(self.devices):
            command = [
                self.emulator_exe,
                "-avd",
                f"{self.source_avd_name}_{idx}",
                "-no-snapshot-save",
                "-no-window",
                "-no-audio",
                "-port",
                str(device.console_port),
                "-grpc",
                str(device.grpc_port),
            ]

            # 添加代理设置
            http_proxy = os.environ.get("HTTP_PROXY")
            if http_proxy:
                command.extend(["-http-proxy", http_proxy])

            subprocess.Popen(
                command,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Starting emulator {device.serial}...")

        # 等待所有模拟器启动
        self._wait_for_emulators()

    def _wait_for_emulators(self):
        """等待所有模拟器启动完成"""
        devices_serial = [device.serial for device in self.devices]
        ready_devices = []

        # 等待设备出现在adb devices中
        print("Waiting for emulators to launch...")
        while True:
            result = self.execute_adb("adb devices")
            if result == "ERROR":
                raise Exception("Error in executing ADB command")

            launched_devices = [
                serial
                for serial, status in self.parse_adb_devices(result).items()
                if status == "device" and serial in devices_serial
            ]

            print(f"{len(launched_devices)}/{self.num_devices} device(s) launched; {len(ready_devices)}/{self.num_devices} device(s) ready")

            if len(launched_devices) == self.num_devices:
                break
            time.sleep(1)

        # 等待设备完全启动
        print("Waiting for emulators to boot...")
        while True:
            for serial in launched_devices:
                if serial in ready_devices:
                    continue
                result = self.execute_adb(f"adb -s {serial} shell getprop sys.boot_completed")
                if result == "1":
                    ready_devices.append(serial)

            print(f"{len(launched_devices)}/{self.num_devices} device(s) launched; {len(ready_devices)}/{self.num_devices} device(s) ready")

            if len(ready_devices) == self.num_devices:
                break
            time.sleep(1)

        print("All emulators are ready!")

    def setup_physical_devices(self):
        """设置物理设备"""
        if self.use_emulator:
            return

        devices_list = self.get_all_devices()
        print(f"{len(devices_list)} device(s) found: {devices_list}")

        if len(devices_list) == 0:
            raise Exception("No devices found")
        elif len(devices_list) < self.num_devices:
            raise Exception(f"Not enough devices found. Required: {self.num_devices}, Found: {len(devices_list)}")

        # 使用前num_devices个设备
        selected_devices = devices_list[:self.num_devices]
        self.devices = [
            DeviceInfo(serial=serial, is_emulator=False)
            for serial in selected_devices
        ]

        print(f"Using devices: {[device.serial for device in self.devices]}")

    def setup_devices(self, perform_initialization: bool = False, skip_app_install: bool = False):
        """设置设备
        
        Args:
            perform_initialization: 是否执行设备初始化（仅对模拟器有效）
            skip_app_install: 是否跳过应用安装（仅在 perform_initialization=True 时有效）
            
        Returns:
            List[DeviceInfo]: 设备信息列表
        """
        if self.use_emulator:
            self.setup_emulators()
        else:
            self.setup_physical_devices()

        # 对模拟器执行初始化（如果启用）
        if perform_initialization and self.use_emulator:
            self._initialize_all_devices(skip_app_install=skip_app_install)

        return self.devices
    
    def _initialize_all_devices(self, skip_app_install: bool = False):
        """初始化所有模拟器设备
        
        执行以下操作：
        - 设置系统时间为 2023-10-15
        - 安装24个必需的应用程序
        - 配置系统参数
        
        Args:
            skip_app_install: 是否跳过应用安装
        """
        if not INITIALIZER_AVAILABLE:
            print("⚠️ Device initializer not available, skipping initialization")
            return
        
        if not self.devices:
            print("⚠️ No devices to initialize")
            return
        
        print(f"\n🔧 Initializing {len(self.devices)} device(s)...")
        
        # 获取设备序列号列表
        device_serials = [device.serial for device in self.devices]
        
        # 创建初始化配置
        config = DeviceInitConfig(
            install_apps=not skip_app_install,
            datetime_base="2023-10-15 10:00:00",
            timezone="UTC",
            brightness="max",
            orientation="portrait",
        )
        
        # 并行初始化所有设备
        results = initialize_devices_parallel(
            device_serials=device_serials,
            config=config,
            skip_app_install=skip_app_install
        )
        
        # 检查结果
        failed_devices = [serial for serial, success in results.items() if not success]
        if failed_devices:
            print(f"⚠️ Some devices failed to initialize: {failed_devices}")
        else:
            print(f"✅ All {len(self.devices)} devices initialized successfully!")

    def check_device_connectivity(self, device_serial: str) -> bool:
        """检查设备连接性"""
        try:
            result = self.execute_adb("adb devices", verbose=False)
            if result == "ERROR":
                return False

            devices = self.parse_adb_devices(result)
            if device_serial not in devices or devices[device_serial] != "device":
                return False

            response = self.execute_adb(f"adb -s {device_serial} shell echo 'alive'", verbose=False)
            return response == "alive"
        except Exception:
            return False

    def restart_emulator(self, device_info: DeviceInfo) -> bool:
        """重启模拟器"""
        if not device_info.is_emulator:
            return False

        try:
            device_serial = device_info.serial
            console_port = device_info.console_port
            grpc_port = device_info.grpc_port

            print(f"Restarting emulator {device_serial}...")

            # 杀死旧进程
            self.execute_adb(f"adb -s {device_serial} emu kill", verbose=False)
            time.sleep(2)

            # 提取设备索引
            if "emulator-" in device_serial:
                port_num = int(device_serial.split("-")[1])
                device_idx = (port_num - 5554) // 2
            else:
                return False

            # 启动新的模拟器
            command = [
                self.emulator_exe,
                "-avd",
                f"{self.source_avd_name}_{device_idx}",
                "-no-snapshot-save",
                "-no-window",
                "-no-audio",
                "-port",
                str(console_port),
                "-grpc",
                str(grpc_port),
            ]

            http_proxy = os.environ.get("HTTP_PROXY")
            if http_proxy:
                command.extend(["-http-proxy", http_proxy])

            subprocess.Popen(
                command,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            print(f"Emulator {device_serial} started, waiting for it to be ready...")

            # 等待启动完成
            max_wait_time = 120
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                result = self.execute_adb("adb devices", verbose=False)
                if result != "ERROR":
                    devices = self.parse_adb_devices(result)
                    if device_serial in devices and devices[device_serial] == "device":
                        boot_result = self.execute_adb(
                            f"adb -s {device_serial} shell getprop sys.boot_completed",
                            verbose=False
                        )
                        if boot_result == "1":
                            print(f"Emulator {device_serial} is ready!")
                            return True
                time.sleep(3)

            print(f"Timeout waiting for emulator {device_serial} to be ready")
            return False

        except Exception as e:
            print(f"Error restarting emulator {device_info}: {e}")
            return False

    def check_and_restart_device_if_needed(self, device_info: DeviceInfo) -> bool:
        """检查设备状态，必要时重启"""
        if self.check_device_connectivity(device_info.serial):
            return True

        if device_info.is_emulator:
            print(f"Device {device_info.serial} is offline, attempting to restart...")
            return self.restart_emulator(device_info)
        else:
            print(f"Physical device {device_info.serial} is offline and cannot be restarted automatically")
            return False

    def terminate_emulators(self):
        """终止所有模拟器"""
        for device in self.devices:
            if device.is_emulator:
                try:
                    result = self.execute_adb(f"adb -s {device.serial} emu kill", verbose=False)
                    if result != "ERROR":
                        print(f"Successfully terminated emulator: {device.serial}")
                    else:
                        print(f"Failed to terminate emulator: {device.serial}")
                except Exception as e:
                    print(f"Exception while terminating emulator {device.serial}: {e}")

    def get_device_list(self) -> List[DeviceInfo]:
        """获取设备列表"""
        return self.devices