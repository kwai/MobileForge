"""
并行探索执行器模块
改编自MobileForge parallel exploration的并发执行机制，专门用于app探索任务

增强功能：
- 支持按需数据注入（在探索每个app前注入该app所需的数据）
"""
import os
import sys
import concurrent.futures
import queue
import threading
import time
import logging
from typing import List, Callable, Any, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 尝试相对导入（当作为模块运行时）
    from .device_manager import DeviceInfo, DeviceManager
except ImportError:
    # 回退到绝对导入（当直接运行时）
    from device_manager import DeviceInfo, DeviceManager

# 导入数据注入模块
try:
    from parallel_exploration.app_data_injector import AppDataInjector, inject_app_data
    DATA_INJECTOR_AVAILABLE = True
except ImportError:
    DATA_INJECTOR_AVAILABLE = False

from utils.device import Device
from MLLM_Agent.GUI_explorer import GUI_explorer

logger = logging.getLogger(__name__)


class ParallelExplorer:
    """并行探索执行器"""

    def __init__(self, device_manager: DeviceManager):
        """
        初始化并行探索器

        Args:
            device_manager: 设备管理器实例
        """
        self.device_manager = device_manager
        self.devices = device_manager.get_device_list()

    def run_exploration_with_multi_devices(self,
                                           exploration_func: Callable,
                                           app_package_list: List[str],
                                           **exploration_kwargs) -> None:
        """
        在多设备上并行运行探索任务

        Args:
            exploration_func: 探索函数
            app_package_list: 应用包名列表
            **exploration_kwargs: 探索函数的其他参数
        """
        if len(self.devices) == 1:
            # 单设备模式，顺序执行
            self._run_sequential_exploration(exploration_func, app_package_list, **exploration_kwargs)
        else:
            # 多设备模式，并行执行
            self._run_parallel_exploration(exploration_func, app_package_list, **exploration_kwargs)

    def _run_sequential_exploration(self,
                                    exploration_func: Callable,
                                    app_package_list: List[str],
                                    **exploration_kwargs):
        """单设备顺序执行探索"""
        device_info = self.devices[0]
        print(f"Running sequential exploration on device: {device_info.serial}")

        for i, package_name in enumerate(app_package_list):
            print(f"[{i+1}/{len(app_package_list)}] Exploring app: {package_name}")

            # 检查设备状态
            if self.device_manager.use_emulator:
                if not self.device_manager.check_and_restart_device_if_needed(device_info):
                    print(f"Device {device_info.serial} is not ready, skipping {package_name}")
                    continue

            # 执行探索任务
            try:
                exploration_func(
                    package_name=package_name,
                    device_serial=device_info.serial,
                    **exploration_kwargs
                )
                print(f"Successfully explored {package_name}")
            except Exception as e:
                print(f"Error exploring {package_name} on {device_info.serial}: {e}")

    def _run_parallel_exploration(self,
                                  exploration_func: Callable,
                                  app_package_list: List[str],
                                  **exploration_kwargs):
        """多设备并行执行探索"""
        print(f"Running parallel exploration on {len(self.devices)} devices")

        # 创建任务队列
        task_queue = queue.Queue()
        for package_name in app_package_list:
            task_queue.put(package_name)

        def worker(device_info: DeviceInfo):
            """工作线程函数"""
            while not task_queue.empty():
                try:
                    # 获取下一个包名
                    package_name = task_queue.get_nowait()
                except queue.Empty:
                    break

                max_retries = 2  # 最大重试次数
                success = False

                for attempt in range(max_retries + 1):
                    try:
                        print(f"[{device_info.serial}] Exploring {package_name} (attempt {attempt + 1})")

                        # 检查设备状态
                        if self.device_manager.use_emulator:
                            if not self.device_manager.check_and_restart_device_if_needed(device_info):
                                print(f"Device {device_info.serial} is not ready, skipping {package_name}")
                                break

                        # 执行探索任务
                        exploration_func(
                            package_name=package_name,
                            device_serial=device_info.serial,
                            **exploration_kwargs
                        )
                        success = True
                        print(f"[{device_info.serial}] Successfully explored {package_name}")
                        break

                    except Exception as e:
                        print(f"Error exploring {package_name} on {device_info.serial} (attempt {attempt + 1}): {e}")

                        # 如果还有重试机会，检查并重启设备
                        if attempt < max_retries:
                            if self.device_manager.use_emulator:
                                print(f"Checking device {device_info.serial} status after failure...")
                                if not self.device_manager.check_and_restart_device_if_needed(device_info):
                                    print(f"Failed to restart device {device_info.serial}, skipping remaining retries")
                                    break
                            else:
                                # 物理设备简单等待后重试
                                time.sleep(5)
                        else:
                            print(f"Max retries exceeded for {package_name} on {device_info.serial}")

                if not success:
                    print(f"[{device_info.serial}] Failed to explore {package_name} after all retries")

                # 标记任务完成
                task_queue.task_done()

        # 使用ThreadPoolExecutor管理并发执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            # 为每个设备提交一个worker
            futures = [executor.submit(worker, device) for device in self.devices]

            # 等待所有任务完成
            concurrent.futures.wait(futures)

        print("All exploration tasks completed")


class AppExplorationTask:
    """应用探索任务封装类"""

    def __init__(self,
                 package_name: str,
                 device_serial: str,
                 exploration_output_root_dir: str,
                 max_exploration_tasks: int,
                 max_exploration_steps: int,
                 max_exploration_depth: int,
                 usage: dict):
        """
        初始化探索任务

        Args:
            package_name: 应用包名
            device_serial: 设备序列号
            exploration_output_root_dir: 输出目录
            max_exploration_tasks: 最大探索任务数
            max_exploration_steps: 最大探索步数
            max_exploration_depth: 最大探索深度
            usage: token使用统计
        """
        self.package_name = package_name
        self.device_serial = device_serial
        self.exploration_output_root_dir = exploration_output_root_dir
        self.max_exploration_tasks = max_exploration_tasks
        self.max_exploration_steps = max_exploration_steps
        self.max_exploration_depth = max_exploration_depth
        self.usage = usage

    def execute(self) -> bool:
        """
        执行探索任务

        Returns:
            bool: 是否成功完成
        """
        try:
            from exploration_and_mining import auto_exploration

            print(f"[{self.device_serial}] Starting exploration for {self.package_name}")

            auto_exploration(
                package_name=self.package_name,
                exploration_output_root_dir=self.exploration_output_root_dir,
                device_serial=self.device_serial,
                max_exploration_tasks=self.max_exploration_tasks,
                max_exploration_steps=self.max_exploration_steps,
                max_exploration_depth=self.max_exploration_depth,
                usage=self.usage,
            )

            print(f"[{self.device_serial}] Completed exploration for {self.package_name}")
            return True

        except Exception as e:
            print(f"[{self.device_serial}] Error during exploration for {self.package_name}: {e}")
            return False


def run_batch_exploration(device_manager: DeviceManager,
                          app_package_list: List[str],
                          exploration_output_root_dir: str = "./exploration_output",
                          max_exploration_tasks: int = 10,
                          max_exploration_steps: int = 30,
                          max_exploration_depth: int = 5,
                          inject_app_data_before_exploration: bool = True) -> dict:
    """
    批量运行应用探索任务

    Args:
        device_manager: 设备管理器
        app_package_list: 应用包名列表
        exploration_output_root_dir: 输出根目录
        max_exploration_tasks: 最大探索任务数
        max_exploration_steps: 最大探索步数
        max_exploration_depth: 最大探索深度
        inject_app_data_before_exploration: 是否在探索前注入应用数据

    Returns:
        dict: 每个设备的token使用统计
    """
    usage_stats = {"prompt_tokens": 0, "completion_tokens": 0}
    usage_lock = threading.Lock()
    
    # 每个设备的数据注入器缓存（避免重复创建连接）
    device_injectors: dict = {}
    injector_lock = threading.Lock()

    def get_or_create_injector(device_serial: str) -> 'AppDataInjector':
        """获取或创建设备的数据注入器"""
        with injector_lock:
            if device_serial not in device_injectors:
                # 从设备序列号解析端口
                console_port = 5554
                grpc_port = 8554
                if device_serial.startswith("emulator-"):
                    port = int(device_serial.split("-")[1])
                    console_port = port
                    grpc_port = port + 3000
                
                device_injectors[device_serial] = AppDataInjector(
                    device_serial=device_serial,
                    console_port=console_port,
                    grpc_port=grpc_port
                )
            return device_injectors[device_serial]

    def exploration_wrapper(package_name: str, device_serial: str, **kwargs):
        """探索任务包装函数（带数据注入）"""
        from exploration_and_mining import auto_exploration

        # 在探索前注入应用数据
        if inject_app_data_before_exploration and DATA_INJECTOR_AVAILABLE:
            try:
                logger.info(f"[{device_serial}] Injecting data for {package_name}...")
                injector = get_or_create_injector(device_serial)
                injector.inject_app_data(package_name)
            except Exception as e:
                logger.warning(f"[{device_serial}] Failed to inject data for {package_name}: {e}")
                # 继续探索，即使数据注入失败
        elif inject_app_data_before_exploration and not DATA_INJECTOR_AVAILABLE:
            logger.warning(f"[{device_serial}] Data injector not available, skipping data injection")

        # 为每个任务创建独立的usage统计
        local_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        auto_exploration(
            package_name=package_name,
            exploration_output_root_dir=exploration_output_root_dir,
            device_serial=device_serial,
            max_exploration_tasks=max_exploration_tasks,
            max_exploration_steps=max_exploration_steps,
            max_exploration_depth=max_exploration_depth,
            usage=local_usage,
        )

        # 累计到全局统计中（线程安全）
        with usage_lock:
            usage_stats["prompt_tokens"] += local_usage["prompt_tokens"]
            usage_stats["completion_tokens"] += local_usage["completion_tokens"]

    # 创建并行探索器
    parallel_explorer = ParallelExplorer(device_manager)

    # 开始时间
    start_time = time.time()

    # 执行并行探索
    print(f"\n🔍 Starting batch exploration of {len(app_package_list)} apps...")
    if inject_app_data_before_exploration:
        print(f"   📥 Data injection enabled: will inject app-specific data before exploring each app")
    
    parallel_explorer.run_exploration_with_multi_devices(
        exploration_func=exploration_wrapper,
        app_package_list=app_package_list
    )

    # 结束时间
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n=== Batch Exploration Summary ===")
    print(f"Total apps explored: {len(app_package_list)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per app: {total_time/len(app_package_list):.2f} seconds")
    print(f"Data injection: {'enabled' if inject_app_data_before_exploration else 'disabled'}")
    print(f"Token usage: {usage_stats}")

    return usage_stats