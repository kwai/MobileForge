"""
Comprehensive Device Setup - 完整设备初始化模块

该模块完全采用 AndroidWorld 原生方法进行设备初始化，包括：
- 连接模拟器
- 配置系统设置（日期时间、时区等）
- 安装和配置所有24个必需的应用程序
- 注入确定性数据

参考: reference/MobileForge Emulator Setup/android_world/comprehensive_setup/comprehensive_device_setup.py
"""

import datetime
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List

# 将本地 android_env 模块路径添加到 sys.path（优先于 pip 安装的版本）
ANDROID_ENV_PATH = os.path.join(os.path.dirname(__file__), "android_env")
if ANDROID_ENV_PATH not in sys.path:
    sys.path.insert(0, ANDROID_ENV_PATH)

# 将 AndroidWorld 模块路径添加到 sys.path
ANDROID_WORLD_PATH = os.path.join(os.path.dirname(__file__), "models", "AndroidWorld")
if ANDROID_WORLD_PATH not in sys.path:
    sys.path.insert(0, ANDROID_WORLD_PATH)

# 导入 AndroidWorld 组件
from android_world.env import env_launcher
from android_world.env import interface
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env.setup_device import setup, apps
from android_world.utils import datetime_utils
from android_world.utils import app_snapshot

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AndroidWorld 标准基准时间
DEVICE_BASE_DATETIME = device_constants.DT


class ComprehensiveDeviceSetup:
    """
    完整设备初始化类，使用 AndroidWorld 原生方法。
    
    参考 reference/MobileForge Emulator Setup/android_world/comprehensive_setup/comprehensive_device_setup.py
    """
    
    def __init__(
        self,
        console_port: int = 5554,
        grpc_port: int = 8554,
        adb_path: str = "adb",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化设备设置器。
        
        Args:
            console_port: 模拟器控制台端口
            grpc_port: gRPC 端口
            adb_path: ADB 路径
            config: 可选配置
        """
        self.console_port = console_port
        self.grpc_port = grpc_port
        self.adb_path = self._find_adb_path(adb_path)
        self.config = config or {}
        self.env: Optional[interface.AsyncEnv] = None
        self.logger = logging.getLogger(__name__)
    
    def _find_adb_path(self, default_path: str = "adb") -> str:
        """查找系统中的 ADB 路径。"""
        import subprocess
        
        potential_paths = [
            os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
            os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
            '/usr/local/bin/adb',
            default_path
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
        
        return default_path
    
    def run_setup(
        self,
        install_apps: bool = True,
        setup_datetime: bool = True,
        inject_data: bool = True
    ) -> bool:
        """
        执行完整的设备初始化流程。
        
        Steps:
        1. 连接模拟器
        2. 配置系统设置（日期时间、时区等）
        3. 安装和配置所有24个应用程序
        4. 注入确定性数据
        5. 验证设置完成
        
        Args:
            install_apps: 是否安装应用程序
            setup_datetime: 是否设置系统时间
            inject_data: 是否注入数据
            
        Returns:
            是否成功
        """
        start_time = time.time()
        
        try:
            self.logger.info("开始完整设备初始化...")
            self.logger.info(f"端口配置: console={self.console_port}, grpc={self.grpc_port}")
            
            # Step 1: 连接模拟器
            self.logger.info("[1/5] 连接模拟器...")
            self._verify_emulator_connection()
            
            # Step 2: 配置系统设置
            if setup_datetime:
                self.logger.info("[2/5] 配置系统设置...")
                self._setup_system_settings()
            
            # Step 3: 安装应用程序（已安装的自动跳过）
            if install_apps:
                self.logger.info("[3/6] 安装24个应用程序（已安装的自动跳过）...")
                self._install_apps_only()
            
            # Step 4: 初始化应用程序（pm clear + wizard + 权限）
            if install_apps:
                self.logger.info("[4/6] 初始化应用程序（清空数据、点击引导页、授权等）...")
                self._initialize_apps_post_install()
            
            # Step 5: 注入确定性数据
            if inject_data:
                self.logger.info("[5/6] 注入确定性数据...")
                self._inject_comprehensive_data()
            
            # Step 6: 验证设置
            self.logger.info("[6/6] 验证设置...")
            self._verify_setup_completion()
            
            # 返回主屏幕
            adb_utils.press_home_button(self.env.controller)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"设备初始化完成，耗时: {elapsed_time:.1f}秒")
            return True
            
        except Exception as e:
            self.logger.error(f"设备初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.env:
                self.env.close()
                self.env = None
    
    def _verify_emulator_connection(self) -> None:
        """
        连接并验证模拟器。
        
        使用 AndroidWorld 的 env_launcher.load_and_setup_env()。
        """
        try:
            self.logger.info(f"使用 ADB: {self.adb_path}")
            
            # 使用 AndroidWorld 原生方法创建环境
            self.env = env_launcher.load_and_setup_env(
                console_port=self.console_port,
                emulator_setup=False,  # 我们会手动处理应用设置
                freeze_datetime=False,  # 我们会手动处理日期时间
                adb_path=self.adb_path,
                grpc_port=self.grpc_port
            )
            
            # 验证连接
            screen_size = self.env.controller.device_screen_size
            self.logger.info(f"已连接到设备，屏幕尺寸: {screen_size}")
            
        except Exception as e:
            raise RuntimeError(f"连接模拟器失败: {e}")
    
    def _setup_system_settings(self) -> None:
        """
        配置系统设置。
        
        使用 AndroidWorld 的 datetime_utils.setup_datetime()。
        """
        # 确保有 root 权限
        adb_utils.set_root_if_needed(self.env.controller)
        
        # 使用 AndroidWorld 原生的日期时间设置
        # 这会：禁用自动时间/时区、设置24小时制、设置时区为UTC
        # 注意：setup_datetime/set_datetime 接受 AndroidEnvInterface，controller (AndroidWorldController) 也是其子类
        datetime_utils.setup_datetime(self.env.controller)
        self.logger.info("已禁用自动时间/时区，启用24小时制，设置时区为 UTC")
        
        # 设置固定日期时间（AndroidWorld 标准：2023-10-15 10:00:00）
        datetime_utils.set_datetime(self.env.controller, DEVICE_BASE_DATETIME)
        self.logger.info(f"已设置设备日期时间为: {DEVICE_BASE_DATETIME}")
        
        # 其他系统设置
        self._configure_additional_settings()
    
    def _configure_additional_settings(self) -> None:
        """配置其他系统设置。"""
        try:
            # 设置屏幕亮度为最大
            adb_utils.set_brightness('max', self.env.controller)
            
            # 设置屏幕方向为竖屏
            adb_utils.change_orientation('portrait', self.env.controller)
            
            self.logger.info("已配置屏幕亮度和方向")
        except Exception as e:
            self.logger.warning(f"配置额外设置时出现警告: {e}")
    
    def _install_apps_only(self) -> None:
        """
        仅安装所有24个必需的应用程序（已安装的自动跳过）。
        
        与 reference/MobileForge Explore 对齐：将安装与初始化分离。
        安装步骤使用 AndroidWorld 的 maybe_install_app()，内部会检测
        应用是否已安装，已安装则跳过下载和安装。
        """
        self.logger.info("安装应用程序（已安装的自动跳过）...")
        self.logger.info("这可能需要几分钟时间...")
        
        # 确保回到主屏幕，避免快速设置菜单阻挡UI导航
        adb_utils.press_home_button(self.env.controller)
        adb_utils.set_root_if_needed(self.env.controller)
        
        # 使用 AndroidWorld 原生的 maybe_install_app() 逐个安装
        # maybe_install_app() 内部会检测应用是否已安装，已安装则跳过
        from android_world.env.setup_device.setup import _APPS
        for app in _APPS:
            try:
                setup.maybe_install_app(app, self.env)
            except Exception as e:
                self.logger.warning(f"安装应用 {app.app_name} 时出现警告: {e}")
        
        self.logger.info("应用程序安装完成（已安装的自动跳过）")
    
    def _initialize_apps_post_install(self) -> None:
        """
        安装后的应用初始化（pm clear + wizard + 权限）。
        
        与 reference/MobileForge Explore/parallel_exploration/app_initializer.py 对齐。
        对 12 个需要特殊初始化的应用执行：
        - 清空应用数据
        - 点击引导页/授权弹窗
        - 确保应用处于可用状态
        
        参考 Android World 的 apps.py 中每个 App 的 setup() 方法。
        """
        from android_world.env import tools
        
        self.logger.info("开始应用初始化（pm clear + wizard + 权限）...")
        
        # 确保回到主屏幕
        adb_utils.press_home_button(self.env.controller)
        
        # 需要初始化的应用列表及其初始化方法
        # 与 reference/MobileForge Explore/parallel_exploration/app_initializer.py 对齐
        init_results = {}
        
        # 1. Markor: pm clear + NEXT x4 + DONE + OK + 文件访问权限
        init_results['markor'] = self._init_app_markor(tools)
        
        # 2. Simple Gallery Pro: pm clear + 权限 + 引导页
        init_results['simple_gallery_pro'] = self._init_app_simple_gallery_pro(tools)
        
        # 3. Joplin: pm clear + monkey 启动 + 创建测试笔记 + 清空DB
        init_results['joplin'] = self._init_app_joplin(tools)
        
        # 4. OpenTracks: 启动 + 位置权限 + 蓝牙允许
        init_results['opentracks'] = self._init_app_opentracks(tools)
        
        # 5. Retro Music: pm clear + 音频权限
        init_results['retro_music'] = self._init_app_retro_music(tools)
        
        # 6. Camera: pm clear + 位置权限 + NEXT
        init_results['camera'] = self._init_app_camera(tools)
        
        # 7. Chrome: pm clear + Accept & continue + No thanks x2
        init_results['chrome'] = self._init_app_chrome(tools)
        
        # 8. Contacts: pm clear + Skip + Don't allow
        init_results['contacts'] = self._init_app_contacts(tools)
        
        # 9. Clock: pm clear + 启动一次
        init_results['clock'] = self._init_app_clock(tools)
        
        # 10. Simple Calendar Pro: pm clear + 日历权限
        init_results['simple_calendar_pro'] = self._init_app_simple_calendar_pro(tools)
        
        # 11. Tasks: pm clear + 启动一次
        init_results['tasks'] = self._init_app_tasks(tools)
        
        # 12. Broccoli: pm clear + 启动一次
        init_results['broccoli'] = self._init_app_broccoli(tools)
        
        # 13. Simple SMS Messenger: pm clear + 设为默认 + UI 确认 + 保存快照
        # 与 reference/android_world 对齐：初始化后保存快照，任务前恢复快照
        init_results['sms_messenger'] = self._init_app_sms_messenger(tools)
        
        # 统计结果
        success_count = sum(1 for v in init_results.values() if v)
        total_count = len(init_results)
        self.logger.info(f"应用初始化完成: {success_count}/{total_count} 个成功")
        
        failed = [k for k, v in init_results.items() if not v]
        if failed:
            self.logger.warning(f"初始化失败的应用: {failed}")
    
    # ====================================================================
    # 各应用初始化方法
    # 参考: reference/MobileForge Explore/parallel_exploration/app_initializer.py
    # ====================================================================
    
    def _init_app_markor(self, tools_module) -> bool:
        """初始化 Markor: pm clear + NEXT x4 + DONE + OK + 文件访问权限"""
        package_name = 'net.gsantner.markor'
        app_name = 'markor'
        try:
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            adb_utils.launch_app(app_name, self.env.controller)
            time.sleep(2.0)
            controller = tools_module.AndroidToolController(env=self.env.controller)
            for _ in range(4):
                try:
                    controller.click_element("NEXT")
                    time.sleep(2.0)
                except Exception:
                    break
            try:
                controller.click_element("DONE")
                time.sleep(2.0)
            except Exception:
                pass
            try:
                controller.click_element("OK")
                time.sleep(2.0)
            except Exception:
                pass
            try:
                controller.click_element("Allow access to manage all files")
                time.sleep(2.0)
            except Exception:
                pass
            adb_utils.close_app(app_name, self.env.controller)
            self.logger.info("Markor 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"Markor 初始化失败: {e}")
            return False
    
    def _init_app_simple_gallery_pro(self, tools_module) -> bool:
        """初始化 Simple Gallery Pro: pm clear + 权限 + 引导页"""
        package_name = 'com.simplemobiletools.gallery.pro'
        app_name = 'simple gallery pro'
        permissions = [
            "android.permission.WRITE_EXTERNAL_STORAGE",
            "android.permission.ACCESS_MEDIA_LOCATION",
            "android.permission.READ_MEDIA_IMAGES",
            "android.permission.READ_MEDIA_VIDEO",
            "android.permission.POST_NOTIFICATIONS",
        ]
        try:
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            for permission in permissions:
                try:
                    adb_utils.grant_permissions(package_name, permission, self.env.controller)
                except Exception:
                    pass
            adb_utils.launch_app(app_name, self.env.controller)
            time.sleep(2.0)
            controller = tools_module.AndroidToolController(env=self.env.controller)
            try:
                controller.click_element("All files")
                time.sleep(2.0)
            except Exception:
                pass
            try:
                controller.click_element("Allow access to manage all files")
                time.sleep(2.0)
            except Exception:
                pass
            adb_utils.close_app(app_name, self.env.controller)
            self.logger.info("Simple Gallery Pro 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"Simple Gallery Pro 初始化失败: {e}")
            return False
    
    def _init_app_joplin(self, tools_module) -> bool:
        """初始化 Joplin: pm clear + monkey 启动 + 创建测试笔记 + 清空DB"""
        package_name = 'net.cozic.joplin'
        app_name = 'joplin'
        try:
            from android_world.task_evals.information_retrieval import joplin_app_utils
            
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            for permission in ["android.permission.ACCESS_COARSE_LOCATION",
                              "android.permission.ACCESS_FINE_LOCATION"]:
                try:
                    adb_utils.grant_permissions(package_name, permission, self.env.controller)
                except Exception:
                    pass
            # 使用 monkey 命令启动应用（触发数据库初始化）
            adb_utils.issue_generic_request(
                ["shell", "monkey", "-p", package_name,
                 "-candroid.intent.category.LAUNCHER", "1"],
                self.env.controller
            )
            time.sleep(10.0)
            adb_utils.close_app(app_name, self.env.controller)
            time.sleep(10.0)
            # 创建测试笔记并清空数据库（确保表可访问）
            try:
                joplin_app_utils.create_note(
                    folder="new folder", title="new_note", body="",
                    folder_mapping={}, env=self.env
                )
                joplin_app_utils.clear_dbs(self.env)
            except Exception as e:
                self.logger.debug(f"创建测试笔记时出现警告: {e}")
            self.logger.info("Joplin 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"Joplin 初始化失败: {e}")
            return False
    
    def _init_app_opentracks(self, tools_module) -> bool:
        """初始化 OpenTracks: 启动 + 位置权限 + 蓝牙允许"""
        package_name = 'de.dennisguse.opentracks'
        app_name = 'open tracks sports tracker'
        try:
            adb_utils.launch_app(app_name, self.env.controller)
            adb_utils.close_app(app_name, self.env.controller)
            for permission in ["android.permission.ACCESS_COARSE_LOCATION",
                              "android.permission.ACCESS_FINE_LOCATION",
                              "android.permission.POST_NOTIFICATIONS"]:
                try:
                    adb_utils.grant_permissions(package_name, permission, self.env.controller)
                except Exception:
                    pass
            time.sleep(2.0)
            controller = tools_module.AndroidToolController(env=self.env.controller)
            try:
                controller.click_element("Allow")
            except Exception:
                pass
            # 再次启动并关闭
            adb_utils.launch_app("activity tracker", self.env.controller)
            adb_utils.close_app("activity tracker", self.env.controller)
            self.logger.info("OpenTracks 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"OpenTracks 初始化失败: {e}")
            return False
    
    def _init_app_retro_music(self, tools_module) -> bool:
        """初始化 Retro Music: pm clear + 音频权限"""
        package_name = 'code.name.monkey.retromusic'
        app_name = 'retro music'
        permissions = [
            "android.permission.READ_MEDIA_AUDIO",
            "android.permission.POST_NOTIFICATIONS",
        ]
        try:
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            for permission in permissions:
                try:
                    adb_utils.grant_permissions(package_name, permission, self.env.controller)
                except Exception:
                    pass
            adb_utils.launch_app(app_name, self.env.controller)
            time.sleep(2.0)
            adb_utils.close_app(app_name, self.env.controller)
            self.logger.info("Retro Music 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"Retro Music 初始化失败: {e}")
            return False
    
    def _init_app_camera(self, tools_module) -> bool:
        """初始化 Camera: pm clear + 位置权限 + NEXT"""
        package_name = 'com.android.camera2'
        app_name = 'camera'
        try:
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            try:
                adb_utils.grant_permissions(
                    package_name, "android.permission.ACCESS_COARSE_LOCATION",
                    self.env.controller
                )
            except Exception:
                pass
            adb_utils.launch_app(app_name, self.env.controller)
            time.sleep(2.0)
            controller = tools_module.AndroidToolController(env=self.env.controller)
            try:
                controller.click_element("NEXT")
                time.sleep(2.0)
            except Exception:
                pass
            adb_utils.close_app(app_name, self.env.controller)
            self.logger.info("Camera 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"Camera 初始化失败: {e}")
            return False
    
    def _init_app_chrome(self, tools_module) -> bool:
        """初始化 Chrome: pm clear + Accept & continue + No thanks x2"""
        package_name = 'com.android.chrome'
        app_name = 'chrome'
        try:
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            adb_utils.launch_app(app_name, self.env.controller)
            time.sleep(2.0)
            controller = tools_module.AndroidToolController(env=self.env.controller)
            try:
                controller.click_element("Accept & continue")
                time.sleep(2.0)
            except Exception:
                pass
            try:
                controller.click_element("No thanks")
                time.sleep(2.0)
            except Exception:
                pass
            try:
                controller.click_element("No thanks")
                time.sleep(2.0)
            except Exception:
                pass
            adb_utils.close_app(app_name, self.env.controller)
            self.logger.info("Chrome 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"Chrome 初始化失败: {e}")
            return False
    
    def _init_app_contacts(self, tools_module) -> bool:
        """初始化 Contacts: pm clear + Skip + Don't allow"""
        package_name = 'com.google.android.contacts'
        app_name = 'contacts'
        try:
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            adb_utils.launch_app(app_name, self.env.controller)
            time.sleep(2.0)
            controller = tools_module.AndroidToolController(env=self.env.controller)
            try:
                controller.click_element("Skip")
                time.sleep(2.0)
            except Exception:
                pass
            try:
                controller.click_element("Don't allow")
                time.sleep(2.0)
            except Exception:
                pass
            adb_utils.close_app(app_name, self.env.controller)
            self.logger.info("Contacts 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"Contacts 初始化失败: {e}")
            return False
    
    def _init_app_clock(self, tools_module) -> bool:
        """初始化 Clock: pm clear + 启动一次"""
        package_name = 'com.google.android.deskclock'
        app_name = 'clock'
        try:
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            adb_utils.launch_app(app_name, self.env.controller)
            time.sleep(2.0)
            adb_utils.close_app(app_name, self.env.controller)
            self.logger.info("Clock 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"Clock 初始化失败: {e}")
            return False
    
    def _init_app_simple_calendar_pro(self, tools_module) -> bool:
        """初始化 Simple Calendar Pro: pm clear + 日历权限"""
        package_name = 'com.simplemobiletools.calendar.pro'
        app_name = 'simple calendar pro'
        permissions = [
            "android.permission.READ_CALENDAR",
            "android.permission.WRITE_CALENDAR",
            "android.permission.POST_NOTIFICATIONS",
        ]
        try:
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            adb_utils.launch_app(app_name, self.env.controller)
            adb_utils.close_app(app_name, self.env.controller)
            for permission in permissions:
                try:
                    adb_utils.grant_permissions(package_name, permission, self.env.controller)
                except Exception:
                    pass
            self.logger.info("Simple Calendar Pro 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"Simple Calendar Pro 初始化失败: {e}")
            return False
    
    def _init_app_tasks(self, tools_module) -> bool:
        """初始化 Tasks: pm clear + 启动一次"""
        package_name = 'org.tasks'
        app_name = 'tasks'
        try:
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            adb_utils.launch_app(app_name, self.env.controller)
            adb_utils.close_app(app_name, self.env.controller)
            self.logger.info("Tasks 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"Tasks 初始化失败: {e}")
            return False
    
    def _init_app_broccoli(self, tools_module) -> bool:
        """初始化 Broccoli: pm clear + 启动一次"""
        package_name = 'com.flauschcode.broccoli'
        app_name = 'broccoli app'
        try:
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            adb_utils.launch_app(app_name, self.env.controller)
            time.sleep(2.0)
            adb_utils.close_app(app_name, self.env.controller)
            self.logger.info("Broccoli 初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"Broccoli 初始化失败: {e}")
            return False
    
    def _init_app_sms_messenger(self, tools_module) -> bool:
        """
        初始化 Simple SMS Messenger: pm clear + 设为默认 + UI 确认 + 保存快照。
        
        与 reference/android_world 的 SimpleSMSMessengerApp.setup() 对齐：
        1. pm clear 清空应用数据
        2. 通过 settings 命令设置为默认 SMS 应用
        3. 启动应用，通过 AndroidToolController 点击 UI 确认默认应用
        4. 关闭应用
        5. 保存快照（供后续任务前恢复使用）
        
        参考: reference/android_world/android_world/env/setup_device/apps.py SimpleSMSMessengerApp
        """
        sms_package = 'com.simplemobiletools.smsmessenger'
        sms_app_name = 'simple sms messenger'
        try:
            # 步骤1：pm clear 清空应用数据（与 reference AppSetup.setup() 基类一致）
            adb_utils.clear_app_data(sms_package, self.env.controller)
            time.sleep(1)
            
            # 步骤2：设置为默认 SMS 应用（与 reference 一致）
            adb_utils.set_default_app(
                "sms_default_application",
                sms_package,
                self.env.controller,
            )
            self.logger.info(f"已设置 {sms_package} 为默认 SMS 应用")
            
            # 步骤3：启动应用并通过 UI 确认设置为默认
            # 使用 AndroidToolController（与 reference 一致，比 uiautomator dump 更可靠）
            adb_utils.launch_app(sms_app_name, self.env.controller)
            try:
                controller = tools_module.AndroidToolController(env=self.env.controller)
                time.sleep(2.0)
                controller.click_element("SMS Messenger")
                time.sleep(2.0)
                controller.click_element("Set as default")
            except Exception as ui_err:
                self.logger.debug(f"SMS 默认应用 UI 确认时出现警告（可能已是默认）: {ui_err}")
            finally:
                adb_utils.close_app(sms_app_name, self.env.controller)
            
            # 步骤4：保存快照（供后续任务前恢复使用）
            # 与 reference/android_world setup.py setup_app() 的 save_snapshot 一致
            app_snapshot.save_snapshot(sms_app_name, self.env.controller)
            self.logger.info("Simple SMS Messenger 初始化完成并已保存快照")
            return True
        except Exception as e:
            self.logger.warning(f"Simple SMS Messenger 初始化失败: {e}")
            return False
    
    def _inject_comprehensive_data(self) -> None:
        """
        注入确定性数据。
        
        使用 AppDataInjector 类。
        """
        self.logger.info("开始注入确定性数据...")
        
        from .native_app_injector import AppDataInjector
        
        # 创建数据注入器
        injector = AppDataInjector(self.env, self.config)
        
        # 注入所有数据
        results = injector.inject_all_data()
        
        # 统计结果
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        self.logger.info(f"数据注入完成: {success_count}/{total_count} 个数据类型成功")
        
        # 验证注入结果
        injector.verify_all_data_injection()
    
    def _verify_setup_completion(self) -> None:
        """验证设置完成。"""
        self.logger.info("验证设置完成...")
        
        try:
            # 检查已安装的应用
            from android_world.env.setup_device.setup import get_installed_packages
            installed_packages = get_installed_packages(self.env)
            
            # 24个必需应用的包名
            required_apps = {
                'com.google.android.apps.androidworld': 'AndroidWorldApp',
                'com.dimowner.audiorecorder': 'AudioRecorder',
                'com.android.camera2': 'CameraApp',
                'com.android.chrome': 'ChromeApp',
                'org.nicbear.clipper': 'ClipperApp',
                'com.google.android.deskclock': 'ClockApp',
                'com.google.android.contacts': 'ContactsApp',
                'com.google.android.dialer': 'DialerApp',
                'com.arduia.expense': 'ExpenseApp',
                'com.google.android.documentsui': 'FilesApp',
                'net.cozic.joplin': 'JoplinApp',
                'net.gsantner.markor': 'MarkorApp',
                'com.nicbear.miniwob': 'MiniWobApp',
                'de.dennisguse.opentracks': 'OpenTracksApp',
                'net.osmand': 'OsmAndApp',
                'com.flauschcode.broccoli': 'RecipeApp',
                'code.name.monkey.retromusic': 'RetroMusicApp',
                'com.android.settings': 'SettingsApp',
                'com.simplemobiletools.calendar.pro': 'SimpleCalendarProApp',
                'com.simplemobiletools.draw.pro': 'SimpleDrawProApp',
                'com.simplemobiletools.gallery.pro': 'SimpleGalleryProApp',
                'com.simplemobiletools.smsmessenger': 'SimpleSMSMessengerApp',
                'org.tasks': 'TasksApp',
                'org.videolan.vlc': 'VlcApp',
            }
            
            installed_count = 0
            missing_apps = []
            
            for package, app_name in required_apps.items():
                if package in installed_packages:
                    installed_count += 1
                else:
                    missing_apps.append(f"{app_name} ({package})")
            
            self.logger.info(f"已验证 {installed_count}/{len(required_apps)} 个应用已安装")
            
            if missing_apps:
                self.logger.warning(f"缺少的应用: {missing_apps}")
            else:
                self.logger.info("所有24个必需应用已成功安装！")
                
        except Exception as e:
            self.logger.warning(f"验证应用安装时出现警告: {e}")
    
    def get_env(self) -> Optional[interface.AsyncEnv]:
        """获取当前的环境对象。"""
        return self.env


def initialize_device(
    console_port: int = 5554,
    grpc_port: int = 8554,
    adb_path: str = "adb",
    install_apps: bool = True,
    setup_datetime: bool = True,
    inject_data: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    便捷函数：初始化设备。
    
    Args:
        console_port: 控制台端口
        grpc_port: gRPC 端口
        adb_path: ADB 路径
        install_apps: 是否安装应用
        setup_datetime: 是否设置时间
        inject_data: 是否注入数据
        config: 可选配置
        
    Returns:
        是否成功
    """
    setup = ComprehensiveDeviceSetup(
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        config=config
    )
    return setup.run_setup(
        install_apps=install_apps,
        setup_datetime=setup_datetime,
        inject_data=inject_data
    )


def initialize_device_from_serial(
    device_serial: str,
    install_apps: bool = True,
    setup_datetime: bool = True,
    inject_data: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    从设备序列号初始化设备。
    
    Args:
        device_serial: 设备序列号（如 "emulator-5554"）
        install_apps: 是否安装应用
        setup_datetime: 是否设置时间
        inject_data: 是否注入数据
        config: 可选配置
        
    Returns:
        是否成功
    """
    # 从序列号提取端口
    console_port = 5554
    grpc_port = 8554
    
    try:
        if device_serial.startswith("emulator-"):
            port_str = device_serial.replace("emulator-", "")
            console_port = int(port_str)
            grpc_port = console_port + 3000
        elif ":" in device_serial:
            port_str = device_serial.split(":")[-1]
            console_port = int(port_str)
            grpc_port = console_port + 3000
    except ValueError:
        pass
    
    return initialize_device(
        console_port=console_port,
        grpc_port=grpc_port,
        install_apps=install_apps,
        setup_datetime=setup_datetime,
        inject_data=inject_data,
        config=config
    )


def _get_a11y_forwarder_apk_cached() -> Optional[str]:
    """
    尝试从本地目录找到已缓存的 accessibility forwarder APK。
    如果找到了（且文件有效），patch android_env 让它直接用本地文件；
    如果找不到，返回 None，让 android_env 走内置下载逻辑（每个进程只下一次）。
    """
    _FW_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _SEARCH_DIRS = [
        os.path.join(_FW_PATH, ".cache", "a11y_apk"),
        os.path.join(os.path.expanduser("~"), ".cache", "a11y_apk"),
        os.path.join(os.environ.get("ANDROID_SDK_ROOT", ""), "a11y_apk"),
        os.path.join(os.environ.get("HOME", ""), "a11y_apk"),
        "/tmp/a11y_apk",
    ]

    for d in _SEARCH_DIRS:
        for fname in ("accessibility_forwarder.apk", "a11y_forwarder.apk"):
            p = os.path.join(d, fname)
            if os.path.exists(p) and os.path.getsize(p) > 1024 * 1024:
                logger.info(f"找到本地缓存 APK: {p} ({os.path.getsize(p) / 1024 / 1024:.1f} MB)")
                return p

    logger.info("未找到本地 APK 缓存，将由 android_env 内置逻辑下载（每个进程只下一次）")
    return None


def _patch_a11y_module_use_cached_apk(cached_apk_path: str) -> None:
    """
    将 android_env.a11y_grpc_wrapper._get_accessibility_forwarder_apk()
    替换为返回本地缓存路径，避免重复下载。
    """
    try:
        from android_env.wrappers import a11y_grpc_wrapper

        def cached_getter():
            return cached_apk_path

        a11y_grpc_wrapper._get_accessibility_forwarder_apk = cached_getter
        logger.info("已 Patch: a11y_grpc_wrapper 使用本地缓存 APK")
    except Exception as e:
        logger.warning(f"Patch a11y_grpc_wrapper 失败: {e}")


def initialize_devices_parallel(
    devices: List[Dict[str, Any]],
    install_apps: bool = False,
    setup_datetime: bool = True,
    inject_data: bool = False,
    max_workers: int = None
) -> Dict[str, bool]:
    """
    并行初始化多个设备。

    Args:
        devices: 设备列表，每个设备包含 console_port, grpc_port 等
        install_apps: 是否安装应用程序
        setup_datetime: 是否设置系统时间
        inject_data: 是否注入数据
        max_workers: 最大并行数

    Returns:
        每个设备的初始化结果
    """
    if not devices:
        return {}

    # 尝试从本地找 APK 缓存；如果有就 patch 模块，绕过网络下载
    cached_apk = _get_a11y_forwarder_apk_cached()
    if cached_apk:
        _patch_a11y_module_use_cached_apk(cached_apk)

    if max_workers is None:
        # 默认并发数从 8 降到 2，避免多路并发下载 APK 导致网络被打爆
        # 如仍不稳定，可设为 1（完全串行）或 2（推荐）
        max_workers = min(2, len(devices))
        logger.info(f"自动设置 max_workers={max_workers}（原设备数={len(devices)}）")

    logger.info(f"开始并行初始化 {len(devices)} 个设备 (max_workers={max_workers})")
    
    def init_single_device(device: Dict[str, Any]) -> tuple:
        """初始化单个设备"""
        console_port = device.get("console_port", 5554)
        grpc_port = device.get("grpc_port", console_port + 3000)
        serial = device.get("serial", f"emulator-{console_port}")
        
        try:
            logger.info(f"[{serial}] 开始初始化...")
            success = initialize_device(
                console_port=console_port,
                grpc_port=grpc_port,
                install_apps=install_apps,
                setup_datetime=setup_datetime,
                inject_data=inject_data
            )
            status = "成功" if success else "失败"
            logger.info(f"[{serial}] 初始化{status}")
            return (serial, success)
        except Exception as e:
            logger.error(f"[{serial}] 初始化异常: {e}")
            return (serial, False)
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(init_single_device, device) for device in devices]
        
        for future in as_completed(futures):
            try:
                serial, success = future.result()
                if serial:
                    results[serial] = success
            except Exception as e:
                logger.error(f"获取初始化结果时出错: {e}")
    
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"并行初始化完成: {success_count}/{len(results)} 个设备成功")
    
    return results


def create_env_for_device(
    console_port: int = 5554,
    grpc_port: int = 8554,
    adb_path: str = "adb"
) -> interface.AsyncEnv:
    """
    为设备创建 AndroidWorld 环境对象。
    
    此函数返回的 env 对象可用于数据注入和任务执行。
    调用者负责在使用完毕后调用 env.close()。
    
    Args:
        console_port: 控制台端口
        grpc_port: gRPC 端口
        adb_path: ADB 路径
        
    Returns:
        AsyncEnv 对象
    """
    return env_launcher.load_and_setup_env(
        console_port=console_port,
        emulator_setup=False,
        freeze_datetime=False,
        adb_path=adb_path,
        grpc_port=grpc_port
    )


def create_env_from_serial(device_serial: str, adb_path: str = "adb") -> interface.AsyncEnv:
    """
    从设备序列号创建 AndroidWorld 环境对象。
    
    Args:
        device_serial: 设备序列号
        adb_path: ADB 路径
        
    Returns:
        AsyncEnv 对象
    """
    console_port = 5554
    grpc_port = 8554
    
    try:
        if device_serial.startswith("emulator-"):
            port_str = device_serial.replace("emulator-", "")
            console_port = int(port_str)
            grpc_port = console_port + 3000
        elif ":" in device_serial:
            port_str = device_serial.split(":")[-1]
            console_port = int(port_str)
            grpc_port = console_port + 3000
    except ValueError:
        pass
    
    return create_env_for_device(console_port, grpc_port, adb_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="完整设备初始化")
    parser.add_argument("--console-port", type=int, default=5554, help="控制台端口")
    parser.add_argument("--grpc-port", type=int, default=8554, help="gRPC 端口")
    parser.add_argument("--no-apps", action="store_true", help="跳过应用安装")
    parser.add_argument("--no-data", action="store_true", help="跳过数据注入")
    
    args = parser.parse_args()
    
    success = initialize_device(
        console_port=args.console_port,
        grpc_port=args.grpc_port,
        install_apps=not args.no_apps,
        inject_data=not args.no_data
    )
    
    sys.exit(0 if success else 1)
