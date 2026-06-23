"""
应用数据注入模块

在探索特定应用前，按需注入该应用所需的确定性数据。
支持14个应用的数据注入：

系统级应用：
- Contacts: 50个联系人
- SMS Messenger: 短信对话
- Dialer: 通话记录
- Gallery: 分类相册
- Files: 文件系统

第三方应用：
- Recipe (Broccoli): 39个食谱
- Tasks: 20个任务
- Joplin: 300+笔记
- OpenTracks: 16类运动
- Calendar: 25个事件
- Markor: 10个文档
- Expense: 30条记录
- Retro Music: 音乐文件
- VLC: 视频文件
"""

import logging
import subprocess
import time
import os
import sys
import re
from typing import Optional, List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入确定性数据模块
from parallel_exploration.deterministic_data import (
    get_all_contacts,
    get_deterministic_sms_conversations,
    get_deterministic_call_history,
    get_deterministic_gallery_images,
    get_deterministic_file_structure,
    get_deterministic_file_contents,
    get_all_recipes,
    create_deterministic_tasks,
    get_all_joplin_folders,
    get_all_joplin_notes,
    create_deterministic_activities,
    create_deterministic_calendar_events,
    get_all_markor_documents,
    create_deterministic_expenses,
    get_deterministic_music_files,
    get_deterministic_video_files,
    get_data_types_for_package,
    ANDROID_WORLD_AVAILABLE,
)

# 尝试导入 android_world 模块
if ANDROID_WORLD_AVAILABLE:
    from android_world.env import env_launcher
    from android_world.env import interface
    from android_world.env import adb_utils
    from android_world.env import device_constants
    from android_world.env import tools  # AndroidToolController for UI automation
    from android_world.task_evals.utils import sqlite_utils
    from android_world.task_evals.information_retrieval import joplin_app_utils
    from android_world.task_evals.information_retrieval import task_app_utils
    from android_world.task_evals.information_retrieval import activity_app_utils
    from android_world.task_evals.single.calendar import calendar_utils
    from android_world.task_evals.utils import user_data_generation
    from android_world.utils import file_utils
    from android_world.utils import contacts_utils


class AppDataInjector:
    """应用数据注入器
    
    根据应用包名，注入该应用所需的确定性数据。
    """
    
    def __init__(self, device_serial: str, console_port: int = 5554, grpc_port: int = 8554):
        """
        初始化数据注入器
        
        Args:
            device_serial: 设备序列号
            console_port: 控制台端口
            grpc_port: gRPC端口
        """
        self.device_serial = device_serial
        self.console_port = console_port
        self.grpc_port = grpc_port
        self.logger = logging.getLogger(__name__)
        self.env: Optional[interface.AsyncEnv] = None
        
        # 从设备序列号解析端口
        if device_serial.startswith("emulator-"):
            port = int(device_serial.split("-")[1])
            self.console_port = port
            self.grpc_port = port + 3000
        
        # 已注入数据的缓存，避免重复注入
        self._injected_data: Dict[str, bool] = {}
    
    def inject_app_data(self, package_name: str) -> bool:
        """
        为指定应用注入所需数据
        
        Args:
            package_name: 应用包名
            
        Returns:
            bool: 是否成功
        """
        data_types = get_data_types_for_package(package_name)
        
        if not data_types:
            self.logger.info(f"No data injection needed for {package_name}")
            return True
        
        self.logger.info(f"Injecting data for {package_name}: {data_types}")
        print(f"📥 Injecting data for {package_name}...")
        
        try:
            # 连接设备
            if not self._ensure_connected():
                self.logger.error("Failed to connect to device")
                return False
            
            success = True
            for data_type in data_types:
                # 检查是否已注入
                if self._injected_data.get(data_type, False):
                    self.logger.info(f"  Data type '{data_type}' already injected, skipping")
                    continue
                
                # 注入数据
                inject_success = self._inject_data_type(data_type)
                if inject_success:
                    self._injected_data[data_type] = True
                    self.logger.info(f"  ✓ Injected {data_type}")
                else:
                    self.logger.warning(f"  ✗ Failed to inject {data_type}")
                    success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"Data injection failed for {package_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self._cleanup()
    
    def _ensure_connected(self) -> bool:
        """确保设备已连接"""
        if self.env is not None:
            return True
        
        if not ANDROID_WORLD_AVAILABLE:
            self.logger.warning("android_world not available, using ADB fallback")
            return self._check_adb_connection()
        
        try:
            adb_path = self._find_adb_path()
            self.env = env_launcher.load_and_setup_env(
                console_port=self.console_port,
                emulator_setup=False,
                freeze_datetime=False,
                adb_path=adb_path,
                grpc_port=self.grpc_port
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
    
    def _check_adb_connection(self) -> bool:
        """检查 ADB 连接"""
        try:
            result = subprocess.run(
                ['adb', '-s', self.device_serial, 'shell', 'echo', 'ok'],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
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
    
    def _inject_data_type(self, data_type: str) -> bool:
        """注入指定类型的数据"""
        inject_methods = {
            'contacts': self._inject_contacts,
            'sms': self._inject_sms,
            'call_history': self._inject_call_history,
            'gallery': self._inject_gallery,
            'files': self._inject_files,
            'recipes': self._inject_recipes,
            'tasks': self._inject_tasks,
            'joplin': self._inject_joplin,
            'activities': self._inject_activities,
            'calendar': self._inject_calendar,
            'markor': self._inject_markor,
            'expenses': self._inject_expenses,
            'music': self._inject_music,
            'videos': self._inject_videos,
        }
        
        method = inject_methods.get(data_type)
        if method is None:
            self.logger.warning(f"Unknown data type: {data_type}")
            return False
        
        try:
            return method()
        except Exception as e:
            self.logger.error(f"Error injecting {data_type}: {e}")
            return False
    
    # ========================================================================
    # 数据注入方法
    # ========================================================================
    
    def _inject_contacts(self) -> bool:
        """注入联系人数据"""
        contacts = get_all_contacts()
        self.logger.info(f"Injecting {len(contacts)} contacts...")
        
        if ANDROID_WORLD_AVAILABLE and self.env:
            try:
                # 清理现有联系人
                contacts_utils.clear_contacts(self.env.controller)
                time.sleep(1)
            except Exception as e:
                self.logger.debug(f"Could not clear contacts: {e}")
        
        success_count = 0
        for contact in contacts:
            if self._add_contact_via_adb(contact['name'], contact['phone']):
                success_count += 1
        
        self.logger.info(f"Injected {success_count}/{len(contacts)} contacts")
        return success_count > len(contacts) * 0.5
    
    def _add_contact_via_adb(self, name: str, phone: str) -> bool:
        """通过 ADB 添加联系人"""
        try:
            # 解析姓名
            name_parts = name.split(' ', 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ''
            
            # 插入 raw_contact
            self._adb_shell(
                "content insert --uri content://com.android.contacts/raw_contacts "
                "--bind account_type:s: --bind account_name:s:"
            )
            
            # 获取 raw_contact_id
            output = self._adb_shell(
                "content query --uri content://com.android.contacts/raw_contacts "
                "--projection _id --sort '_id DESC' --where 'deleted=0'"
            )
            
            match = re.search(r'_id=(\d+)', output)
            if not match:
                return False
            
            raw_contact_id = match.group(1)
            
            # 插入姓名
            self._adb_shell(
                f"content insert --uri content://com.android.contacts/data "
                f"--bind raw_contact_id:i:{raw_contact_id} "
                f"--bind mimetype:s:vnd.android.cursor.item/name "
                f"--bind 'data1:s:{name}' "
                f"--bind 'data2:s:{first_name}' "
                f"--bind 'data3:s:{last_name}'"
            )
            
            # 插入电话号码
            phone_clean = phone.replace('-', '').replace('+', '').replace(' ', '')
            self._adb_shell(
                f"content insert --uri content://com.android.contacts/data "
                f"--bind raw_contact_id:i:{raw_contact_id} "
                f"--bind mimetype:s:vnd.android.cursor.item/phone_v2 "
                f"--bind 'data1:s:{phone}' "
                f"--bind data2:i:2"
            )
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Failed to add contact {name}: {e}")
            return False
    
    def _inject_sms(self) -> bool:
        """注入短信数据
        
        使用模拟器内置的短信模拟功能（adb emu sms send）注入确定性短信。
        
        关键步骤（模拟 SimpleSMSMessengerApp.setup 的完整流程）：
        1. 将 Simple SMS Messenger 设置为默认 SMS 应用（命令行方式）
        2. 启动应用并通过 UI 点击确认设置为默认（避免弹出选择对话框）
        3. 使用 text_emulator 模拟接收短信
        
        参考: 
        - reference/MobileForge Emulator Setup/android_world/comprehensive_setup/app_data_injector.py
        - reference/MobileForge Rollout/framework/models/AndroidWorld/android_world/env/setup_device/apps.py
          (SimpleSMSMessengerApp.setup 方法中设置默认 SMS 应用并通过 UI 确认)
        
        注意：text_emulator 模拟的是接收短信，因此只注入 type=1 的消息。
        """
        messages = get_deterministic_sms_conversations()
        self.logger.info(f"Injecting {len(messages)} SMS messages via emulator...")
        
        # 只注入收到的消息（type=1），因为 text_emulator 模拟的是接收短信
        incoming_messages = [msg for msg in messages if msg['type'] == 1]
        self.logger.info(f"Filtering to {len(incoming_messages)} incoming messages for injection...")
        
        sms_package = "com.simplemobiletools.smsmessenger"
        sms_activity = "com.simplemobiletools.smsmessenger.activities.MainActivity"
        
        # 步骤1：设置默认 SMS 应用（命令行方式）
        try:
            self._adb_shell(f'settings put secure sms_default_application {sms_package}')
            self.logger.info(f"已设置 {sms_package} 为默认 SMS 应用（命令行）")
        except Exception as e:
            self.logger.warning(f"设置默认 SMS 应用失败: {e}")
        
        # 步骤2：启动应用并通过 UI 确认设置为默认
        # 这是 SimpleSMSMessengerApp.setup() 中的关键步骤
        # 仅通过命令行设置不够，需要在 UI 上确认
        try:
            self.logger.info("启动 SMS Messenger 应用并确认设置为默认...")
            
            # 启动应用
            self._adb_shell(f'am start -n {sms_package}/{sms_activity}')
            time.sleep(3)
            
            # 尝试通过 UI 自动化点击确认设置默认应用
            # 方法1：使用 uiautomator 按文本查找并点击元素
            try:
                # 点击 "SMS Messenger" 选项
                self._click_ui_element_by_text("SMS Messenger")
                time.sleep(1)
                # 点击 "Set as default" 按钮
                self._click_ui_element_by_text("Set as default")
                time.sleep(2)
            except Exception as ui_err:
                self.logger.debug(f"UI 自动化点击失败，尝试坐标点击: {ui_err}")
                # 方法2：回退到坐标点击（基于截图中的位置）
                try:
                    self._adb_shell('input tap 350 877')  # 点击 SMS Messenger 选项
                    time.sleep(1)
                    self._adb_shell('input tap 540 958')  # 点击 Set as default 按钮
                    time.sleep(2)
                except Exception:
                    pass
            
            # 关闭应用
            self._adb_shell(f'am force-stop {sms_package}')
            self.logger.info("SMS Messenger 默认应用设置完成")
            
        except Exception as e:
            self.logger.warning(f"UI 确认设置默认应用时出现警告: {e}")
        
        # 步骤3：注入短信
        if ANDROID_WORLD_AVAILABLE and self.env:
            success_count = 0
            for msg in incoming_messages:
                try:
                    # 清理电话号码（移除非数字字符，保留+号）
                    phone = re.sub(r"[^0-9+]", "", msg['address'])
                    message_text = msg['body']
                    
                    # 使用模拟器内置功能发送短信
                    adb_utils.text_emulator(
                        self.env.controller.env,
                        phone,
                        message_text
                    )
                    success_count += 1
                    # 短暂延迟避免消息顺序混乱
                    time.sleep(0.5)
                except Exception as e:
                    self.logger.debug(f"Failed to inject SMS: {e}")
            
            self.logger.info(f"Injected {success_count}/{len(incoming_messages)} SMS messages")
            return success_count > 0
        else:
            # 回退方式：使用 ADB 直接发送
            success_count = 0
            for msg in incoming_messages:
                try:
                    phone = re.sub(r"[^0-9+]", "", msg['address'])
                    message_text = msg['body']
                    
                    # 使用 adb emu sms send 命令
                    result = subprocess.run(
                        ['adb', '-s', self.device_serial, 'emu', 'sms', 'send', phone, message_text],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        success_count += 1
                    time.sleep(0.5)
                except Exception as e:
                    self.logger.debug(f"Failed to inject SMS via ADB: {e}")
            
            self.logger.info(f"Injected {success_count}/{len(incoming_messages)} SMS messages (ADB fallback)")
            return success_count > 0
    
    def _inject_call_history(self) -> bool:
        """注入通话记录"""
        calls = get_deterministic_call_history()
        self.logger.info(f"Injecting {len(calls)} call records...")
        
        success_count = 0
        for call in calls:
            try:
                self._adb_shell(
                    f"content insert --uri content://call_log/calls "
                    f"--bind number:s:{call['number']} "
                    f"--bind type:i:{call['type']} "
                    f"--bind duration:i:{call['duration']} "
                    f"--bind date:l:{call['date']}"
                )
                success_count += 1
            except Exception as e:
                self.logger.debug(f"Failed to insert call: {e}")
        
        self.logger.info(f"Injected {success_count}/{len(calls)} call records")
        return success_count > 0
    
    def _inject_gallery(self) -> bool:
        """注入相册图片"""
        images = get_deterministic_gallery_images()
        self.logger.info(f"Injecting {len(images)} gallery images...")
        
        if not ANDROID_WORLD_AVAILABLE or not self.env:
            self.logger.warning("Cannot inject gallery: android_world not available")
            return False
        
        success_count = 0
        for img in images:
            try:
                # 确保目录存在
                self._adb_shell(f"mkdir -p {img['directory']}")
                
                # 使用 user_data_generation 创建图片
                user_data_generation.write_to_gallery(
                    img['text'],
                    img['filename'],
                    self.env
                )
                success_count += 1
            except Exception as e:
                self.logger.debug(f"Failed to create image {img['filename']}: {e}")
        
        self.logger.info(f"Injected {success_count}/{len(images)} images")
        return success_count > len(images) * 0.5
    
    def _inject_files(self) -> bool:
        """注入文件系统数据"""
        file_structure = get_deterministic_file_structure()
        file_contents = get_deterministic_file_contents()
        
        self.logger.info(f"Injecting file system structure...")
        
        # 创建目录结构
        for directory in file_structure.keys():
            self._adb_shell(f"mkdir -p /storage/emulated/0/{directory}")
        
        # 创建文件
        success_count = 0
        for file_info in file_contents:
            try:
                path = file_info['path']
                content = file_info['content'].replace("'", "\\'").replace('\n', '\\n')
                
                # 使用 echo 创建文件
                self._adb_shell(f"echo -e '{content}' > {path}")
                success_count += 1
            except Exception as e:
                self.logger.debug(f"Failed to create file: {e}")
        
        self.logger.info(f"Injected {success_count}/{len(file_contents)} files")
        return success_count > 0
    
    def _inject_recipes(self) -> bool:
        """注入食谱数据"""
        if not ANDROID_WORLD_AVAILABLE or not self.env:
            self.logger.warning("Cannot inject recipes: android_world not available")
            return False
        
        recipes = get_all_recipes()
        if not recipes:
            self.logger.warning("No recipes available")
            return False
        
        self.logger.info(f"Injecting {len(recipes)} recipes...")
        
        try:
            sqlite_utils.insert_rows_to_remote_db(
                rows=recipes,
                exclude_key='recipeId',
                table_name='recipes',
                remote_db_file_path='/data/data/com.flauschcode.broccoli/databases/broccoli',
                app_name='broccoli app',
                env=self.env,
                timeout_sec=60.0
            )
            self.logger.info(f"Injected {len(recipes)} recipes")
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject recipes: {e}")
            return False
    
    def _inject_tasks(self) -> bool:
        """注入任务数据"""
        if not ANDROID_WORLD_AVAILABLE or not self.env:
            self.logger.warning("Cannot inject tasks: android_world not available")
            return False
        
        tasks = create_deterministic_tasks()
        if not tasks:
            self.logger.warning("No tasks available")
            return False
        
        self.logger.info(f"Injecting {len(tasks)} tasks...")
        
        try:
            # 清理现有任务
            task_app_utils.clear_task_db(self.env)
            
            # 添加任务
            task_app_utils.add_tasks(tasks, self.env)
            self.logger.info(f"Injected {len(tasks)} tasks")
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject tasks: {e}")
            return False
    
    def _inject_joplin(self) -> bool:
        """注入 Joplin 笔记数据"""
        if not ANDROID_WORLD_AVAILABLE or not self.env:
            self.logger.warning("Cannot inject Joplin notes: android_world not available")
            return False
        
        all_notes = get_all_joplin_notes()
        if not all_notes:
            self.logger.warning("No Joplin notes available")
            return False
        
        total_notes = sum(len(notes) for notes in all_notes.values())
        self.logger.info(f"Injecting {total_notes} Joplin notes across {len(all_notes)} folders...")
        
        try:
            # 清理现有数据
            joplin_app_utils.clear_dbs(self.env)
            
            # 创建笔记
            notes = []
            folder_mapping = {}
            
            for folder_name in sorted(all_notes.keys()):
                folder_notes = all_notes[folder_name]
                for note_data in folder_notes:
                    note = joplin_app_utils.create_note(
                        folder=folder_name,
                        title=note_data['title'],
                        body=note_data['body'],
                        folder_mapping=folder_mapping,
                        env=self.env
                    )
                    notes.append(note)
            
            # 添加笔记
            joplin_app_utils.add_notes(notes, self.env)
            self.logger.info(f"Injected {len(notes)} Joplin notes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to inject Joplin notes: {e}")
            return False
    
    def _inject_activities(self) -> bool:
        """注入运动活动数据"""
        if not ANDROID_WORLD_AVAILABLE or not self.env:
            self.logger.warning("Cannot inject activities: android_world not available")
            return False
        
        activities = create_deterministic_activities()
        if not activities:
            self.logger.warning("No activities available")
            return False
        
        self.logger.info(f"Injecting {len(activities)} activities...")
        
        try:
            # 清理现有数据
            activity_app_utils.clear_db(self.env)
            
            # 添加活动
            activity_app_utils._add_activities(activities, self.env)
            self.logger.info(f"Injected {len(activities)} activities")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to inject activities: {e}")
            return False
    
    def _inject_calendar(self) -> bool:
        """注入日历事件数据"""
        if not ANDROID_WORLD_AVAILABLE or not self.env:
            self.logger.warning("Cannot inject calendar: android_world not available")
            return False
        
        events = create_deterministic_calendar_events()
        if not events:
            self.logger.warning("No calendar events available")
            return False
        
        self.logger.info(f"Injecting {len(events)} calendar events...")
        
        try:
            # 清理现有数据
            calendar_utils.clear_calendar_db(self.env)
            
            # 添加事件
            calendar_utils.add_events(events, self.env)
            self.logger.info(f"Injected {len(events)} calendar events")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to inject calendar events: {e}")
            return False
    
    def _inject_markor(self) -> bool:
        """注入 Markor 文档数据"""
        documents = get_all_markor_documents()
        self.logger.info(f"Injecting {len(documents)} Markor documents...")
        
        # 创建 Markor 目录
        self._adb_shell("mkdir -p /storage/emulated/0/Documents/Markor")
        self._adb_shell("chmod 777 /storage/emulated/0/Documents/Markor")
        
        success_count = 0
        for doc in documents:
            try:
                filename = doc['filename']
                content = doc['content']
                
                if ANDROID_WORLD_AVAILABLE and self.env:
                    # 使用 android_world 的方法
                    user_data_generation.write_to_markor(
                        content,
                        filename,
                        self.env
                    )
                else:
                    # ADB 回退方式
                    escaped_content = content.replace("'", "\\'").replace('\n', '\\n')
                    self._adb_shell(
                        f"echo -e '{escaped_content}' > /storage/emulated/0/Documents/Markor/{filename}"
                    )
                
                success_count += 1
                
            except Exception as e:
                self.logger.debug(f"Failed to create Markor doc {doc.get('filename')}: {e}")
        
        self.logger.info(f"Injected {success_count}/{len(documents)} Markor documents")
        return success_count > len(documents) * 0.5
    
    def _complete_expense_app_initialization(self) -> bool:
        """
        完成 Pro Expense (com.arduia.expense) 应用的初始化流程。
        
        参考: reference/MobileForge Rollout/framework/models/AndroidWorld/android_world/env/setup_device/apps.py
        ExpenseApp.setup() 方法 (473-491行)
        
        初始化步骤：
        1. 清空应用数据
        2. 授予权限
        3. 启动应用
        4. 点击 "NEXT" 按钮（选择语言后）
        5. 点击 "CONTINUE" 按钮
        6. 关闭应用
        
        Returns:
            bool: 是否成功完成初始化
        """
        package_name = 'com.arduia.expense'
        app_name = 'pro expense'
        
        self.logger.info(f"开始完整初始化 {app_name} 应用...")
        
        try:
            # 1. 清空应用数据（参考 apps.py 中的 super().setup(env)）
            self.logger.info(f"清空 {app_name} 应用数据...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                try:
                    adb_utils.clear_app_data(package_name, self.env.controller)
                except Exception as e:
                    self.logger.debug(f"使用 adb_utils 清空数据失败，尝试 pm clear: {e}")
                    self._adb_shell(f'pm clear {package_name}')
            else:
                self._adb_shell(f'pm clear {package_name}')
            
            time.sleep(1)
            
            # 2. 授予权限
            permissions = [
                'android.permission.READ_EXTERNAL_STORAGE',
                'android.permission.WRITE_EXTERNAL_STORAGE'
            ]
            for permission in permissions:
                try:
                    self._adb_shell(f'pm grant {package_name} {permission}')
                except Exception as e:
                    self.logger.debug(f"授予权限 {permission} 时出现警告: {e}")
            
            # 3. 启动应用
            self.logger.info(f"启动 {app_name} 应用...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                adb_utils.launch_app(app_name, self.env.controller)
            else:
                self._adb_shell(f'am start -n {package_name}/com.arduia.expense.ui.MainActivity')
            
            time.sleep(2.0)
            
            # 4. 点击 "NEXT" 按钮（参考 apps.py: controller.click_element("NEXT")）
            self.logger.info("点击 NEXT 按钮...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                try:
                    controller = tools.AndroidToolController(env=self.env.controller)
                    controller.click_element("NEXT")
                except Exception as e:
                    self.logger.debug(f"使用 AndroidToolController 点击失败，尝试回退方法: {e}")
                    self._click_ui_element_by_text("NEXT")
            else:
                self._click_ui_element_by_text("NEXT")
            
            time.sleep(2.0)
            
            # 5. 点击 "CONTINUE" 按钮（参考 apps.py: controller.click_element("CONTINUE")）
            self.logger.info("点击 CONTINUE 按钮...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                try:
                    controller = tools.AndroidToolController(env=self.env.controller)
                    controller.click_element("CONTINUE")
                except Exception as e:
                    self.logger.debug(f"使用 AndroidToolController 点击失败，尝试回退方法: {e}")
                    self._click_ui_element_by_text("CONTINUE")
            else:
                self._click_ui_element_by_text("CONTINUE")
            
            time.sleep(3.0)
            
            # 6. 关闭应用
            self.logger.info(f"关闭 {app_name} 应用...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                adb_utils.close_app(app_name, self.env.controller)
            else:
                self._adb_shell(f'am force-stop {package_name}')
            
            self.logger.info(f"{app_name} 应用初始化完成")
            return True
            
        except Exception as e:
            self.logger.warning(f"{app_name} 应用初始化失败: {e}")
            # 确保应用被关闭
            try:
                self._adb_shell(f'am force-stop {package_name}')
            except:
                pass
            return False
    
    def _inject_expenses(self) -> bool:
        """注入费用记录数据
        
        参考: reference/MobileForge Rollout/framework/native_app_injector.py
        和 reference/MobileForge Rollout/framework/models/AndroidWorld/android_world/env/setup_device/apps.py
        
        关键步骤：
        1. 完成应用初始化（点击引导页按钮，创建数据库）
        2. 清理现有数据
        3. 注入新数据
        """
        if not ANDROID_WORLD_AVAILABLE or not self.env:
            self.logger.warning("Cannot inject expenses: android_world not available")
            return False
        
        # 1. 完成应用初始化（包括点击引导页按钮）
        # 这是关键步骤，确保应用完全初始化，数据库被创建
        self._complete_expense_app_initialization()
        
        # 额外等待以确保数据库初始化完成
        time.sleep(2)
        
        # 2. 清理现有 expense 数据，确保数据一致性
        try:
            self._adb_shell(
                "sqlite3 /data/data/com.arduia.expense/databases/accounting.db 'DELETE FROM expense;'"
            )
            self.logger.info("已清理现有 expense 数据")
        except Exception as e:
            self.logger.warning(f"清理 expense 数据时出现警告: {e}")
        
        expenses = create_deterministic_expenses()
        
        if not expenses:
            self.logger.warning("No expenses available")
            return False
        
        self.logger.info(f"Injecting {len(expenses)} expense records...")
        
        try:
            sqlite_utils.insert_rows_to_remote_db(
                rows=expenses,
                exclude_key='expense_id',
                table_name='expense',
                remote_db_file_path='/data/data/com.arduia.expense/databases/accounting.db',
                app_name='pro expense',
                env=self.env,
                timeout_sec=60.0
            )
            self.logger.info(f"Injected {len(expenses)} expense records")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to inject expenses: {e}")
            return False
    
    def _inject_music(self) -> bool:
        """注入音乐文件数据"""
        if not ANDROID_WORLD_AVAILABLE or not self.env:
            self.logger.warning("Cannot inject music: android_world not available")
            return False
        
        music_files = get_deterministic_music_files()
        self.logger.info(f"Injecting {len(music_files)} music files...")
        
        # 创建音乐目录
        self._adb_shell("mkdir -p /storage/emulated/0/Music")
        
        success_count = 0
        for music in music_files:
            try:
                user_data_generation.write_mp3_file_to_device(
                    f"{music['directory']}/{music['filename']}",
                    self.env,
                    title=music['title'],
                    artist=music['artist'],
                    duration_milliseconds=music['duration_ms'],
                )
                success_count += 1
            except Exception as e:
                self.logger.debug(f"Failed to create music file {music['filename']}: {e}")
        
        self.logger.info(f"Injected {success_count}/{len(music_files)} music files")
        return success_count > 0
    
    def _complete_vlc_app_initialization(self) -> bool:
        """
        完成 VLC (org.videolan.vlc) 应用的初始化流程。
        
        参考: reference/MobileForge Rollout/framework/models/AndroidWorld/android_world/env/setup_device/apps.py
        VlcApp.setup() 方法 (635-683行)
        
        初始化步骤：
        1. 清空应用数据
        2. 授予 POST_NOTIFICATIONS 权限
        3. 创建 /storage/emulated/0/VLCVideos 目录
        4. 使用 monkey 命令启动应用（模拟从启动器打开，以触发数据库创建）
        5. 点击 "Skip" 按钮
        6. 点击 "GRANT PERMISSION" 按钮
        7. 点击 "OK" 按钮
        8. 点击 "Allow access to manage all files" 按钮
        9. 关闭应用
        
        Returns:
            bool: 是否成功完成初始化
        """
        package_name = 'org.videolan.vlc'
        app_name = 'vlc'
        videos_path = '/storage/emulated/0/VLCVideos'
        
        self.logger.info(f"开始完整初始化 {app_name} 应用...")
        
        try:
            # 1. 清空应用数据（参考 apps.py 中的 super().setup(env)）
            self.logger.info(f"清空 {app_name} 应用数据...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                try:
                    adb_utils.clear_app_data(package_name, self.env.controller)
                except Exception as e:
                    self.logger.debug(f"使用 adb_utils 清空数据失败，尝试 pm clear: {e}")
                    self._adb_shell(f'pm clear {package_name}')
            else:
                self._adb_shell(f'pm clear {package_name}')
            
            time.sleep(1)
            
            # 2. 授予权限（参考 apps.py: adb_utils.grant_permissions()）
            permissions = [
                'android.permission.POST_NOTIFICATIONS',
                'android.permission.READ_EXTERNAL_STORAGE',
                'android.permission.WRITE_EXTERNAL_STORAGE',
                'android.permission.RECORD_AUDIO'
            ]
            for permission in permissions:
                try:
                    if ANDROID_WORLD_AVAILABLE and self.env:
                        adb_utils.grant_permissions(package_name, permission, self.env.controller)
                    else:
                        self._adb_shell(f'pm grant {package_name} {permission}')
                except Exception as e:
                    self.logger.debug(f"授予权限 {permission} 时出现警告: {e}")
            
            # 3. 创建视频目录（参考 apps.py: file_utils.mkdir(cls.videos_path, env.controller)）
            self.logger.info(f"创建视频目录 {videos_path}...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                try:
                    if not file_utils.check_directory_exists(videos_path, self.env.controller):
                        file_utils.mkdir(videos_path, self.env.controller)
                except Exception as e:
                    self.logger.debug(f"使用 file_utils 创建目录失败，尝试 mkdir: {e}")
                    self._adb_shell(f'mkdir -p {videos_path}')
            else:
                self._adb_shell(f'mkdir -p {videos_path}')
            
            time.sleep(2.0)
            
            # 4. 使用 monkey 命令启动应用（参考 apps.py: 模拟从启动器打开以触发数据库创建）
            # 这是关键步骤，使用 monkey 命令而不是 am start 可以触发完整的初始化流程
            self.logger.info(f"使用 monkey 命令启动 {app_name} 应用...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                adb_utils.issue_generic_request(
                    [
                        "shell",
                        "monkey",
                        "-p",
                        package_name,
                        "-c", "android.intent.category.LAUNCHER",
                        "1",
                    ],
                    self.env.controller,
                )
            else:
                self._adb_shell(f'monkey -p {package_name} -c android.intent.category.LAUNCHER 1')
            
            time.sleep(2.0)
            
            # 5. 点击 "Skip" 按钮（参考 apps.py: controller.click_element("Skip")）
            self.logger.info("点击 Skip 按钮...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                try:
                    controller = tools.AndroidToolController(env=self.env.controller)
                    controller.click_element("Skip")
                except Exception as e:
                    self.logger.debug(f"使用 AndroidToolController 点击失败，尝试回退方法: {e}")
                    # 尝试点击 "SKIP" (大写) 或 "Skip" (混合大小写)
                    if not self._click_ui_element_by_text("Skip"):
                        self._click_ui_element_by_text("SKIP")
            else:
                if not self._click_ui_element_by_text("Skip"):
                    self._click_ui_element_by_text("SKIP")
            
            time.sleep(2.0)
            
            # 6. 点击 "GRANT PERMISSION" 按钮
            self.logger.info("点击 GRANT PERMISSION 按钮...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                try:
                    controller = tools.AndroidToolController(env=self.env.controller)
                    controller.click_element("GRANT PERMISSION")
                except Exception as e:
                    self.logger.debug(f"使用 AndroidToolController 点击失败，尝试回退方法: {e}")
                    self._click_ui_element_by_text("GRANT PERMISSION")
            else:
                self._click_ui_element_by_text("GRANT PERMISSION")
            
            time.sleep(2.0)
            
            # 7. 点击 "OK" 按钮
            self.logger.info("点击 OK 按钮...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                try:
                    controller = tools.AndroidToolController(env=self.env.controller)
                    controller.click_element("OK")
                except Exception as e:
                    self.logger.debug(f"使用 AndroidToolController 点击失败，尝试回退方法: {e}")
                    self._click_ui_element_by_text("OK")
            else:
                self._click_ui_element_by_text("OK")
            
            time.sleep(2.0)
            
            # 8. 点击 "Allow access to manage all files" 按钮
            self.logger.info("点击 Allow access to manage all files 按钮...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                try:
                    controller = tools.AndroidToolController(env=self.env.controller)
                    controller.click_element("Allow access to manage all files")
                except Exception as e:
                    self.logger.debug(f"使用 AndroidToolController 点击失败，尝试回退方法: {e}")
                    # 尝试多种可能的文本匹配
                    if not self._click_ui_element_by_text("Allow access to manage all files"):
                        self._click_ui_element_by_text("Allow")
            else:
                if not self._click_ui_element_by_text("Allow access to manage all files"):
                    self._click_ui_element_by_text("Allow")
            
            time.sleep(2.0)
            
            # 9. 关闭应用
            self.logger.info(f"关闭 {app_name} 应用...")
            if ANDROID_WORLD_AVAILABLE and self.env:
                adb_utils.close_app(app_name, self.env.controller)
            else:
                self._adb_shell(f'am force-stop {package_name}')
            
            self.logger.info(f"{app_name} 应用初始化完成")
            return True
            
        except Exception as e:
            self.logger.warning(f"{app_name} 应用初始化失败: {e}")
            # 确保应用被关闭
            try:
                self._adb_shell(f'am force-stop {package_name}')
            except:
                pass
            return False
    
    def _inject_videos(self) -> bool:
        """注入视频文件数据
        
        参考: reference/MobileForge Rollout/framework/native_app_injector.py
        和 reference/MobileForge Rollout/framework/models/AndroidWorld/android_world/env/setup_device/apps.py
        
        关键步骤：
        1. 完成 VLC 应用初始化（点击引导页按钮，创建数据库）
        2. 创建视频目录
        3. 清理现有视频文件
        4. 注入新视频文件
        """
        if not ANDROID_WORLD_AVAILABLE or not self.env:
            self.logger.warning("Cannot inject videos: android_world not available")
            return False
        
        # 1. 完成 VLC 应用初始化（包括点击引导页按钮）
        # 这是关键步骤，确保应用完全初始化，数据库被创建
        self._complete_vlc_app_initialization()
        
        # 额外等待以确保初始化完成
        time.sleep(2)
        
        videos = get_deterministic_video_files()
        self.logger.info(f"Injecting {len(videos)} video files...")
        
        # 2. 创建视频目录（初始化时已创建，这里确保存在）
        vlc_video_dir = "/storage/emulated/0/VLCVideos"
        self._adb_shell(f"mkdir -p {vlc_video_dir}")
        
        # 3. 清理现有视频文件，确保数据一致性（与 reference 完全一致）
        try:
            self._adb_shell(f"rm -f {vlc_video_dir}/*.mp4")
            self.logger.info("已清理现有 VLC 视频文件")
        except Exception as e:
            self.logger.warning(f"清理 VLC 视频目录时出现警告: {e}")
        
        # 4. 注入新视频文件
        success_count = 0
        for video in videos:
            try:
                user_data_generation.write_video_file_to_device(
                    video['filename'],
                    video['directory'],
                    self.env,
                    messages=video['messages'],
                    fps=1,
                    message_display_time=video['duration_seconds']
                )
                success_count += 1
            except Exception as e:
                self.logger.debug(f"Failed to create video {video['filename']}: {e}")
        
        self.logger.info(f"Injected {success_count}/{len(videos)} video files")
        return success_count > 0
    
    def _adb_shell(self, command: str) -> str:
        """执行 ADB shell 命令"""
        try:
            result = subprocess.run(
                ['adb', '-s', self.device_serial, 'shell', command],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout.strip()
        except Exception as e:
            self.logger.debug(f"ADB command failed: {command[:50]}..., error: {e}")
            return ""
    
    def _click_ui_element_by_text(self, text: str) -> bool:
        """
        通过文本查找并点击 UI 元素
        
        使用 uiautomator 导出当前 UI 层次结构，查找包含指定文本的元素，
        然后点击该元素的中心位置。
        
        Args:
            text: 要查找的文本
            
        Returns:
            bool: 是否成功点击
        """
        try:
            # 导出 UI 层次结构到文件
            self._adb_shell('uiautomator dump /sdcard/ui_dump.xml')
            time.sleep(0.5)
            
            # 读取 UI dump 文件
            ui_content = self._adb_shell('cat /sdcard/ui_dump.xml')
            
            if not ui_content:
                return False
            
            # 查找包含指定文本的元素的 bounds
            # bounds 格式: [left,top][right,bottom]
            import re as re_module
            pattern = rf'text="{re_module.escape(text)}"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"'
            match = re_module.search(pattern, ui_content)
            
            if not match:
                # 尝试部分匹配
                pattern = rf'text="[^"]*{re_module.escape(text)}[^"]*"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"'
                match = re_module.search(pattern, ui_content)
            
            if match:
                left, top, right, bottom = map(int, match.groups())
                # 计算中心点
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                
                # 点击中心点
                self._adb_shell(f'input tap {center_x} {center_y}')
                self.logger.debug(f"点击元素 '{text}' 位置: ({center_x}, {center_y})")
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"点击 UI 元素 '{text}' 失败: {e}")
            return False
    
    def _ensure_app_ready(self, package_name: str, main_activity: str = None) -> bool:
        """
        确保应用已准备好接收数据。
        
        启动应用以初始化其数据库，然后关闭应用。
        参考: reference/MobileForge Rollout/framework/native_app_injector.py
        
        Args:
            package_name: 应用包名
            main_activity: 主 Activity（可选）
            
        Returns:
            bool: 是否成功
        """
        try:
            # 授予应用所需权限
            app_permissions = {
                'com.arduia.expense': [
                    'android.permission.READ_EXTERNAL_STORAGE',
                    'android.permission.WRITE_EXTERNAL_STORAGE'
                ],
                'org.videolan.vlc': [
                    'android.permission.READ_EXTERNAL_STORAGE',
                    'android.permission.WRITE_EXTERNAL_STORAGE',
                    'android.permission.RECORD_AUDIO'
                ],
                'com.simplemobiletools.smsmessenger': [
                    'android.permission.READ_SMS',
                    'android.permission.SEND_SMS',
                    'android.permission.RECEIVE_SMS',
                    'android.permission.READ_CONTACTS',
                ],
            }
            
            if package_name in app_permissions:
                for permission in app_permissions[package_name]:
                    try:
                        self._adb_shell(f'pm grant {package_name} {permission}')
                    except Exception as e:
                        self.logger.debug(f"授予权限 {permission} 时出现警告: {e}")
            
            # 启动应用以初始化数据库
            if main_activity:
                try:
                    self._adb_shell(f'am start -n {package_name}/{main_activity}')
                    time.sleep(3)
                    # 强制停止应用
                    self._adb_shell(f'am force-stop {package_name}')
                except Exception as e:
                    self.logger.debug(f"启动应用 {package_name} 时出现警告: {e}")
            
            return True
        except Exception as e:
            self.logger.warning(f"准备应用 {package_name} 时出错: {e}")
            return False
    
    def _cleanup(self) -> None:
        """清理资源"""
        if self.env:
            try:
                self.env.close()
            except Exception:
                pass
            finally:
                self.env = None


# 便捷函数
def inject_app_data(device_serial: str, 
                   package_name: str,
                   console_port: int = 5554,
                   grpc_port: int = 8554) -> bool:
    """
    便捷函数：为应用注入数据
    
    Args:
        device_serial: 设备序列号
        package_name: 应用包名
        console_port: 控制台端口
        grpc_port: gRPC端口
        
    Returns:
        bool: 是否成功
    """
    injector = AppDataInjector(device_serial, console_port, grpc_port)
    return injector.inject_app_data(package_name)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inject app data for exploration")
    parser.add_argument("-s", "--serial", default="emulator-5554", help="Device serial")
    parser.add_argument("-p", "--package", required=True, help="App package name")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    success = inject_app_data(args.serial, args.package)
    sys.exit(0 if success else 1)

