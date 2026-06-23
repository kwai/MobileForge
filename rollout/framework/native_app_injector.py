"""
App Data Injector - 使用 AndroidWorld 原生工具的数据注入器

该模块完全采用 AndroidWorld 框架的原生数据注入工具，确保数据格式与 schema 完全匹配。

参考: reference/MobileForge Emulator Setup/android_world/comprehensive_setup/app_data_injector.py
"""

import logging
import os
import re
import subprocess
import sys
import time
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
from android_world.env import interface
from android_world.env import adb_utils
from android_world.env import device_constants
from android_env.proto import adb_pb2
from android_world.task_evals.utils import sqlite_utils
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import user_data_generation
from android_world.task_evals.information_retrieval import joplin_app_utils
from android_world.task_evals.information_retrieval import task_app_utils
from android_world.task_evals.information_retrieval import activity_app_utils
from android_world.task_evals.single.calendar import calendar_utils
from android_world.task_evals.common_validators import sms_validators
from android_world.env.setup_device import apps as setup_apps
from android_world.task_evals.single import recipe as recipe_module
from android_world.task_evals.single import expense as expense_module
from android_world.utils import contacts_utils
from android_world.utils import file_utils

# 导入本地确定性数据模块
from . import deterministic_data

logger = logging.getLogger(__name__)


class AppDataInjector:
    """
    使用 AndroidWorld 原生工具的数据注入器。
    
    参考 reference/MobileForge Emulator Setup/android_world/comprehensive_setup/app_data_injector.py
    """
    
    # 应用权限映射
    APP_PERMISSIONS = {
        'net.gsantner.markor': [
            'android.permission.READ_EXTERNAL_STORAGE',
            'android.permission.WRITE_EXTERNAL_STORAGE',
            'android.permission.MANAGE_EXTERNAL_STORAGE'
        ],
        'com.simplemobiletools.calendar.pro': [
            'android.permission.READ_CALENDAR',
            'android.permission.WRITE_CALENDAR',
            'android.permission.POST_NOTIFICATIONS'
        ],
        'net.cozic.joplin': [
            'android.permission.READ_EXTERNAL_STORAGE',
            'android.permission.WRITE_EXTERNAL_STORAGE'
        ],
        'org.tasks': [
            'android.permission.READ_CALENDAR',
            'android.permission.WRITE_CALENDAR'
        ],
        'com.flauschcode.broccoli': [
            'android.permission.READ_EXTERNAL_STORAGE',
            'android.permission.WRITE_EXTERNAL_STORAGE'
        ],
        'com.arduia.expense': [
            'android.permission.READ_EXTERNAL_STORAGE',
            'android.permission.WRITE_EXTERNAL_STORAGE'
        ],
        'code.name.monkey.retromusic': [
            'android.permission.READ_EXTERNAL_STORAGE',
            'android.permission.WRITE_EXTERNAL_STORAGE'
        ],
        'de.dennisguse.opentracks': [
            'android.permission.ACCESS_FINE_LOCATION',
            'android.permission.ACCESS_COARSE_LOCATION'
        ],
        'org.videolan.vlc': [
            'android.permission.READ_EXTERNAL_STORAGE',
            'android.permission.WRITE_EXTERNAL_STORAGE',
            'android.permission.RECORD_AUDIO'
        ],
        # SMS 应用权限 - 与 reference/MobileForge Explore 完全一致
        'com.simplemobiletools.smsmessenger': [
            'android.permission.READ_SMS',
            'android.permission.SEND_SMS',
            'android.permission.RECEIVE_SMS',
            'android.permission.READ_CONTACTS',
        ]
    }
    
    def __init__(self, env: interface.AsyncEnv, config: Optional[Dict[str, Any]] = None):
        """
        初始化注入器。
        
        Args:
            env: AndroidWorld 环境对象
            config: 可选配置
        """
        self.env = env
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 从 env 中提取 device_serial - 使用多种方法确保可靠性
        try:
            simulator = env.controller.env._coordinator._simulator
            # 优先使用 adb_device_name() 方法（最可靠）
            if hasattr(simulator, 'adb_device_name'):
                self.device_serial = simulator.adb_device_name()
                self.logger.info(f"device_serial 获取方式: adb_device_name() = {self.device_serial}")
            else:
                # 回退到配置中的 emulator_console_port
                console_port = simulator._config.emulator_launcher.emulator_console_port
                self.device_serial = f"emulator-{console_port}"
                self.logger.info(f"device_serial 获取方式: emulator_console_port = {self.device_serial}")
        except Exception as e:
            self.logger.warning(f"获取 device_serial 失败: {e}，使用默认值 emulator-5554")
            self.device_serial = "emulator-5554"
    
    def _grant_app_permissions(self, package_name: str) -> bool:
        """为指定应用授予所需权限。"""
        if package_name not in self.APP_PERMISSIONS:
            return True
        
        permissions = self.APP_PERMISSIONS[package_name]
        success_count = 0
        
        for permission in permissions:
            try:
                adb_utils.issue_generic_request(
                    ['shell', 'pm', 'grant', package_name, permission],
                    self.env.controller
                )
                success_count += 1
            except Exception as e:
                self.logger.debug(f"授予权限 {permission} 给 {package_name} 时出现警告: {e}")
        
        return success_count > 0
    
    def _decode_adb_output(self, output) -> str:
        """解码 ADB 输出。"""
        if isinstance(output, bytes):
            return output.decode('utf-8')
        return output if output else ''
    
    def _adb_shell(self, command: str) -> str:
        """
        执行 ADB shell 命令。
        
        与 reference/MobileForge Explore/parallel_exploration/app_data_injector.py 完全一致。
        使用 subprocess.run 直接调用 ADB，而不是通过 gRPC 接口。
        
        Args:
            command: 要执行的 shell 命令
            
        Returns:
            命令输出字符串
        """
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
        通过文本查找并点击 UI 元素。
        
        与 reference/MobileForge Explore/parallel_exploration/app_data_injector.py 完全一致。
        使用 uiautomator dump 导出当前 UI 层次结构，查找包含指定文本的元素，
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
            pattern = rf'text="{re.escape(text)}"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"'
            match = re.search(pattern, ui_content)
            
            if not match:
                # 尝试部分匹配
                pattern = rf'text="[^"]*{re.escape(text)}[^"]*"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"'
                match = re.search(pattern, ui_content)
            
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
        """确保应用已准备好接收数据。"""
        try:
            self._grant_app_permissions(package_name)
            
            if main_activity:
                try:
                    adb_utils.send_android_intent(
                        'start',
                        'android.intent.action.MAIN',
                        self.env.controller,
                        component=f'{package_name}/{main_activity}'
                    )
                    import time
                    time.sleep(3)
                    adb_utils.issue_generic_request(
                        ['shell', 'am', 'force-stop', package_name],
                        self.env.controller
                    )
                except Exception as e:
                    self.logger.debug(f"启动应用 {package_name} 时出现警告: {e}")
            
            return True
        except Exception as e:
            self.logger.warning(f"准备应用 {package_name} 时出错: {e}")
            return False
    
    def inject_all_data(self) -> Dict[str, bool]:
        """
        注入所有应用的确定性数据。
        
        使用预定义的固定数据，无需随机种子，保证每次注入结果完全一致。
        
        Returns:
            每个数据类型的注入结果
        """
        self.logger.info("开始注入确定性数据（使用预定义固定数据，无随机依赖）...")
        self.logger.info("将注入 AndroidWorld 原始数据集的完整数据")
        
        results = {}
        
        # ============================================================
        # 按照参考实现的顺序调用注入函数，确保随机种子消耗顺序一致
        # 参考: reference/MobileForge Emulator Setup/android_world/comprehensive_setup/app_data_injector.py
        # ============================================================
        
        # 1. 注入联系人数据
        results['contacts'] = self._inject_contacts_data_safe()
        
        # 2. 注入短信和通话记录（消耗随机数）
        results['messaging'] = self._inject_messaging_data_safe()
        
        # 3. 注入相册数据（消耗随机数）
        results['gallery'] = self._inject_gallery_data_safe()
        
        # 4. 注入日历数据
        results['calendar'] = self._inject_calendar_data_safe()
        
        # 5. 注入文件系统数据
        results['file_system'] = self._inject_files_data_safe()
        
        # 6. 注入食谱数据 (Broccoli)
        results['recipes'] = self._inject_recipe_data_safe()
        
        # 7. 注入费用数据 (Pro Expense)
        results['expenses'] = self._inject_expense_data_safe()
        
        # 8. 注入 Joplin 笔记数据
        results['joplin'] = self._inject_joplin_data_safe()
        
        # 9. 注入任务数据 (Tasks)
        results['tasks'] = self._inject_tasks_data_safe()
        
        # 10. 注入 Retro Music 数据
        results['retro_music'] = self._inject_retro_music_data_safe()
        
        # 11. 注入 Simple Calendar Pro 数据（与第4步的系统日历不同）
        # 注：本项目将系统日历和 Simple Calendar Pro 合并处理，已在步骤4完成
        
        # 12. 注入 OpenTracks 活动数据
        results['opentracks'] = self._inject_opentracks_data_safe()
        
        # 13. 注入 VLC 视频数据
        results['vlc'] = self._inject_vlc_data_safe()
        
        # 14. 注入 Markor 文档数据
        results['markor'] = self._inject_markor_data_safe()
        
        # 15. 注入 Files 应用数据（DocumentsUI 初始化）
        # 注：文件系统数据已在步骤5创建，这里只需初始化 Files 应用
        self._initialize_files_app()
        
        self.logger.info("确定性数据注入完成！（按参考实现顺序）")
        return results
    
    def _inject_contacts_data_safe(self) -> bool:
        """安全地注入联系人数据。"""
        try:
            return self._inject_contacts_data()
        except Exception as e:
            self.logger.warning(f"注入联系人数据失败: {e}")
            return False
    
    def _inject_contacts_data(self) -> bool:
        """
        注入联系人数据（50个固定联系人）。
        
        使用 ADB content provider 直接注入，完全按照 reference 实现。
        """
        self.logger.info("注入联系人数据（50个固定联系人）...")
        
        # 清空现有联系人
        self.logger.info("清空现有联系人...")
        try:
            contacts_utils.clear_contacts(self.env.controller)
        except Exception as e:
            self.logger.warning(f"清空联系人时出现警告: {e}")
        
        # 获取确定性联系人数据
        contacts = deterministic_data.get_all_contacts()
        self.logger.info(f"将注入 {len(contacts)} 个确定性联系人")
        
        success_count = 0
        for i, contact in enumerate(contacts):
            try:
                name = contact['name']
                phone = contact['phone']
                
                # 使用 content provider 插入
                success = self._add_contact_via_content_provider(name, phone)
                if success:
                    success_count += 1
                
                # 每10个联系人记录一次进度
                if (i + 1) % 10 == 0:
                    self.logger.info(f"联系人注入进度: {i + 1}/{len(contacts)}...")
                    
            except Exception as e:
                self.logger.warning(f"添加联系人失败 ({contact.get('name', 'unknown')}): {e}")
        
        self.logger.info(f"联系人数据注入完成: {success_count}/{len(contacts)} (确定性数据)")
        
        # 记录实际注入数量
        self._injected_contacts_count = success_count
        
        return success_count > 0
    
    def _add_contact_via_content_provider(self, name: str, phone: str) -> bool:
        """
        使用 content provider 添加联系人。
        
        完全按照 reference 的实现，使用单行命令格式处理带引号的参数。
        增加延迟和重试机制以提高成功率。
        
        Args:
            name: 联系人姓名 (e.g., "John Doe")
            phone: 电话号码
            
        Returns:
            True if successful, False otherwise
        """
        import re
        import time
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 解析姓名为 first name 和 last name
                name_parts = name.split(' ', 1)
                first_name = name_parts[0]
                last_name = name_parts[1] if len(name_parts) > 1 else ''
                
                # Step 1: 获取当前最大 ID（用于验证）
                old_max_id = self._get_max_contact_id()
                
                # Step 2: 插入 raw_contact 记录
                insert_raw_cmd = [
                    'shell', 'content', 'insert',
                    '--uri', 'content://com.android.contacts/raw_contacts',
                    '--bind', 'account_type:s:',
                    '--bind', 'account_name:s:'
                ]
                adb_utils.issue_generic_request(insert_raw_cmd, self.env.controller)
                
                # Step 3: 添加延迟确保数据库提交完成
                time.sleep(0.1)
                
                # Step 4: 获取新创建的 raw_contact_id（带 WHERE 条件和重试）
                raw_contact_id = None
                for query_attempt in range(3):
                    query_cmd = [
                        'shell', 'content', 'query',
                        '--uri', 'content://com.android.contacts/raw_contacts',
                        '--projection', '_id',
                        '--sort', '"_id DESC"',
                        '--where', '"deleted=0"'
                    ]
                    query_result = adb_utils.issue_generic_request(query_cmd, self.env.controller)
                    
                    output = self._decode_adb_output(query_result.generic.output if query_result.generic else '')
                    new_id = self._parse_contact_id(output)
                    
                    # 验证新 ID 是否大于旧的最大 ID
                    if new_id and (not old_max_id or int(new_id) > int(old_max_id)):
                        raw_contact_id = new_id
                        break
                    
                    time.sleep(0.05)  # 短暂等待后重试
                
                if not raw_contact_id:
                    if attempt < max_retries - 1:
                        self.logger.debug(f"无法获取 raw_contact_id for {name}, 尝试 {attempt + 1}/{max_retries}")
                        time.sleep(0.2)
                        continue
                    self.logger.warning(f"无法获取 raw_contact_id for {name}")
                    return False
                
                # Step 5: 使用单行命令格式插入姓名（与 reference 一致，使用单引号）
                name_cmd = (
                    f"content insert --uri content://com.android.contacts/data "
                    f"--bind raw_contact_id:i:{raw_contact_id} "
                    f"--bind mimetype:s:vnd.android.cursor.item/name "
                    f"--bind 'data1:s:{name}' "
                    f"--bind 'data2:s:{first_name}' "
                    f"--bind 'data3:s:{last_name}'"
                )
                adb_utils.issue_generic_request(['shell', name_cmd], self.env.controller)
                
                # Step 6: 使用单行命令格式插入电话号码（与 reference 一致，使用单引号）
                phone_clean = phone.replace('-', '').replace('+', '').replace(' ', '')
                phone_cmd = (
                    f"content insert --uri content://com.android.contacts/data "
                    f"--bind raw_contact_id:i:{raw_contact_id} "
                    f"--bind mimetype:s:vnd.android.cursor.item/phone_v2 "
                    f"--bind 'data1:s:{phone}' "
                    f"--bind data2:i:2"
                )
                adb_utils.issue_generic_request(['shell', phone_cmd], self.env.controller)
                
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.debug(f"Contact insertion attempt {attempt + 1} failed for {name}: {e}")
                    time.sleep(0.2)
                    continue
                self.logger.debug(f"Content provider insertion failed for {name}: {e}")
                return False
        
        return False
    
    def _get_max_contact_id(self) -> str:
        """获取当前联系人表中的最大 ID。"""
        try:
            query_cmd = [
                'shell', 'content', 'query',
                '--uri', 'content://com.android.contacts/raw_contacts',
                '--projection', '_id',
                '--sort', '"_id DESC"',
                '--where', '"deleted=0"'
            ]
            query_result = adb_utils.issue_generic_request(query_cmd, self.env.controller)
            output = self._decode_adb_output(query_result.generic.output if query_result.generic else '')
            return self._parse_contact_id(output)
        except Exception:
            return ''
    
    def _parse_contact_id(self, output: str) -> str:
        """
        从 content query 输出中解析 raw_contact_id。
        
        Args:
            output: content query 命令的输出
            
        Returns:
            raw_contact_id 字符串，或空字符串如果未找到
        """
        import re
        # 输出格式: "Row: 0 _id=123" 或类似格式
        match = re.search(r'_id=(\d+)', output)
        if match:
            return match.group(1)
        return ''
    
    def _inject_recipe_data_safe(self) -> bool:
        """安全地注入食谱数据。"""
        try:
            return self._inject_recipe_data()
        except Exception as e:
            self.logger.warning(f"注入食谱数据失败: {e}")
            return False
    
    def _inject_recipe_data(self) -> bool:
        """
        注入食谱数据（39个固定食谱）。
        
        完全按照 reference 实现，确保应用已初始化并使用 AndroidWorld 原生方法。
        """
        import time
        
        self.logger.info("注入食谱数据（39个固定食谱）...")
        
        # 确保应用已初始化（这会创建数据库）
        # 增加更长的等待时间确保数据库创建完成
        self._ensure_app_ready(
            'com.flauschcode.broccoli',
            'com.flauschcode.broccoli.MainActivity'
        )
        
        # 额外等待以确保数据库初始化完成
        time.sleep(2)
        
        # 清空现有数据
        self.logger.info("清空现有 Broccoli 数据...")
        try:
            recipe_module.clear_db(self.env)
        except Exception as e:
            self.logger.warning(f"清空 Broccoli 时出现警告: {e}")
            # 如果清空失败，尝试重新启动应用创建数据库
            self._ensure_app_ready(
                'com.flauschcode.broccoli',
                'com.flauschcode.broccoli.MainActivity'
            )
            time.sleep(3)
        
        # 获取确定性食谱数据（使用 sqlite_schema_utils.Recipe 类型）
        recipes = deterministic_data.get_all_recipes()
        self.logger.info(f"加载了 {len(recipes)} 个食谱")
        
        if not recipes:
            self.logger.warning("没有加载到食谱数据")
            return False
        
        # 使用 AndroidWorld 原生方法注入
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
            
            self.logger.info(f"成功注入 {len(recipes)} 个食谱")
            return True
        except Exception as e:
            self.logger.error(f"注入食谱失败: {e}")
            return False
    
    def _inject_tasks_data_safe(self) -> bool:
        """安全地注入任务数据。"""
        try:
            return self._inject_tasks_data()
        except Exception as e:
            self.logger.warning(f"注入任务数据失败: {e}")
            return False
    
    def _inject_tasks_data(self) -> bool:
        """
        注入任务数据（20个固定任务）。
        
        使用 deterministic_data.create_deterministic_tasks() 创建符合 AndroidWorld 格式的任务。
        增加延迟和重试机制以提高成功率。
        """
        import time
        
        self.logger.info("注入任务数据（20个固定任务）...")
        
        # 确保应用已初始化
        self._ensure_app_ready('org.tasks', 'org.tasks.activities.MainActivity')
        
        # 清空现有任务
        self.logger.info("清空现有任务...")
        try:
            task_app_utils.clear_task_db(self.env)
        except Exception as e:
            self.logger.warning(f"清空任务时出现警告: {e}")
        
        # 等待数据库完全清空并初始化
        time.sleep(1.0)
        
        # 使用 device_constants.DT 作为基准时间
        base_dt = device_constants.DT
        base_timestamp_ms = int(base_dt.timestamp() * 1000)
        
        # 使用 deterministic_data 创建符合 sqlite_schema_utils.Task 规范的任务
        tasks = deterministic_data.create_deterministic_tasks(base_timestamp_ms)
        
        self.logger.info(f"创建了 {len(tasks)} 个 AndroidWorld 格式任务")
        
        if not tasks:
            self.logger.warning("没有创建任务数据")
            return False
        
        # 统计完成状态
        completed_count = sum(1 for t in tasks if t.completed > 0)
        self.logger.info(f"任务状态: {completed_count} 已完成, {len(tasks) - completed_count} 待处理")
        
        # 使用 AndroidWorld 原生方法注入（带重试）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                task_app_utils.add_tasks(tasks, self.env)
                self.logger.info(f"成功注入 {len(tasks)} 个任务")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"原生方法尝试 {attempt + 1} 失败: {e}, 重试中...")
                    time.sleep(0.5)
                    continue
                self.logger.warning(f"原生方法失败，尝试回退方法: {e}")
                return self._inject_tasks_fallback(tasks)
        
        return False
    
    def _inject_tasks_fallback(self, tasks: List) -> bool:
        """回退方法：直接向 Tasks 数据库注入任务数据（带重试）。"""
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"使用回退方法注入 Tasks 数据（尝试 {attempt + 1}/{max_retries}）...")
                
                sqlite_utils.insert_rows_to_remote_db(
                    rows=tasks,
                    exclude_key='_id',  # 自增主键
                    table_name='tasks',
                    remote_db_file_path='/data/data/org.tasks/databases/database',
                    app_name='tasks',
                    env=self.env,
                    timeout_sec=60.0
                )
                
                self.logger.info(f"成功使用回退方法注入 {len(tasks)} 个任务")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"回退方法尝试 {attempt + 1} 失败: {e}, 重试中...")
                    time.sleep(0.5)
                    continue
                self.logger.error(f"回退方法也失败: {e}")
                return False
        
        return False
    
    def _inject_calendar_data_safe(self) -> bool:
        """安全地注入日历数据。"""
        try:
            return self._inject_calendar_data()
        except Exception as e:
            self.logger.warning(f"注入日历数据失败: {e}")
            return False
    
    def _inject_calendar_data(self) -> bool:
        """
        注入日历事件（25个固定事件）。
        
        使用 calendar_utils.add_events()。
        """
        self.logger.info("注入日历数据（25个固定事件）...")
        
        # 清空现有日历
        self.logger.info("清空现有日历...")
        try:
            calendar_utils.clear_calendar_db(self.env)
        except Exception as e:
            self.logger.warning(f"清空日历时出现警告: {e}")
        
        # 使用 device_constants.DT 作为基准时间
        base_dt = device_constants.DT
        base_timestamp = int(base_dt.timestamp())
        
        # 获取确定性日历事件
        events = deterministic_data.create_deterministic_calendar_events(base_timestamp)
        self.logger.info(f"加载了 {len(events)} 个日历事件")
        
        if not events:
            self.logger.warning("没有加载到日历事件")
            return False
        
        # 使用 AndroidWorld 原生方法注入
        calendar_utils.add_events(events, self.env)
        
        self.logger.info(f"成功注入 {len(events)} 个日历事件")
        return True
    
    def _inject_joplin_data_safe(self) -> bool:
        """安全地注入 Joplin 数据。"""
        try:
            return self._inject_joplin_data()
        except Exception as e:
            self.logger.warning(f"注入 Joplin 数据失败: {e}")
            return False
    
    def _inject_joplin_data(self) -> bool:
        """
        注入 Joplin 笔记（12个文件夹，300+笔记）。
        
        完全按照 reference 实现，确保应用已初始化。
        """
        self.logger.info("注入 Joplin 笔记数据...")
        
        # 确保应用已初始化（这会创建数据库）
        self._ensure_app_ready(
            'net.cozic.joplin',
            'net.cozic.joplin.MainActivity'
        )
        
        # 获取统计信息
        all_folders = deterministic_data.get_all_joplin_folders()
        total_notes = deterministic_data.get_total_joplin_notes_count()
        self.logger.info(f"将注入 {total_notes} 个笔记，分布在 {len(all_folders)} 个文件夹中")
        
        # 清空现有数据
        self.logger.info("清空现有 Joplin 数据...")
        try:
            joplin_app_utils.clear_dbs(self.env)
        except Exception as e:
            self.logger.warning(f"清空 Joplin 时出现警告: {e}")
        
        # 获取所有笔记数据
        all_notes_data = deterministic_data.get_all_joplin_notes()
        
        # 使用 AndroidWorld 原生的笔记创建方法
        notes = []
        folder_mapping = {}
        
        for folder_name in sorted(all_notes_data.keys()):
            folder_notes = all_notes_data[folder_name]
            self.logger.debug(f"创建文件夹 {folder_name} 中的 {len(folder_notes)} 个笔记")
            
            for note_data in folder_notes:
                try:
                    note = joplin_app_utils.create_note(
                        folder=folder_name,
                        title=note_data['title'],
                        body=note_data['body'],
                        folder_mapping=folder_mapping,
                        env=self.env
                    )
                    notes.append(note)
                except Exception as e:
                    self.logger.debug(f"创建笔记失败: {e}")
        
        if not notes:
            self.logger.warning("没有创建任何 Joplin 笔记")
            return False
        
        # 使用 AndroidWorld 原生方法注入
        joplin_app_utils.add_notes(notes, self.env)
        
        self.logger.info(f"成功注入 {len(notes)} 个 Joplin 笔记")
        return True
    
    def _inject_opentracks_data_safe(self) -> bool:
        """安全地注入 OpenTracks 数据。"""
        try:
            return self._inject_opentracks_data()
        except Exception as e:
            self.logger.warning(f"注入 OpenTracks 数据失败: {e}")
            return False
    
    def _inject_opentracks_data(self) -> bool:
        """
        注入 OpenTracks 活动数据（16个运动类别）。
        
        使用 deterministic_data.create_deterministic_activities() 确保数据一致性。
        不手动设置 uuid，让 AndroidWorld 原生 SportsActivity 使用默认 uuid4()。
        """
        self.logger.info("注入 OpenTracks 活动数据...")
        
        # 确保应用已初始化
        self._ensure_app_ready(
            'de.dennisguse.opentracks',
            'de.dennisguse.opentracks.TrackListActivity'
        )
        
        # 获取统计信息
        all_categories = deterministic_data.get_all_activity_categories()
        self.logger.info(f"将注入 {len(all_categories)} 个运动类别的活动")
        
        # 清空现有数据
        self.logger.info("清空现有 OpenTracks 数据...")
        try:
            activity_app_utils.clear_db(self.env)
        except Exception as e:
            self.logger.warning(f"清空 OpenTracks 时出现警告: {e}")
        
        # 使用 device_constants.DT 作为基准时间
        base_dt = device_constants.DT
        base_timestamp_ms = int(base_dt.timestamp() * 1000)
        
        # 使用 deterministic_data 创建活动数据（使用默认 uuid4）
        activities = deterministic_data.create_deterministic_activities(base_timestamp_ms)
        
        self.logger.info(f"创建了 {len(activities)} 个 AndroidWorld 格式活动")
        
        if not activities:
            self.logger.warning("没有创建活动数据")
            return False
        
        # 统计每个类别的活动数
        category_counts = {}
        for act in activities:
            cat = act.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        for cat, count in sorted(category_counts.items()):
            self.logger.debug(f"  {cat}: {count} 个活动")
        
        # 使用 AndroidWorld 原生方法注入
        try:
            activity_app_utils._add_activities(activities, self.env)
            self.logger.info(f"成功注入 {len(activities)} 个活动")
            return True
        except Exception as e:
            self.logger.error(f"注入活动失败: {e}")
            return False
    
    def _inject_expense_data_safe(self) -> bool:
        """安全地注入费用数据。"""
        try:
            return self._inject_expense_data()
        except Exception as e:
            self.logger.warning(f"注入费用数据失败: {e}")
            return False
    
    def _complete_expense_app_initialization(self) -> bool:
        """
        完成 Pro Expense 应用的完整初始化流程。
        
        与 reference/MobileForge Explore/parallel_exploration/app_data_injector.py 对齐：
        1. 清空应用数据
        2. 授予权限
        3. 启动应用
        4. 点击 NEXT 按钮
        5. 点击 CONTINUE 按钮
        6. 关闭应用
        """
        from android_world.env import tools
        
        package_name = 'com.arduia.expense'
        app_name = 'pro expense'
        
        self.logger.info(f"开始完整初始化 {app_name} 应用...")
        
        try:
            # 1. 清空应用数据
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            
            # 2. 授予权限
            for permission in ['android.permission.READ_EXTERNAL_STORAGE',
                              'android.permission.WRITE_EXTERNAL_STORAGE']:
                try:
                    adb_utils.grant_permissions(package_name, permission, self.env.controller)
                except Exception:
                    pass
            
            # 3. 启动应用
            adb_utils.launch_app(app_name, self.env.controller)
            time.sleep(2.0)
            
            # 4. 点击 NEXT 按钮
            controller = tools.AndroidToolController(env=self.env.controller)
            try:
                controller.click_element("NEXT")
                time.sleep(2.0)
            except Exception:
                self._click_ui_element_by_text("NEXT")
                time.sleep(2.0)
            
            # 5. 点击 CONTINUE 按钮
            try:
                controller.click_element("CONTINUE")
                time.sleep(3.0)
            except Exception:
                self._click_ui_element_by_text("CONTINUE")
                time.sleep(3.0)
            
            # 6. 关闭应用
            adb_utils.close_app(app_name, self.env.controller)
            
            self.logger.info(f"{app_name} 应用初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"{app_name} 应用初始化失败: {e}")
            try:
                adb_utils.issue_generic_request(
                    ['shell', 'am', 'force-stop', package_name],
                    self.env.controller
                )
            except Exception:
                pass
            return False
    
    def _inject_expense_data(self) -> bool:
        """
        注入费用数据，使用完全确定性的固定费用模板。
        
        与 reference/MobileForge Explore 对齐：
        1. 完成应用完整初始化（pm clear + NEXT + CONTINUE）
        2. 清理现有数据
        3. 注入确定性费用数据
        """
        self.logger.info("注入费用数据（使用固定费用模板）...")
        
        # 完成应用完整初始化（与 Explore 的 _complete_expense_app_initialization 对齐）
        self._complete_expense_app_initialization()
        
        # 额外等待以确保数据库初始化完成
        time.sleep(2)
        
        # 清理现有 expense 数据，确保数据一致性
        try:
            adb_utils.execute_sql_command(
                '/data/data/com.arduia.expense/databases/accounting.db',
                'DELETE FROM expense;',
                self.env.controller
            )
            self.logger.info("已清理现有 expense 数据")
        except Exception as e:
            self.logger.warning(f"清理 expense 数据时出现警告: {e}")
        
        # 使用 deterministic_data 中的固定费用模板（完全确定性）
        base_timestamp_ms = int(device_constants.DT.timestamp() * 1000)
        expenses = deterministic_data.create_deterministic_expenses(base_timestamp_ms)
        
        if not expenses:
            self.logger.warning("没有加载到费用数据")
            return False
        
        self.logger.info(f"加载了 {len(expenses)} 个确定性费用记录")
        
        # 使用 AndroidWorld 原生方法注入
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
            self.logger.info(f"成功注入 {len(expenses)} 个费用记录（固定模板）")
            return True
        except Exception as e:
            self.logger.error(f"注入费用数据失败: {e}")
            return False
    
    def _inject_markor_data_safe(self) -> bool:
        """安全地注入 Markor 数据。"""
        try:
            return self._inject_markor_data()
        except Exception as e:
            self.logger.warning(f"注入 Markor 数据失败: {e}")
            return False
    
    def _inject_markor_data(self) -> bool:
        """
        注入 Markor 文档（10个固定文档）。
        
        与 MobileForge Explore/parallel_exploration/app_data_injector.py 完全一致:
        - 创建 Markor 目录并设置权限
        - 清理现有文件
        - 使用 user_data_generation.write_to_markor 写入文件
        - 权限设置和媒体扫描
        """
        import time
        
        self.logger.info("注入 Markor 文档数据（与 MobileForge Explore 一致）...")
        
        # 确保 Markor 应用已准备好（授予权限）
        self._ensure_app_ready(
            'net.gsantner.markor',
            'net.gsantner.markor.activity.MainActivity'
        )
        
        # 等待应用初始化
        time.sleep(2)
        
        # 创建 Markor 目录并设置权限（与 reference 完全一致）
        markor_dir = "/storage/emulated/0/Documents/Markor"
        adb_utils.issue_generic_request(
            ['shell', 'mkdir', '-p', markor_dir],
            self.env.controller
        )
        adb_utils.issue_generic_request(
            ['shell', 'chmod', '777', markor_dir],
            self.env.controller
        )
        
        # 清理现有的 Markor 文件（确保数据一致性）
        try:
            adb_utils.issue_generic_request(
                ['shell', 'rm', '-f', f'{markor_dir}/*.md'],
                self.env.controller
            )
            self.logger.debug("已清理现有 Markor 文件")
        except Exception as e:
            self.logger.debug(f"清理 Markor 文件时的警告: {e}")
        
        # 获取确定性文档数据
        documents = deterministic_data.get_all_markor_documents()
        self.logger.info(f"将注入 {len(documents)} 个 Markor 文档")
        
        success_count = 0
        for doc in documents:
            try:
                filename = doc['filename']
                content = doc['content']
                
                # 使用 AndroidWorld 原生方法写入文件（与 reference 完全一致）
                user_data_generation.write_to_markor(
                    content,
                    filename,
                    self.env
                )
                
                # 设置文件权限确保可读
                file_path = f"{markor_dir}/{filename}"
                adb_utils.issue_generic_request(
                    ['shell', 'chmod', '644', file_path],
                    self.env.controller
                )
                
                # 验证文件是否成功创建
                if self._verify_markor_file_created(filename):
                    success_count += 1
                    self.logger.debug(f"成功创建 Markor 文件: {filename}")
                
            except Exception as e:
                self.logger.debug(f"写入 Markor 文档失败 ({doc.get('filename', 'unknown')}): {e}")
        
        # 触发媒体扫描以更新文件索引
        self._trigger_media_scan(markor_dir)
        
        # 强制停止 Markor 应用以确保下次启动时刷新文件列表
        try:
            adb_utils.issue_generic_request(
                ['shell', 'am', 'force-stop', 'net.gsantner.markor'],
                self.env.controller
            )
        except Exception as e:
            self.logger.debug(f"停止 Markor 应用时的警告: {e}")
        
        self.logger.info(f"成功注入 {success_count}/{len(documents)} 个 Markor 文档")
        return success_count > 0
    
    def _trigger_media_scan(self, directory: str) -> None:
        """触发媒体扫描以更新文件索引。"""
        try:
            # 发送 MEDIA_SCANNER_SCAN_FILE 广播
            adb_utils.issue_generic_request(
                ['shell', 'am', 'broadcast', '-a', 'android.intent.action.MEDIA_SCANNER_SCAN_FILE',
                 '-d', f'file://{directory}'],
                self.env.controller
            )
            self.logger.debug(f"触发媒体扫描: {directory}")
        except Exception as e:
            self.logger.debug(f"触发媒体扫描失败: {e}")
    
    def _create_markor_directories(self) -> None:
        """创建 Markor 工作目录结构。"""
        directories = [
            "/storage/emulated/0/Documents/Markor",
            "/storage/emulated/0/Documents/Markor/Personal",
            "/storage/emulated/0/Documents/Markor/Work",
            "/storage/emulated/0/Documents/Markor/Notes"
        ]
        
        for directory in directories:
            try:
                adb_utils.issue_generic_request(
                    ['shell', 'mkdir', '-p', directory],
                    self.env.controller
                )
                # 设置目录权限
                adb_utils.issue_generic_request(
                    ['shell', 'chmod', '777', directory],
                    self.env.controller
                )
            except Exception as e:
                self.logger.debug(f"创建目录 {directory} 时的警告: {e}")
    
    def _verify_markor_file_created(self, filename: str) -> bool:
        """验证 Markor 文件是否成功创建。"""
        try:
            file_path = f"/storage/emulated/0/Documents/Markor/{filename}"
            result = adb_utils.issue_generic_request(
                ['shell', 'test', '-f', file_path, '&&', 'echo', 'exists'],
                self.env.controller
            )
            output = ''
            if result.generic:
                output = result.generic.output.decode('utf-8', errors='ignore') if isinstance(result.generic.output, bytes) else str(result.generic.output)
            return 'exists' in output
        except Exception as e:
            self.logger.debug(f"验证文件 {filename} 时出错: {e}")
            return False
    
    def _inject_messaging_data_safe(self) -> bool:
        """安全地注入 SMS 和通话记录数据。"""
        try:
            return self._inject_messaging_data()
        except Exception as e:
            self.logger.warning(f"注入消息数据失败: {e}")
            return False
    
    def _inject_messaging_data(self) -> bool:
        """
        注入 SMS 消息和通话记录。
        
        与 reference/android_world 的流程对齐：
        - 默认 SMS 应用的设置和快照恢复由 clear_sms_app_data() 处理
        - 本函数只负责：关闭飞行模式 -> 清空系统 SMS DB -> 注入消息 -> 验证
        
        关键步骤：
        1. 关闭飞行模式（防御性措施，与 reference sms_validators.py 一致）
        2. 授予 SMS 应用必要权限
        3. 清空系统 SMS 数据库（防御性，确保注入前为空）
        4. 使用 text_emulator 模拟接收短信
        5. 验证注入结果
        6. 注入通话记录
        
        注意：text_emulator 模拟的是接收短信，因此只注入 type=1 的消息。
        """
        self.logger.info("注入 SMS 消息和通话记录...")
        
        sms_package = "com.simplemobiletools.smsmessenger"
        sms_app_name = "simple sms messenger"
        
        # 步骤1：关闭飞行模式（与 reference sms_validators.py initialize_task 一致）
        # 防御性措施：如果前一个任务开启了飞行模式，确保电话功能可用
        self.logger.info("关闭飞行模式（确保电话功能可用）...")
        try:
            adb_utils.toggle_airplane_mode("off", self.env.controller)
        except Exception as e:
            self.logger.warning(f"关闭飞行模式失败: {e}")
        
        # 步骤2：授予 SMS 应用必要权限
        self.logger.info("授予 SMS 应用必要权限...")
        self._grant_app_permissions(sms_package)
        
        # 步骤3：验证默认 SMS 应用状态（只验证不设置，设置已由快照恢复或 setup 完成）
        # 使用 gRPC 统一 ADB 通道，避免 subprocess 与 gRPC 看到不同设备状态
        try:
            result = adb_utils.issue_generic_request(
                ['shell', 'settings', 'get', 'secure', 'sms_default_application'],
                self.env.controller
            )
            current_default = self._decode_adb_output(
                result.generic.output if result.generic else ''
            )
            if sms_package in current_default:
                self.logger.info(f"验证通过：{sms_package} 是默认 SMS 应用")
            else:
                self.logger.warning(f"默认 SMS 应用不是 {sms_package}，当前为 '{current_default.strip()}'，尝试修正...")
                # 应急修正：快照可能未正确恢复，尝试重新设置
                try:
                    adb_utils.set_default_app(
                        "sms_default_application",
                        sms_package,
                        self.env.controller,
                    )
                    # 使用 AndroidToolController 进行 UI 确认（与 reference 一致）
                    from android_world.env import tools
                    adb_utils.launch_app(sms_app_name, self.env.controller)
                    try:
                        controller = tools.AndroidToolController(env=self.env.controller)
                        time.sleep(2.0)
                        controller.click_element("SMS Messenger")
                        time.sleep(2.0)
                        controller.click_element("Set as default")
                    except Exception:
                        pass
                    finally:
                        adb_utils.close_app(sms_app_name, self.env.controller)
                    self.logger.info("已应急修正默认 SMS 应用设置")
                except Exception as fix_err:
                    self.logger.warning(f"应急修正默认 SMS 应用失败: {fix_err}")
        except Exception as e:
            self.logger.warning(f"验证默认 SMS 应用失败: {e}")
        
        # 步骤4：验证 telephony ContentProvider 缓存状态
        # clear_sms_and_threads() + ContentProvider API 删除 + Provider 重启
        # 已在 clear_sms_app_data() 中完成。此处做最终验证查询，
        # 与 reference sms_validators.py:240 get_sent_messages() 的作用一致。
        self.logger.info("验证 telephony ContentProvider 缓存状态...")
        try:
            result = adb_utils.issue_generic_request(
                "shell content query --uri content://sms/sent".split(),
                self.env.controller
            )
            self.logger.info("ContentProvider 缓存验证完成")
        except Exception as e:
            self.logger.warning(f"验证 ContentProvider 缓存失败: {e}")
        
        # 与 reference sms_validators.py:241 的 time.sleep(5) 对齐，
        # 确保 telephony 框架完全处理 DB 变更后再注入新消息
        time.sleep(5)
        
        # 步骤5：获取并注入 SMS 数据
        sms_data = deterministic_data.get_all_sms_messages()
        # 只注入收到的消息（type=1），因为 adb emu sms send 模拟的是接收短信
        incoming_messages = [msg for msg in sms_data if msg['type'] == 1]
        self.logger.info(f"准备注入 {len(incoming_messages)} 条收件箱消息...")
        
        sms_count = 0
        failed_count = 0
        for i, msg in enumerate(incoming_messages):
            # 清理电话号码（移除非数字字符，保留+号）
            phone = re.sub(r"[^0-9+]", "", msg['address'])
            message_text = msg['body']
            
            injected = False
            inject_method = None
            
            # 使用 adb_utils.text_emulator（通过 gRPC），
            # 与 reference/android_world sms.py 的注入方式完全一致。
            # 统一使用 gRPC 通道，避免混用 subprocess 导致设备状态不一致。
            # 失败时进行一次重试（间隔 1 秒）。
            for attempt in range(2):
                try:
                    adb_utils.text_emulator(
                        self.env.controller,
                        phone,
                        message_text
                    )
                    injected = True
                    inject_method = "text_emulator"
                    break
                except Exception as e:
                    if attempt == 0:
                        self.logger.warning(f"[{i+1}/{len(incoming_messages)}] text_emulator 首次注入失败，1秒后重试: {e}")
                        time.sleep(1)
                    else:
                        self.logger.error(f"[{i+1}/{len(incoming_messages)}] text_emulator 重试仍失败: {e}")
            
            if injected:
                sms_count += 1
                if i < 3 or (i + 1) % 10 == 0:
                    self.logger.info(f"[{i+1}/{len(incoming_messages)}] 成功注入 SMS: {phone} -> '{message_text[:30]}...' (方法: {inject_method})")
            else:
                failed_count += 1
                self.logger.error(f"[{i+1}/{len(incoming_messages)}] 注入失败: {phone} -> '{message_text[:30]}...'")
            
            # 短暂延迟避免消息顺序混乱
            time.sleep(0.5)
        
        self.logger.info(f"SMS 注入完成: 成功 {sms_count}/{len(incoming_messages)} 条, 失败 {failed_count} 条")
        
        # 步骤6：验证 SMS 是否真的存在于设备数据库中
        # 统一使用 gRPC 通道查询，避免 subprocess 与 gRPC 看到不同设备状态
        try:
            verify_result = adb_utils.issue_generic_request(
                ['shell', 'content', 'query', '--uri', 'content://sms', '--projection', '_id'],
                self.env.controller
            )
            output = self._decode_adb_output(
                verify_result.generic.output if verify_result.generic else ''
            )
            lines = [l for l in output.split('\n') if l.strip()]
            sms_in_db = len([l for l in lines if l.strip().startswith('Row:')])
            self.logger.info(f"验证: 系统 SMS 数据库中共有 {sms_in_db} 条消息")
            
            if sms_in_db == 0 and sms_count > 0:
                self.logger.error(f"警告: SMS 注入命令执行成功 ({sms_count} 条)，但数据库中没有消息！"
                                  "可能原因：飞行模式未关闭、应用非默认 SMS 应用、emu sms send 未生效")
        except Exception as e:
            self.logger.warning(f"验证 SMS 数量失败: {e}")
        
        # 步骤7：注入通话记录
        call_count = self._inject_call_history()
        self.logger.info(f"成功注入 {call_count} 条通话记录")
        
        return sms_count > 0 or call_count > 0
    
    def _inject_call_history(self) -> int:
        """
        注入通话记录，使用与 MobileForge Explore 完全相同的数据。
        
        使用 deterministic_data 中的固定通话记录，无需随机数，
        保证每次注入结果完全一致。
        """
        # 获取预定义的固定数据 - 格式与 MobileForge Explore 完全一致
        call_data = deterministic_data.get_all_call_history()
        
        call_count = 0
        
        for call in call_data:
            try:
                phone = call['number']
                call_type = call['type']  # 1=incoming, 2=outgoing, 3=missed
                duration = call['duration']
                timestamp_ms = call['date']  # 已经计算好的时间戳
                
                # 使用 content provider 插入通话记录
                call_cmd = (
                    f'content insert --uri content://call_log/calls '
                    f'--bind number:s:{phone} '
                    f'--bind type:i:{call_type} '
                    f'--bind duration:i:{duration} '
                    f'--bind date:l:{timestamp_ms}'
                )
                
                adb_utils.issue_generic_request(
                    ['shell', call_cmd],
                    self.env.controller
                )
                call_count += 1
                
            except Exception as e:
                self.logger.debug(f"插入通话记录失败: {e}")
                continue
        
        return call_count
    
    def _setup_sms_app_fallback(self, sms_package: str, sms_activity: str) -> None:
        """
        回退方法：手动设置 SMS 应用为默认应用。
        
        当 AndroidWorld 原生的 SimpleSMSMessengerApp.setup() 失败时使用此方法。
        
        Args:
            sms_package: SMS 应用包名
            sms_activity: SMS 应用主 Activity
        """
        self.logger.info("使用回退方法设置 SMS 应用为默认应用...")
        
        # 授予 SMS 应用必要权限
        self._grant_app_permissions(sms_package)
        self.logger.info(f"已授予 {sms_package} 必要权限")
        
        # 设置默认 SMS 应用（命令行方式）
        try:
            self._adb_shell(f'settings put secure sms_default_application {sms_package}')
            self.logger.info(f"已设置 {sms_package} 为默认 SMS 应用（命令行）")
        except Exception as e:
            self.logger.warning(f"设置默认 SMS 应用失败: {e}")
        
        # 启动应用并通过 UI 确认设置为默认
        try:
            self.logger.info("启动 SMS Messenger 应用并确认设置为默认...")
            
            # 启动应用
            self._adb_shell(f'am start -n {sms_package}/{sms_activity}')
            time.sleep(3)
            
            # 尝试通过 UI 自动化点击确认设置默认应用
            try:
                # 点击 "SMS Messenger" 选项
                self._click_ui_element_by_text("SMS Messenger")
                time.sleep(1)
                # 点击 "Set as default" 按钮
                self._click_ui_element_by_text("Set as default")
                time.sleep(2)
            except Exception as ui_err:
                self.logger.debug(f"UI 自动化点击失败，尝试坐标点击: {ui_err}")
                # 回退到坐标点击（基于截图中的位置）
                try:
                    self._adb_shell('input tap 350 877')  # 点击 SMS Messenger 选项
                    time.sleep(1)
                    self._adb_shell('input tap 540 958')  # 点击 Set as default 按钮
                    time.sleep(2)
                except Exception:
                    pass
            
            # 关闭应用
            self._adb_shell(f'am force-stop {sms_package}')
            self.logger.info("SMS Messenger 默认应用设置完成（回退方法）")
            
        except Exception as e:
            self.logger.warning(f"回退方法设置默认应用失败: {e}")
    
    def inject_data_for_package(self, package_name: str) -> bool:
        """
        为指定应用注入数据。
        
        Args:
            package_name: 应用包名
            
        Returns:
            是否成功
        """
        # 包名到注入函数的映射
        package_to_injector = {
            "com.google.android.contacts": self._inject_contacts_data_safe,
            "com.google.android.dialer": self._inject_messaging_data_safe,  # 拨号器需要联系人和通话记录
            "com.simplemobiletools.smsmessenger": self._inject_messaging_data_safe,  # SMS 应用
            "com.simplemobiletools.calendar.pro": self._inject_calendar_data_safe,
            "com.flauschcode.broccoli": self._inject_recipe_data_safe,
            "org.tasks": self._inject_tasks_data_safe,
            "net.cozic.joplin": self._inject_joplin_data_safe,
            "net.gsantner.markor": self._inject_markor_data_safe,
            "de.dennisguse.opentracks": self._inject_opentracks_data_safe,
            "com.arduia.expense": self._inject_expense_data_safe,
            "com.google.android.documentsui": self._inject_files_data_safe,  # Files 应用
            "org.videolan.vlc": self._inject_vlc_data_safe,  # VLC 应用
            "com.simplemobiletools.gallery.pro": self._inject_gallery_data_safe,  # Gallery 应用
            "code.name.monkey.retromusic": self._inject_retro_music_data_safe,  # Retro Music 应用
        }
        
        if package_name not in package_to_injector:
            self.logger.debug(f"包 {package_name} 没有对应的数据注入器")
            return True
        
        try:
            injector_func = package_to_injector[package_name]
            return injector_func()
        except Exception as e:
            self.logger.error(f"为 {package_name} 注入数据失败: {e}")
            return False
    
    def clear_data_for_package(self, package_name: str) -> bool:
        """
        清空指定应用的数据。
        
        Args:
            package_name: 应用包名
            
        Returns:
            是否成功
        """
        # 包名到清理函数的映射
        package_to_cleaner = {
            "com.google.android.contacts": lambda: contacts_utils.clear_contacts(self.env.controller),
            "com.simplemobiletools.calendar.pro": lambda: calendar_utils.clear_calendar_db(self.env),
            "org.tasks": lambda: task_app_utils.clear_task_db(self.env),
            "net.cozic.joplin": lambda: joplin_app_utils.clear_dbs(self.env),
            "de.dennisguse.opentracks": lambda: activity_app_utils.clear_db(self.env),
        }
        
        if package_name not in package_to_cleaner:
            # 对于没有专门清理函数的应用，使用 pm clear
            try:
                adb_utils.issue_generic_request(
                    ['shell', 'pm', 'clear', package_name],
                    self.env.controller
                )
                return True
            except Exception as e:
                self.logger.warning(f"清空应用 {package_name} 失败: {e}")
                return False
        
        try:
            cleaner_func = package_to_cleaner[package_name]
            cleaner_func()
            return True
        except Exception as e:
            self.logger.warning(f"清空应用 {package_name} 数据失败: {e}")
            return False
    
    def verify_all_data_injection(self) -> Dict[str, Dict[str, Any]]:
        """验证所有应用的数据注入结果。"""
        self.logger.info("=" * 60)
        self.logger.info("验证数据注入结果...")
        self.logger.info("=" * 60)
        
        results = {
            'contacts': self._verify_contacts_data(),
            'recipes': self._verify_recipes_data(),
            'tasks': self._verify_tasks_data(),
            'joplin': self._verify_joplin_data(),
            'opentracks': self._verify_opentracks_data(),
            'calendar': self._verify_calendar_data(),
            'markor': self._verify_markor_data(),
            'expenses': self._verify_expenses_data(),
        }
        
        # 统计结果
        total_apps = len(results)
        successful_apps = sum(1 for r in results.values() if r.get('success', False))
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("数据注入验证报告")
        self.logger.info("=" * 60)
        
        self.logger.info("期望的确定性数据量:")
        self.logger.info("  - Recipes (Broccoli): 39 个食谱")
        self.logger.info("  - Tasks: 20 个任务")
        self.logger.info("  - Joplin: 12 个文件夹, 300+ 笔记")
        self.logger.info("  - OpenTracks: 16 个运动类别")
        self.logger.info("  - Calendar: 25 个事件")
        self.logger.info("  - Markor: 10 个文档")
        self.logger.info("  - Contacts: 50 个联系人")
        self.logger.info("")
        
        self.logger.info("验证结果:")
        for app_name, result in results.items():
            status = "✅" if result.get('success', False) else "❌"
            message = result.get('message', '未知')
            self.logger.info(f"  {status} {app_name}: {message}")
        
        self.logger.info("")
        self.logger.info(f"总计: {successful_apps}/{total_apps} 应用数据验证成功")
        self.logger.info("=" * 60)
        
        return results
    
    def _verify_contacts_data(self) -> Dict[str, Any]:
        """验证联系人数据。"""
        try:
            expected_count = 50
            query_result = adb_utils.issue_generic_request(
                ['shell', 'content', 'query',
                 '--uri', 'content://com.android.contacts/contacts',
                 '--projection', '_id'],
                self.env.controller
            )
            
            output = self._decode_adb_output(query_result.generic.output if query_result.generic else '')
            lines = [l for l in output.strip().split('\n') if 'Row:' in l]
            count = len(lines)
            
            if count >= expected_count:
                return {'success': True, 'message': f'成功注入 {count} 个联系人'}
            elif count > 0:
                return {'success': False, 'message': f'只有 {count} 个联系人 (期望 {expected_count})'}
            else:
                return {'success': False, 'message': '没有找到联系人'}
        except Exception as e:
            return {'success': False, 'message': f'验证失败: {e}'}
    
    def _verify_recipes_data(self) -> Dict[str, Any]:
        """验证食谱数据。"""
        try:
            db_path = '/data/data/com.flauschcode.broccoli/databases/broccoli'
            expected_count = 39
            
            result = adb_utils.issue_generic_request(
                ['shell', 'sqlite3', db_path, 'SELECT COUNT(*) FROM recipes;'],
                self.env.controller
            )
            
            output = self._decode_adb_output(result.generic.output if result.generic else '')
            count = int(output.strip())
            
            if count >= expected_count:
                return {'success': True, 'message': f'成功注入 {count} 个食谱'}
            else:
                return {'success': False, 'message': f'只有 {count} 个食谱 (期望 {expected_count})'}
        except Exception as e:
            return {'success': False, 'message': f'验证失败: {e}'}
    
    def _verify_tasks_data(self) -> Dict[str, Any]:
        """验证任务数据。"""
        try:
            db_path = '/data/data/org.tasks/databases/database'
            expected_count = 20
            
            result = adb_utils.issue_generic_request(
                ['shell', 'sqlite3', db_path, 'SELECT COUNT(*) FROM tasks WHERE deleted = 0;'],
                self.env.controller
            )
            
            output = self._decode_adb_output(result.generic.output if result.generic else '')
            count = int(output.strip())
            
            if count >= expected_count:
                return {'success': True, 'message': f'成功注入 {count} 个任务'}
            else:
                return {'success': False, 'message': f'只有 {count} 个任务 (期望 {expected_count})'}
        except Exception as e:
            return {'success': False, 'message': f'验证失败: {e}'}
    
    def _verify_joplin_data(self) -> Dict[str, Any]:
        """验证 Joplin 数据。"""
        try:
            db_path = '/data/data/net.cozic.joplin/databases/joplin.sqlite'
            
            result = adb_utils.issue_generic_request(
                ['shell', 'sqlite3', db_path, 'SELECT COUNT(*) FROM notes;'],
                self.env.controller
            )
            
            output = self._decode_adb_output(result.generic.output if result.generic else '')
            count = int(output.strip())
            
            if count > 0:
                return {'success': True, 'message': f'成功注入 {count} 个笔记'}
            else:
                return {'success': False, 'message': '没有找到笔记'}
        except Exception as e:
            return {'success': False, 'message': f'验证失败: {e}'}
    
    def _verify_opentracks_data(self) -> Dict[str, Any]:
        """验证 OpenTracks 数据。"""
        try:
            db_path = '/data/data/de.dennisguse.opentracks/databases/database.db'
            
            result = adb_utils.issue_generic_request(
                ['shell', 'test', '-f', db_path, '&&', 'echo', 'exists'],
                self.env.controller
            )
            
            output = self._decode_adb_output(result.generic.output if result.generic else '')
            
            if 'exists' in output:
                return {'success': True, 'message': 'OpenTracks 数据库存在'}
            else:
                return {'success': False, 'message': 'OpenTracks 数据库不存在'}
        except Exception as e:
            return {'success': False, 'message': f'验证失败: {e}'}
    
    def _verify_calendar_data(self) -> Dict[str, Any]:
        """验证日历数据。"""
        try:
            db_path = '/data/data/com.simplemobiletools.calendar.pro/databases/events.db'
            expected_count = 25
            
            result = adb_utils.issue_generic_request(
                ['shell', 'sqlite3', db_path, 'SELECT COUNT(*) FROM events;'],
                self.env.controller
            )
            
            output = self._decode_adb_output(result.generic.output if result.generic else '')
            count = int(output.strip())
            
            if count >= expected_count:
                return {'success': True, 'message': f'成功注入 {count} 个日历事件'}
            else:
                return {'success': False, 'message': f'只有 {count} 个事件 (期望 {expected_count})'}
        except Exception as e:
            return {'success': False, 'message': f'验证失败: {e}'}
    
    def _verify_markor_data(self) -> Dict[str, Any]:
        """验证 Markor 数据。"""
        try:
            markor_dir = '/storage/emulated/0/Documents/Markor'
            expected_count = 10
            
            result = adb_utils.issue_generic_request(
                ['shell', 'ls', markor_dir, '|', 'wc', '-l'],
                self.env.controller
            )
            
            output = self._decode_adb_output(result.generic.output if result.generic else '')
            count = int(output.strip())
            
            if count >= expected_count:
                return {'success': True, 'message': f'成功注入 {count} 个 Markor 文档'}
            else:
                return {'success': False, 'message': f'只有 {count} 个文档 (期望 {expected_count})'}
        except Exception as e:
            return {'success': False, 'message': f'验证失败: {e}'}
    
    def _verify_expenses_data(self) -> Dict[str, Any]:
        """验证费用数据。"""
        try:
            db_path = '/data/data/com.arduia.expense/databases/accounting.db'
            
            result = adb_utils.issue_generic_request(
                ['shell', 'test', '-f', db_path, '&&', 'echo', 'exists'],
                self.env.controller
            )
            
            output = self._decode_adb_output(result.generic.output if result.generic else '')
            
            if 'exists' in output:
                return {'success': True, 'message': 'Pro Expense 数据库存在'}
            else:
                return {'success': False, 'message': 'Pro Expense 数据库不存在'}
        except Exception as e:
            return {'success': False, 'message': f'验证失败: {e}'}

    # ====================================================================
    # FILES APP DATA INJECTION - 文件系统数据注入
    # ====================================================================

    def _inject_files_data_safe(self) -> bool:
        """安全地注入 Files 应用数据。"""
        try:
            return self._inject_files_data()
        except Exception as e:
            self.logger.warning(f"注入 Files 应用数据失败: {e}")
            return False

    def _inject_files_data(self) -> bool:
        """
        注入 Files 应用（DocumentsUI）的综合文件数据。
        
        按照 reference 实现，包括:
        - 创建丰富的目录结构
        - 注入各类文档文件
        - 注入下载文件
        - 注入多媒体文件
        - 初始化 Files 应用以刷新视图
        """
        import time
        
        self.logger.info("注入 Files 应用综合数据...")
        
        # 创建丰富的目录结构
        self._create_comprehensive_directory_structure()
        
        # 注入各类文件
        self._inject_document_files()
        self._inject_download_files()
        self._inject_multimedia_files()
        self._inject_app_specific_files()
        
        # 初始化 Files 应用以刷新视图
        self._initialize_files_app()
        
        self.logger.info("Files 应用数据注入完成")
        return True

    def _create_comprehensive_directory_structure(self) -> None:
        """创建全面的目录结构供 Files 应用浏览。"""
        try:
            # 基础系统目录
            system_directories = [
                "/storage/emulated/0/DCIM/Camera",
                "/storage/emulated/0/Pictures/Screenshots",
                "/storage/emulated/0/Documents/Work",
                "/storage/emulated/0/Documents/Personal",
                "/storage/emulated/0/Documents/Projects",
                "/storage/emulated/0/Documents/Reports",
                "/storage/emulated/0/Downloads",
                "/storage/emulated/0/Music/Albums",
                "/storage/emulated/0/Music/Playlists",
                "/storage/emulated/0/Videos/Movies",
                "/storage/emulated/0/Videos/Clips",
                "/storage/emulated/0/Audiobooks",
                "/storage/emulated/0/Podcasts",
                "/storage/emulated/0/Recordings",
                "/storage/emulated/0/Archive",
                "/storage/emulated/0/Backup",
                "/storage/emulated/0/Temp",
                "/storage/emulated/0/VLCVideos",
            ]

            for directory in system_directories:
                adb_utils.issue_generic_request(
                    ['shell', 'mkdir', '-p', directory],
                    self.env.controller
                )
                # 设置适当的权限
                adb_utils.issue_generic_request(
                    ['shell', 'chmod', '755', directory],
                    self.env.controller
                )

            self.logger.info(f"创建了 {len(system_directories)} 个目录")

        except Exception as e:
            self.logger.warning(f"创建目录结构失败: {e}")

    def _create_text_file(self, file_path: str, content: str) -> bool:
        """在设备上创建文本文件。"""
        try:
            import os
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            
            # 使用 AndroidWorld 原生的 file_utils.create_file() 方法
            file_utils.create_file(
                file_name=filename,
                directory_path=directory,
                env=self.env.controller,
                content=content
            )
            
            return True
            
        except Exception as e:
            self.logger.debug(f"创建文件 {file_path} 失败: {e}")
            return False

    def _inject_document_files(self) -> None:
        """注入文档文件。"""
        try:
            document_types = [
                ('report_2023.txt', 'Documents/Work', 'Annual Report 2023\n\nThis is a comprehensive annual report.\nKey findings:\n- Revenue increased by 15%\n- Customer satisfaction improved\n- New markets opened'),
                ('meeting_notes.txt', 'Documents/Work', 'Meeting Notes - Project Alpha\n\nAttendees: John, Sarah, Mike\nDate: 2023-10-15\n\nAgenda:\n1. Project status review\n2. Resource allocation\n3. Next steps'),
                ('todo_list.txt', 'Documents/Personal', 'Personal Todo List\n\n1. Buy groceries\n2. Call dentist\n3. Pay bills\n4. Schedule car service\n5. Plan weekend trip'),
                ('recipe_collection.txt', 'Documents/Personal', 'Favorite Recipes\n\nChocolate Cake:\nIngredients:\n- 2 cups flour\n- 1 cup sugar\n- 1/2 cup cocoa\n\nInstructions:\n1. Preheat oven to 350F\n2. Mix dry ingredients'),
                ('project_proposal.txt', 'Documents/Projects', 'Project Proposal - Mobile App\n\nOverview:\nThis project aims to develop a mobile application\nfor task management and productivity.\n\nTimeline: 6 months\nBudget: $50,000'),
                ('budget_2023.txt', 'Documents/Reports', 'Budget Report 2023\n\nIncome: $50,000\nExpenses: $45,000\nSavings: $5,000\n\nBreakdown:\n- Housing: $15,000\n- Food: $6,000\n- Transportation: $4,000'),
                ('notes.txt', 'Documents', 'Quick Notes\n\n- Remember to call mom\n- Meeting at 3pm tomorrow\n- Finish reading chapter 5\n- Buy birthday gift'),
                ('contacts_backup.txt', 'Documents', 'Contact Backup\n\nJohn: 555-0101\nSarah: 555-0102\nMike: 555-0103\nEmergency: 911'),
            ]

            successful_files = 0
            for filename, subfolder, content in document_types:
                full_path = f"/storage/emulated/0/{subfolder}/{filename}"
                
                if self._create_text_file(full_path, content):
                    successful_files += 1
                    self.logger.debug(f"创建文档文件: {filename}")

            self.logger.info(f"创建了 {successful_files}/{len(document_types)} 个文档文件")

        except Exception as e:
            self.logger.warning(f"创建文档文件失败: {e}")

    def _inject_download_files(self) -> None:
        """注入下载文件。"""
        try:
            downloads_path = "/storage/emulated/0/Downloads"
            
            download_files = [
                ('readme.txt', 'README\n\nThis is a downloaded text file.\nVersion: 1.0\nDate: 2023-10-15'),
                ('user_manual.txt', 'User Manual\n\nChapter 1: Getting Started\nChapter 2: Basic Operations\nChapter 3: Advanced Features'),
                ('install_notes.txt', 'Installation Notes\n\n1. Extract the archive\n2. Run setup.exe\n3. Follow the prompts'),
                ('changelog.txt', 'Changelog\n\nv1.0 - Initial release\nv1.1 - Bug fixes\nv1.2 - New features'),
                ('license.txt', 'MIT License\n\nCopyright (c) 2023\n\nPermission is hereby granted, free of charge...'),
            ]

            successful_files = 0
            for filename, content in download_files:
                full_path = f"{downloads_path}/{filename}"
                if self._create_text_file(full_path, content):
                    successful_files += 1
                    self.logger.debug(f"创建下载文件: {filename}")

            self.logger.info(f"创建了 {successful_files} 个下载文件")

        except Exception as e:
            self.logger.warning(f"创建下载文件失败: {e}")

    def _inject_multimedia_files(self) -> None:
        """注入多媒体文件（图片和视频）。"""
        try:
            # 创建图片文件（使用 user_data_generation）
            image_categories = [
                ('vacation_photos', 'Pictures', 3),
                ('family_photos', 'Pictures', 2),
            ]

            for category, folder, count in image_categories:
                for i in range(count):
                    try:
                        filename = f"{category}_{i+1}.jpg"
                        user_data_generation.write_to_gallery(
                            f"{category.replace('_', ' ').title()} Photo {i+1}",
                            filename,
                            self.env
                        )
                    except Exception as e:
                        self.logger.debug(f"创建图片 {filename} 失败: {e}")

            self.logger.info("创建了多媒体文件")

        except Exception as e:
            self.logger.warning(f"创建多媒体文件失败: {e}")

    def _inject_app_specific_files(self) -> None:
        """注入应用特定的文件。"""
        try:
            import datetime
            
            app_files = [
                ('/storage/emulated/0/Backup/contacts_backup.vcf', 'Contacts backup file'),
                ('/storage/emulated/0/Backup/sms_backup.xml', 'SMS backup data'),
                ('/storage/emulated/0/Archive/old_photos.zip', 'Archived photos'),
                ('/storage/emulated/0/Temp/cache_data.tmp', 'Temporary cache data'),
            ]

            for file_path, description in app_files:
                try:
                    directory = '/'.join(file_path.split('/')[:-1])
                    adb_utils.issue_generic_request(
                        ['shell', 'mkdir', '-p', directory],
                        self.env.controller
                    )

                    content = f"{description}\nCreated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    self._create_text_file(file_path, content)
                except Exception as e:
                    self.logger.debug(f"创建应用文件 {file_path} 失败: {e}")

            self.logger.info("创建了应用特定文件")

        except Exception as e:
            self.logger.warning(f"创建应用特定文件失败: {e}")

    def _initialize_files_app(self) -> None:
        """初始化 Files 应用（DocumentsUI）以刷新其视图。"""
        import time
        
        try:
            self.logger.info("初始化 Files 应用 (DocumentsUI)...")
            
            # 启动 Files 应用
            adb_utils.launch_app('files', self.env.controller)
            
            # 等待应用初始化
            time.sleep(2)
            
            # 关闭应用
            adb_utils.close_app('files', self.env.controller)
            
            # 触发媒体扫描以更新文件索引
            adb_utils.issue_generic_request(
                ['shell', 'am', 'broadcast', '-a', 'android.intent.action.MEDIA_SCANNER_SCAN_FILE', 
                 '-d', 'file:///storage/emulated/0/'],
                self.env.controller
            )
            
            self.logger.info("Files 应用初始化完成")
            
        except Exception as e:
            self.logger.warning(f"初始化 Files 应用失败: {e}")

    # ====================================================================
    # VLC DATA INJECTION - VLC 视频数据注入
    # ====================================================================

    def _inject_vlc_data_safe(self) -> bool:
        """安全地注入 VLC 数据。"""
        try:
            return self._inject_vlc_data()
        except Exception as e:
            self.logger.warning(f"注入 VLC 数据失败: {e}")
            return False

    def _complete_vlc_app_initialization(self) -> bool:
        """
        完成 VLC 应用的完整初始化流程。
        
        与 reference/MobileForge Explore/parallel_exploration/app_data_injector.py 对齐：
        1. 清空应用数据
        2. 授予权限
        3. 创建 VLCVideos 目录
        4. 使用 monkey 命令启动应用
        5. 点击 Skip -> GRANT PERMISSION -> OK -> Allow access
        6. 关闭应用
        """
        from android_world.env import tools
        
        package_name = 'org.videolan.vlc'
        app_name = 'vlc'
        videos_path = '/storage/emulated/0/VLCVideos'
        
        self.logger.info(f"开始完整初始化 {app_name} 应用...")
        
        try:
            # 1. 清空应用数据
            adb_utils.clear_app_data(package_name, self.env.controller)
            time.sleep(1)
            
            # 2. 授予权限
            for permission in ['android.permission.POST_NOTIFICATIONS',
                              'android.permission.READ_EXTERNAL_STORAGE',
                              'android.permission.WRITE_EXTERNAL_STORAGE',
                              'android.permission.RECORD_AUDIO']:
                try:
                    adb_utils.grant_permissions(package_name, permission, self.env.controller)
                except Exception:
                    pass
            
            # 3. 创建视频目录
            try:
                if not file_utils.check_directory_exists(videos_path, self.env.controller):
                    file_utils.mkdir(videos_path, self.env.controller)
            except Exception:
                adb_utils.issue_generic_request(
                    ['shell', 'mkdir', '-p', videos_path],
                    self.env.controller
                )
            
            time.sleep(2.0)
            
            # 4. 使用 monkey 命令启动应用（触发完整初始化）
            adb_utils.issue_generic_request(
                ["shell", "monkey", "-p", package_name,
                 "-c", "android.intent.category.LAUNCHER", "1"],
                self.env.controller,
            )
            time.sleep(2.0)
            
            # 5. 点击引导页按钮
            controller = tools.AndroidToolController(env=self.env.controller)
            
            # Skip
            try:
                controller.click_element("Skip")
                time.sleep(2.0)
            except Exception:
                try:
                    self._click_ui_element_by_text("Skip")
                    time.sleep(2.0)
                except Exception:
                    pass
            
            # GRANT PERMISSION
            try:
                controller.click_element("GRANT PERMISSION")
                time.sleep(2.0)
            except Exception:
                try:
                    self._click_ui_element_by_text("GRANT PERMISSION")
                    time.sleep(2.0)
                except Exception:
                    pass
            
            # OK
            try:
                controller.click_element("OK")
                time.sleep(2.0)
            except Exception:
                try:
                    self._click_ui_element_by_text("OK")
                    time.sleep(2.0)
                except Exception:
                    pass
            
            # Allow access to manage all files
            try:
                controller.click_element("Allow access to manage all files")
                time.sleep(2.0)
            except Exception:
                try:
                    if not self._click_ui_element_by_text("Allow access to manage all files"):
                        self._click_ui_element_by_text("Allow")
                    time.sleep(2.0)
                except Exception:
                    pass
            
            # 6. 关闭应用
            adb_utils.close_app(app_name, self.env.controller)
            
            self.logger.info(f"{app_name} 应用初始化完成")
            return True
        except Exception as e:
            self.logger.warning(f"{app_name} 应用初始化失败: {e}")
            try:
                adb_utils.issue_generic_request(
                    ['shell', 'am', 'force-stop', package_name],
                    self.env.controller
                )
            except Exception:
                pass
            return False
    
    def _inject_vlc_data(self) -> bool:
        """
        注入 VLC 视频数据，使用预定义的固定数据。
        
        与 reference/MobileForge Explore 对齐：
        1. 完成 VLC 应用完整初始化
        2. 创建视频目录
        3. 清理现有视频文件
        4. 注入确定性视频文件
        """
        self.logger.info("注入 VLC 视频数据（使用与 MobileForge Explore 一致的固定数据）...")
        
        # 完成 VLC 应用完整初始化（与 Explore 的 _complete_vlc_app_initialization 对齐）
        self._complete_vlc_app_initialization()
        
        # 额外等待以确保初始化完成
        time.sleep(2)
        
        # 创建视频目录（初始化时已创建，这里确保存在）
        vlc_video_dir = "/storage/emulated/0/VLCVideos"
        adb_utils.issue_generic_request(
            ['shell', 'mkdir', '-p', vlc_video_dir],
            self.env.controller
        )
        
        # 清理现有视频文件
        try:
            adb_utils.issue_generic_request(
                ['shell', 'rm', '-f', f'{vlc_video_dir}/*.mp4'],
                self.env.controller
            )
            self.logger.info("已清理现有 VLC 视频文件")
        except Exception as e:
            self.logger.warning(f"清理 VLC 视频目录时出现警告: {e}")
        
        # 获取预定义的完整视频数据（与 MobileForge Explore 一致）
        videos = deterministic_data.get_deterministic_video_files()
        successful_count = 0
        
        for video in videos:
            try:
                # 使用 AndroidWorld 原生方法创建视频文件
                user_data_generation.write_video_file_to_device(
                    video['filename'],
                    video['directory'],
                    self.env,
                    messages=video['messages'],
                    fps=1,
                    message_display_time=video['duration_seconds']
                )
                successful_count += 1
                self.logger.debug(f"创建 VLC 视频: {video['filename']}")
            except Exception as e:
                self.logger.debug(f"创建 VLC 视频 {video['filename']} 失败: {e}")
        
        self.logger.info(f"成功注入 {successful_count}/{len(videos)} 个 VLC 视频文件")
        return successful_count > 0
    
    def _generate_vlc_video_filename(self, index: int) -> str:
        """
        生成 VLC 视频文件名，使用预定义的固定文件名列表。
        
        Args:
            index: 视频索引（从1开始）
            
        Returns:
            固定的视频文件名
        """
        video_filenames = deterministic_data.get_all_vlc_videos()
        # 使用索引取模确保不越界
        return video_filenames[(index - 1) % len(video_filenames)]

    # ====================================================================
    # GALLERY DATA INJECTION - 图库数据注入
    # ====================================================================

    def _inject_gallery_data_safe(self) -> bool:
        """安全地注入图库数据。"""
        try:
            return self._inject_gallery_data()
        except Exception as e:
            self.logger.warning(f"注入图库数据失败: {e}")
            return False

    def _inject_gallery_data(self) -> bool:
        """注入图库数据（图片和视频）。"""
        self.logger.info("注入图库数据...")
        
        try:
            # 创建一组确定性的图片
            gallery_images = [
                ("nature_landscape_1.jpg", "Beautiful mountain landscape"),
                ("nature_landscape_2.jpg", "Ocean sunset view"),
                ("family_photo_1.jpg", "Family gathering"),
                ("family_photo_2.jpg", "Birthday celebration"),
                ("work_presentation_1.jpg", "Project presentation"),
                ("food_recipe_1.jpg", "Homemade pasta"),
                ("travel_paris_1.jpg", "Paris Eiffel Tower"),
                ("travel_tokyo_1.jpg", "Tokyo skyline"),
                ("pet_photo_1.jpg", "Happy dog"),
                ("pet_photo_2.jpg", "Sleeping cat"),
            ]
            
            successful_count = 0
            for filename, description in gallery_images:
                try:
                    user_data_generation.write_to_gallery(
                        description,
                        filename,
                        self.env
                    )
                    successful_count += 1
                except Exception as e:
                    self.logger.debug(f"创建图库图片 {filename} 失败: {e}")
            
            self.logger.info(f"成功注入 {successful_count} 个图库图片")
            return successful_count > 0
            
        except Exception as e:
            self.logger.warning(f"注入图库数据失败: {e}")
            return False

    # ====================================================================
    # RETRO MUSIC DATA INJECTION - Retro Music 数据注入
    # ====================================================================

    def _inject_retro_music_data_safe(self) -> bool:
        """安全地注入 Retro Music 数据。"""
        try:
            return self._inject_retro_music_data()
        except Exception as e:
            self.logger.warning(f"注入 Retro Music 数据失败: {e}")
            return False

    def _inject_retro_music_data(self) -> bool:
        """
        注入 Retro Music 数据（确定性音乐文件）。
        
        与 reference/MobileForge Explore/parallel_exploration/app_data_injector.py 对齐：
        使用 deterministic_data.get_deterministic_music_files() 获取固定的音乐数据，
        然后用 user_data_generation.write_mp3_file_to_device() 写入设备。
        完全不依赖 random，保证每次注入结果一致。
        """
        self.logger.info("注入 Retro Music 数据（使用确定性音乐文件）...")
        
        try:
            # 清理现有音乐文件
            music_dir = "/storage/emulated/0/Music"
            try:
                file_utils.clear_directory(music_dir, self.env.controller)
            except Exception:
                adb_utils.issue_generic_request(
                    ['shell', 'rm', '-rf', f'{music_dir}/*'],
                    self.env.controller
                )
            
            # 清理 Retro Music 播放列表数据库
            retro_music_db = "/data/data/code.name.monkey.retromusic/databases/playlist.db"
            for table_name in ["Playlist", "PlaylistEntity", "SongEntity"]:
                try:
                    if sqlite_utils.table_exists(table_name, retro_music_db, self.env):
                        sqlite_utils.delete_all_rows_from_table(
                            table_name, retro_music_db, self.env, "retro music"
                        )
                except Exception:
                    pass
            
            # 获取确定性音乐文件数据
            music_files = deterministic_data.get_deterministic_music_files()
            self.logger.info(f"将注入 {len(music_files)} 个确定性音乐文件")
            
            successful_count = 0
            for music in music_files:
                try:
                    user_data_generation.write_mp3_file_to_device(
                        f"{music['directory']}/{music['filename']}",
                        self.env,
                        title=music['title'],
                        artist=music['artist'],
                        duration_milliseconds=music['duration_ms'],
                    )
                    successful_count += 1
                except Exception as e:
                    self.logger.debug(f"创建音乐文件 {music['filename']} 失败: {e}")
            
            self.logger.info(f"Retro Music 数据注入完成: {successful_count}/{len(music_files)} 个文件")
            return successful_count > 0
            
        except Exception as e:
            self.logger.warning(f"注入 Retro Music 数据失败: {e}")
            return False
