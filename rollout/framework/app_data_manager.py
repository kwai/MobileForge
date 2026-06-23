"""
App Data Manager - 应用数据管理模块

该模块负责在每个任务执行前清空并重置应用数据，确保初始环境一致。
完全使用 AndroidWorld 原生工具进行数据清理和注入。

参考: reference/MobileForge Emulator Setup/android_world/comprehensive_setup/app_data_injector.py
"""

import logging
import os
import sys
from typing import Optional, Dict, List, Any

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
from android_world.env.setup_device import setup, apps
from android_world.task_evals.information_retrieval import joplin_app_utils
from android_world.task_evals.information_retrieval import task_app_utils
from android_world.task_evals.information_retrieval import activity_app_utils
from android_world.task_evals.single.calendar import calendar_utils
from android_world.task_evals.common_validators import sms_validators
from android_world.utils import contacts_utils
from android_world.utils import file_utils
from android_world.task_evals.utils import sqlite_utils

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 需要首次初始化的应用映射
# 这些应用在 pm clear 后需要重新执行 setup() 来创建数据库
# ============================================================================

APPS_REQUIRING_SETUP = {
    # Broccoli Recipe: 启动应用 2 秒，创建 broccoli 数据库
    "com.flauschcode.broccoli": apps.RecipeApp,
    # Pro Expense: 点击 NEXT -> CONTINUE 完成引导，创建 accounting.db
    "com.arduia.expense": apps.ExpenseApp,
    # Markor: 点击 NEXT x 4 -> DONE -> OK -> Allow access 完成引导
    "net.gsantner.markor": apps.MarkorApp,
    # VLC: 创建目录，点击 Skip 跳过引导，创建 vlc_media.db
    "org.videolan.vlc": apps.VlcApp,
    # Joplin: 点击 GET STARTED -> OK -> ALLOW 完成引导，创建 joplin.sqlite
    "net.cozic.joplin": apps.JoplinApp,
    # Camera: 授予位置权限，点击 NEXT
    "com.android.camera2": apps.CameraApp,
    # Chrome: 接受条款，跳过登录
    "com.android.chrome": apps.ChromeApp,
    # Simple Gallery Pro: 授予权限，点击引导页
    "com.simplemobiletools.gallery.pro": apps.SimpleGalleryProApp,
    # OpenTracks: 启动 + 位置权限 + 蓝牙允许
    "de.dennisguse.opentracks": apps.OpenTracksApp,
    # Retro Music: pm clear + 音频权限
    "code.name.monkey.retromusic": apps.RetroMusicApp,
    # Contacts: pm clear + Skip + Don't allow
    "com.google.android.contacts": apps.ContactsApp,
    # Clock: pm clear + 启动一次（消除 tooltip）
    "com.google.android.deskclock": apps.ClockApp,
    # Simple Calendar Pro: pm clear + 日历权限
    "com.simplemobiletools.calendar.pro": apps.SimpleCalendarProApp,
    # Tasks: pm clear + 启动一次
    "org.tasks": apps.TasksApp,
    # 注意：Simple SMS Messenger 已从此列表移除
    # 因为我们不再使用 pm clear 清空它的数据（保持其为默认 SMS 应用状态）
    # 这样应用可以直接读取系统 SMS 数据库中的新数据
}


# ============================================================================
# App 包名映射
# ============================================================================

APP_PACKAGES = {
    # 系统应用
    "camera": "com.android.camera2",
    "chrome": "com.android.chrome",
    "clock": "com.google.android.deskclock",
    "deskclock": "com.google.android.deskclock",
    "contacts": "com.google.android.contacts",
    "files": "com.google.android.documentsui",
    "documentsui": "com.google.android.documentsui",
    "settings": "com.android.settings",
    "dialer": "com.google.android.dialer",
    # 第三方应用
    "markor": "net.gsantner.markor",
    "calendar": "com.simplemobiletools.calendar.pro",
    "tasks": "org.tasks",
    "draw": "com.simplemobiletools.draw.pro",
    "gallery": "com.simplemobiletools.gallery.pro",
    "sms": "com.simplemobiletools.smsmessenger",
    "smsmessenger": "com.simplemobiletools.smsmessenger",
    "audiorecorder": "com.dimowner.audiorecorder",
    "expense": "com.arduia.expense",
    "broccoli": "com.flauschcode.broccoli",
    "osmand": "net.osmand",
    "opentracks": "de.dennisguse.opentracks",
    "vlc": "org.videolan.vlc",
    "joplin": "net.cozic.joplin",
    "retromusic": "code.name.monkey.retromusic",
}

PACKAGE_TO_APP_NAME = {v: k for k, v in APP_PACKAGES.items()}


# ============================================================================
# 使用 AndroidWorld 原生工具的数据清理函数
# ============================================================================

def clear_contacts_data(env: interface.AsyncEnv) -> bool:
    """清空联系人数据。"""
    logger.info("清空联系人数据...")
    try:
        contacts_utils.clear_contacts(env.controller)
        logger.info("联系人数据已清空")
        return True
    except Exception as e:
        logger.warning(f"清空联系人数据失败: {e}")
        return False


def clear_calendar_data(env: interface.AsyncEnv) -> bool:
    """清空日历数据。"""
    logger.info("清空日历数据...")
    try:
        calendar_utils.clear_calendar_db(env)
        logger.info("日历数据已清空")
        return True
    except Exception as e:
        logger.warning(f"清空日历数据失败: {e}")
        return False


def clear_tasks_data(env: interface.AsyncEnv) -> bool:
    """清空任务数据。"""
    logger.info("清空任务数据...")
    try:
        task_app_utils.clear_task_db(env)
        logger.info("任务数据已清空")
        return True
    except Exception as e:
        logger.warning(f"清空任务数据失败: {e}")
        return False


def clear_joplin_data(env: interface.AsyncEnv) -> bool:
    """清空 Joplin 数据。"""
    logger.info("清空 Joplin 数据...")
    try:
        joplin_app_utils.clear_dbs(env)
        logger.info("Joplin 数据已清空")
        return True
    except Exception as e:
        logger.warning(f"清空 Joplin 数据失败: {e}")
        return False


def clear_opentracks_data(env: interface.AsyncEnv) -> bool:
    """清空 OpenTracks 数据。"""
    logger.info("清空 OpenTracks 数据...")
    try:
        activity_app_utils.clear_db(env)
        logger.info("OpenTracks 数据已清空")
        return True
    except Exception as e:
        logger.warning(f"清空 OpenTracks 数据失败: {e}")
        return False


def clear_sms_data(env: interface.AsyncEnv) -> bool:
    """
    清理系统 SMS 数据。
    
    使用 AndroidWorld 原生的 sms_validators.clear_sms_and_threads() 清理
    /data/data/com.android.providers.telephony/databases/mmssms.db 中的
    sms 和 threads 表。
    
    Args:
        env: AndroidWorld 环境对象
        
    Returns:
        是否成功
    """
    logger.info("清空系统 SMS 数据...")
    try:
        sms_validators.clear_sms_and_threads(env.controller)
        logger.info("系统 SMS 数据已清空")
        return True
    except Exception as e:
        logger.warning(f"清空 SMS 数据失败: {e}")
        return False


def clear_sms_app_data(env: interface.AsyncEnv) -> bool:
    """
    重置 SMS 应用到干净的已配置状态，并清空系统 SMS 数据库。
    
    与 reference/android_world 的流程对齐：
    1. 优先从快照恢复应用数据（快速路径）
       - 快照包含已配置为默认 SMS 应用的干净状态
       - 恢复后应用内部缓存/数据库也回到初始状态
    2. 若无快照，执行 pm clear + 完整 setup + 保存快照（慢速路径，仅首次执行）
    3. 关闭飞行模式（确保 radio/modem 活跃，与 reference sms_validators.py:236 一致）
    4. 清空系统 SMS 数据库（mmssms.db 的 sms 和 threads 表）
    
    参考: reference/android_world task_eval.py _initialize_apps() + sms_validators.py initialize_task()
    
    Args:
        env: AndroidWorld 环境对象
        
    Returns:
        是否成功
    """
    from android_world.utils import app_snapshot
    from android_world.env import tools
    
    sms_app_name = "simple sms messenger"
    sms_package = "com.simplemobiletools.smsmessenger"
    
    # 步骤1：恢复应用快照或重新 setup
    # 优先从快照恢复（快速路径），与 reference task_eval._initialize_apps() 一致
    snapshot_restored = False
    try:
        app_snapshot.restore_snapshot(sms_app_name, env.controller)
        snapshot_restored = True
        logger.info("SMS 应用快照恢复成功（应用已回到干净的已配置状态）")
    except RuntimeError as e:
        logger.info(f"SMS 应用快照不存在或恢复失败: {e}，将执行完整 setup...")
        # 慢速路径：pm clear + 完整 setup + 保存快照
        try:
            # pm clear 清空所有应用数据
            adb_utils.clear_app_data(sms_package, env.controller)
            
            # 设置为默认 SMS 应用
            adb_utils.set_default_app(
                "sms_default_application",
                sms_package,
                env.controller,
            )
            
            # 启动应用并通过 UI 确认默认应用
            adb_utils.launch_app(sms_app_name, env.controller)
            try:
                import time
                controller = tools.AndroidToolController(env=env.controller)
                time.sleep(2.0)
                controller.click_element("SMS Messenger")
                time.sleep(2.0)
                controller.click_element("Set as default")
            except Exception as ui_err:
                logger.debug(f"SMS 默认应用 UI 确认时出现警告（可能已是默认）: {ui_err}")
            finally:
                adb_utils.close_app(sms_app_name, env.controller)
            
            # 保存快照供下次使用
            app_snapshot.save_snapshot(sms_app_name, env.controller)
            snapshot_restored = True
            logger.info("SMS 应用完整 setup 完成并已保存快照")
        except Exception as setup_err:
            logger.warning(f"SMS 应用完整 setup 失败: {setup_err}")
    
    # 步骤2：关闭飞行模式（与 reference sms_validators.py:236 对齐）
    # Reference 在 clear_sms_and_threads 之前关闭飞行模式，确保 radio/modem 在
    # 整个清理和注入流程中都是活跃的，避免 telephony ContentProvider 行为异常
    import time
    try:
        adb_utils.toggle_airplane_mode("off", env.controller)
        logger.info("已关闭飞行模式（确保 telephony 正常工作）")
    except Exception as e:
        logger.warning(f"关闭飞行模式失败: {e}")
    
    # 步骤3：清空系统 SMS 数据库
    # 与 reference sms_validators.clear_sms_and_threads() 一致
    try:
        sms_validators.clear_sms_and_threads(env.controller)
        logger.info("系统 SMS 数据库已清空")
    except Exception as e:
        logger.warning(f"清空系统 SMS 数据库失败: {e}")
    
    # 注意：不再执行 ContentProvider API 删除、kill telephony provider、验证查询等额外操作。
    # Reference/android_world 仅使用 SQL DELETE 清空 sms 和 threads 表，
    # 不做任何 ContentProvider 层面的操作。额外操作（尤其是 kill telephony provider）
    # 会破坏 telephony 框架的正常状态，导致后续 text_emulator 注入的 SMS 无法被持久化。
    
    return snapshot_restored


def clear_expense_data(env: interface.AsyncEnv) -> bool:
    """
    清理 Pro Expense 数据库中的 expense 表。
    
    Args:
        env: AndroidWorld 环境对象
        
    Returns:
        是否成功
    """
    logger.info("清空 Pro Expense 数据...")
    try:
        adb_utils.execute_sql_command(
            '/data/data/com.arduia.expense/databases/accounting.db',
            'DELETE FROM expense;',
            env.controller
        )
        logger.info("Pro Expense 数据已清空")
        return True
    except Exception as e:
        logger.warning(f"清空 Pro Expense 数据失败: {e}")
        return False


def clear_vlc_data(env: interface.AsyncEnv) -> bool:
    """
    清理 VLC 视频目录。
    
    Args:
        env: AndroidWorld 环境对象
        
    Returns:
        是否成功
    """
    logger.info("清空 VLC 视频目录...")
    try:
        vlc_video_dir = "/storage/emulated/0/VLCVideos"
        file_utils.clear_directory(vlc_video_dir, env.controller)
        logger.info("VLC 视频目录已清空")
        return True
    except Exception as e:
        logger.warning(f"清空 VLC 视频目录失败: {e}")
        return False


def clear_call_history_data(env: interface.AsyncEnv) -> bool:
    """
    清空通话记录。
    
    使用 content provider 删除通话记录。
    参考: reference/MobileForge Explore/parallel_exploration/app_data_cleaner.py
    """
    logger.info("清空通话记录...")
    try:
        adb_utils.clear_android_emulator_call_log(env.controller)
        logger.info("通话记录已清空")
        return True
    except Exception as e:
        logger.warning(f"清空通话记录失败: {e}")
        return False


def clear_gallery_data(env: interface.AsyncEnv) -> bool:
    """
    清空相册图片。
    
    删除 DCIM 和 Pictures 目录下的所有文件。
    参考: reference/MobileForge Explore/parallel_exploration/app_data_cleaner.py
    """
    logger.info("清空相册数据...")
    try:
        directories_to_clear = [
            "/storage/emulated/0/DCIM",
            "/storage/emulated/0/Pictures",
        ]
        for directory in directories_to_clear:
            try:
                file_utils.clear_directory(directory, env.controller)
            except Exception as e:
                logger.debug(f"清理目录 {directory} 时出现警告: {e}")
        logger.info("相册数据已清空")
        return True
    except Exception as e:
        logger.warning(f"清空相册数据失败: {e}")
        return False


def clear_files_data(env: interface.AsyncEnv) -> bool:
    """
    清空文件系统数据。
    
    清理内部存储和下载目录的文件。
    参考: reference/MobileForge Explore/parallel_exploration/app_data_cleaner.py
    """
    logger.info("清空文件系统数据...")
    try:
        internal_storage = "/data/local/tmp/android_world"
        try:
            adb_utils.issue_generic_request(
                [
                    "shell", "find", internal_storage,
                    "-mindepth", "1", "-type", "f", "-delete"
                ],
                env.controller
            )
        except Exception as e:
            logger.debug(f"清理内部存储时出现警告: {e}")
        
        try:
            adb_utils.issue_generic_request(
                "shell content delete --uri content://media/external/downloads",
                env.controller,
                timeout_sec=20.0
            )
        except Exception as e:
            logger.debug(f"清理外部下载时出现警告: {e}")
        
        logger.info("文件系统数据已清空")
        return True
    except Exception as e:
        logger.warning(f"清空文件系统数据失败: {e}")
        return False


def clear_recipes_data(env: interface.AsyncEnv) -> bool:
    """
    清空食谱数据库。
    
    删除 Broccoli 应用的 recipes 表数据。
    参考: reference/MobileForge Explore/parallel_exploration/app_data_cleaner.py
    """
    logger.info("清空食谱数据...")
    try:
        db_path = "/data/data/com.flauschcode.broccoli/databases/broccoli"
        table_name = "recipes"
        app_name = "broccoli app"
        
        if sqlite_utils.table_exists(table_name, db_path, env):
            sqlite_utils.delete_all_rows_from_table(
                table_name, db_path, env, app_name
            )
        logger.info("食谱数据已清空")
        return True
    except Exception as e:
        logger.warning(f"清空食谱数据失败: {e}")
        return False


def clear_markor_data(env: interface.AsyncEnv) -> bool:
    """
    清空 Markor 文档。
    
    删除 Markor 目录下的所有文件。
    参考: reference/MobileForge Explore/parallel_exploration/app_data_cleaner.py
    """
    logger.info("清空 Markor 文档...")
    try:
        markor_dir = "/storage/emulated/0/Documents/Markor"
        try:
            file_utils.clear_directory(markor_dir, env.controller)
        except Exception:
            # 回退到 ADB 命令
            adb_utils.issue_generic_request(
                ["shell", "rm", "-rf", f"{markor_dir}/*"],
                env.controller
            )
        logger.info("Markor 文档已清空")
        return True
    except Exception as e:
        logger.warning(f"清空 Markor 文档失败: {e}")
        return False


def clear_music_data(env: interface.AsyncEnv) -> bool:
    """
    清空音乐文件和播放列表。
    
    删除 Music 目录下的文件和 Retro Music 播放列表数据库。
    参考: reference/MobileForge Explore/parallel_exploration/app_data_cleaner.py
    """
    logger.info("清空音乐数据...")
    try:
        music_dir = "/storage/emulated/0/Music"
        retro_music_db = "/data/data/code.name.monkey.retromusic/databases/playlist.db"
        
        # 清理音乐文件
        try:
            file_utils.clear_directory(music_dir, env.controller)
        except Exception:
            adb_utils.issue_generic_request(
                ["shell", "rm", "-rf", f"{music_dir}/*"],
                env.controller
            )
        
        # 清理播放列表数据库
        for table_name in ["Playlist", "PlaylistEntity", "SongEntity"]:
            try:
                if sqlite_utils.table_exists(table_name, retro_music_db, env):
                    sqlite_utils.delete_all_rows_from_table(
                        table_name, retro_music_db, env, "retro music"
                    )
            except Exception as e:
                logger.debug(f"清理播放列表表 {table_name} 时出现警告: {e}")
        
        logger.info("音乐数据已清空")
        return True
    except Exception as e:
        logger.warning(f"清空音乐数据失败: {e}")
        return False


def clear_app_with_pm(env: interface.AsyncEnv, package_name: str) -> bool:
    """使用 pm clear 命令清空应用数据。"""
    logger.info(f"使用 pm clear 清空应用数据: {package_name}")
    try:
        # 先停止应用
        adb_utils.close_app(PACKAGE_TO_APP_NAME.get(package_name, package_name), env.controller)
        
        # 执行 pm clear
        adb_utils.issue_generic_request(
            ['shell', 'pm', 'clear', package_name],
            env.controller
        )
        logger.info(f"应用数据已清空: {package_name}")
        return True
    except Exception as e:
        logger.warning(f"清空应用数据失败: {package_name}, {e}")
        return False


def clear_app_data(env: interface.AsyncEnv, package_name: str) -> bool:
    """
    清空指定应用的数据。
    
    使用 AndroidWorld 原生工具根据应用类型选择最佳清理方法。
    
    Args:
        env: AndroidWorld 环境对象
        package_name: 应用包名
        
    Returns:
        是否成功
    """
    # 使用原生清理函数的应用映射
    # 与 reference/MobileForge Explore/parallel_exploration/app_data_cleaner.py 对齐
    native_cleaners = {
        "com.google.android.contacts": lambda: clear_contacts_data(env),
        "com.google.android.dialer": lambda: (
            clear_contacts_data(env),
            clear_sms_data(env),
            clear_call_history_data(env),
        ),
        # Simple SMS Messenger: 使用 pm clear 完全清空应用数据，
        # 以便 setup 流程能够重新设置为默认 SMS 应用
        "com.simplemobiletools.smsmessenger": lambda: clear_sms_app_data(env),
        "com.simplemobiletools.calendar.pro": lambda: clear_calendar_data(env),
        "org.tasks": lambda: clear_tasks_data(env),
        "net.cozic.joplin": lambda: clear_joplin_data(env),
        "de.dennisguse.opentracks": lambda: clear_opentracks_data(env),
        "com.arduia.expense": lambda: clear_expense_data(env),
        "org.videolan.vlc": lambda: clear_vlc_data(env),
        # 以下为新增的清理方法，与 Explore 的 AppDataCleaner 对齐
        "com.simplemobiletools.gallery.pro": lambda: clear_gallery_data(env),
        "com.google.android.documentsui": lambda: clear_files_data(env),
        "com.flauschcode.broccoli": lambda: clear_recipes_data(env),
        "net.gsantner.markor": lambda: clear_markor_data(env),
        "code.name.monkey.retromusic": lambda: clear_music_data(env),
    }
    
    # 跳过不需要清理的应用
    skip_packages = {
        "com.android.chrome",
        "com.android.settings",
    }
    
    if package_name in skip_packages:
        logger.debug(f"跳过清理: {package_name}")
        return True
    
    # 优先使用原生清理函数
    if package_name in native_cleaners:
        try:
            return native_cleaners[package_name]()
        except Exception as e:
            logger.warning(f"原生清理失败: {package_name}, {e}")
            # 回退到 pm clear
            return clear_app_with_pm(env, package_name)
    
    # 对于其他应用，使用 pm clear
    return clear_app_with_pm(env, package_name)


def inject_app_data(env: interface.AsyncEnv, package_name: str) -> bool:
    """
    为指定应用注入数据（仅注入单个应用）。
    
    注意：此函数仅注入单个应用的数据，不保证与参考实现的随机种子状态一致。
    如需确保数据与参考实现完全一致，请使用 inject_all_app_data() 函数。
    
    Args:
        env: AndroidWorld 环境对象
        package_name: 应用包名
        
    Returns:
        是否成功
    """
    from .native_app_injector import AppDataInjector
    
    try:
        injector = AppDataInjector(env)
        return injector.inject_data_for_package(package_name)
    except Exception as e:
        logger.error(f"注入应用数据失败: {package_name}, {e}")
        return False


def inject_all_app_data(env: interface.AsyncEnv, config: dict = None) -> dict:
    """
    按照参考实现的固定顺序注入所有应用数据。
    
    此函数确保：
    1. 随机种子在开头设置为42
    2. 按与参考实现完全一致的顺序调用所有注入函数
    3. 每次调用生成的数据完全相同
    
    Args:
        env: AndroidWorld 环境对象
        config: 可选配置
        
    Returns:
        每个数据类型的注入结果字典
    """
    from .native_app_injector import AppDataInjector
    
    try:
        injector = AppDataInjector(env, config)
        return injector.inject_all_data()
    except Exception as e:
        logger.error(f"全量注入应用数据失败: {e}")
        return {}


def reset_app_for_task(env: interface.AsyncEnv, package_name: str, inject_data: bool = True, use_full_injection: bool = False) -> bool:
    """
    为任务重置应用数据。
    
    使用 AndroidWorld 原生工具清空并注入数据。
    对于需要首次初始化的应用，在清空数据后会调用 setup.setup_app() 
    来重新执行首次初始化流程（如点击引导界面），确保数据库被正确创建。
    
    Args:
        env: AndroidWorld 环境对象
        package_name: 应用包名
        inject_data: 是否注入数据
        use_full_injection: 是否使用全量注入（按参考实现顺序注入所有应用数据，确保数据一致性）
        
    Returns:
        是否成功
    """
    import time
    
    logger.info(f"为任务重置应用: {package_name}")
    
    # 1. 清空应用数据
    if not clear_app_data(env, package_name):
        logger.warning(f"清空应用数据失败: {package_name}")
        # 继续执行，某些清除失败不应阻止任务执行
    
    # 2. 如果应用需要首次初始化，执行 setup.setup_app()
    # 这会完成应用的首次启动流程（如点击引导界面），创建数据库文件
    if package_name in APPS_REQUIRING_SETUP:
        app_class = APPS_REQUIRING_SETUP[package_name]
        logger.info(f"执行应用首次初始化: {package_name} ({app_class.app_name})")
        try:
            # 确保回到主屏幕，避免快速设置菜单阻挡UI导航
            adb_utils.press_home_button(env.controller)
            # 执行应用首次初始化（包括点击引导界面，创建数据库）
            setup.setup_app(app_class, env)
            logger.info(f"应用首次初始化完成: {package_name}")
        except Exception as e:
            logger.warning(f"应用首次初始化失败: {package_name}, {e}")
            # 继续执行，尝试注入数据
    
    # 3. 注入数据
    if inject_data:
        if use_full_injection:
            # 使用全量注入，按参考实现顺序注入所有应用数据
            logger.info("使用全量注入模式...")
            results = inject_all_app_data(env)
            if not results:
                logger.warning("全量注入失败")
        else:
            # 仅注入单个应用的数据
            if not inject_app_data(env, package_name):
                logger.warning(f"注入应用数据失败: {package_name}")
    
    # 4. 等待应用稳定
    time.sleep(1)
    
    logger.info(f"应用重置完成: {package_name}")
    return True


def reset_apps_for_task_with_full_injection(env: interface.AsyncEnv, packages: list, config: dict = None) -> dict:
    """
    为任务重置多个应用，并使用全量注入确保数据与参考实现一致。
    
    此函数按以下步骤执行：
    1. 清空所有指定应用的数据
    2. 执行需要首次初始化的应用的 setup
    3. 按参考实现顺序全量注入所有应用数据（仅执行一次）
    
    这确保了随机种子消耗顺序与参考实现一致，从而保证数据一致性。
    
    Args:
        env: AndroidWorld 环境对象
        packages: 需要重置的应用包名列表
        config: 可选配置
        
    Returns:
        每个应用的重置结果字典
    """
    import time
    
    logger.info(f"使用全量注入模式重置 {len(packages)} 个应用...")
    results = {}
    
    # 1. 清空所有应用数据
    for package_name in packages:
        logger.info(f"清空应用数据: {package_name}")
        if not clear_app_data(env, package_name):
            logger.warning(f"清空应用数据失败: {package_name}")
            results[package_name] = False
        else:
            results[package_name] = True
    
    # 2. 执行需要首次初始化的应用的 setup
    for package_name in packages:
        if package_name in APPS_REQUIRING_SETUP:
            app_class = APPS_REQUIRING_SETUP[package_name]
            logger.info(f"执行应用首次初始化: {package_name} ({app_class.app_name})")
            try:
                adb_utils.press_home_button(env.controller)
                setup.setup_app(app_class, env)
                logger.info(f"应用首次初始化完成: {package_name}")
            except Exception as e:
                logger.warning(f"应用首次初始化失败: {package_name}, {e}")
    
    # 3. 全量注入所有应用数据（仅执行一次，确保随机种子消耗顺序一致）
    logger.info("执行全量数据注入（按参考实现顺序）...")
    injection_results = inject_all_app_data(env, config)
    
    if injection_results:
        logger.info(f"全量注入完成，成功注入 {sum(1 for v in injection_results.values() if v)}/{len(injection_results)} 个数据类型")
    else:
        logger.warning("全量注入失败")
    
    # 4. 等待稳定
    time.sleep(1)
    
    logger.info(f"应用重置完成（全量注入模式）")
    return results


def get_task_app_packages(task: Any) -> List[str]:
    """
    从任务对象中获取相关的应用包名列表。
    
    Args:
        task: 任务对象
        
    Returns:
        应用包名列表
    """
    packages = []
    
    # 尝试从 app_package 字段获取
    if hasattr(task, 'app_package'):
        app_pkg = getattr(task, 'app_package', None)
        if app_pkg:
            if isinstance(app_pkg, (list, tuple)):
                packages.extend(app_pkg)
            else:
                packages.append(str(app_pkg))
    
    # 尝试从任务名称或描述推断
    if not packages and hasattr(task, 'task_identifier'):
        task_id = getattr(task, 'task_identifier', '')
        inferred = _infer_package_from_task_id(task_id)
        if inferred:
            packages.append(inferred)
    
    return packages


def _infer_package_from_task_id(task_id: str) -> Optional[str]:
    """从任务 ID 推断应用包名。"""
    task_id_lower = task_id.lower()
    
    # 任务 ID 到包名的映射
    task_patterns = {
        'calendar': 'com.simplemobiletools.calendar.pro',
        'tasks': 'org.tasks',
        'joplin': 'net.cozic.joplin',
        'markor': 'net.gsantner.markor',
        'broccoli': 'com.flauschcode.broccoli',
        'recipe': 'com.flauschcode.broccoli',
        'expense': 'com.arduia.expense',
        'opentracks': 'de.dennisguse.opentracks',
        'activity': 'de.dennisguse.opentracks',
        'contacts': 'com.google.android.contacts',
        'dialer': 'com.google.android.dialer',
        'phone': 'com.google.android.dialer',
        'sms': 'com.simplemobiletools.smsmessenger',
        'message': 'com.simplemobiletools.smsmessenger',
        'clock': 'com.google.android.deskclock',
        'alarm': 'com.google.android.deskclock',
        'timer': 'com.google.android.deskclock',
        'gallery': 'com.simplemobiletools.gallery.pro',
        'photo': 'com.simplemobiletools.gallery.pro',
        'draw': 'com.simplemobiletools.draw.pro',
        'vlc': 'org.videolan.vlc',
        'video': 'org.videolan.vlc',
        'music': 'code.name.monkey.retromusic',
        'retromusic': 'code.name.monkey.retromusic',
        'camera': 'com.android.camera2',
        'chrome': 'com.android.chrome',
        'browser': 'com.android.chrome',
        'files': 'com.google.android.documentsui',
        'audio': 'com.dimowner.audiorecorder',
        'record': 'com.dimowner.audiorecorder',
        'osmand': 'net.osmand',
        'map': 'net.osmand',
    }
    
    for pattern, package in task_patterns.items():
        if pattern in task_id_lower:
            return package
    
    return None


class AppDataManager:
    """
    应用数据管理器类。
    
    封装了应用数据的清理和注入功能。
    支持两种注入模式：
    1. 单应用注入：仅注入指定应用的数据（效率高，但可能与参考实现数据不一致）
    2. 全量注入：按参考实现顺序注入所有应用数据（确保数据一致性，但耗时较长）
    """
    
    def __init__(self, env: interface.AsyncEnv, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据管理器。
        
        Args:
            env: AndroidWorld 环境对象
            config: 可选配置
        """
        self.env = env
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def clear_app_data(self, package_name: str) -> bool:
        """清空应用数据。"""
        return clear_app_data(self.env, package_name)
    
    def inject_app_data(self, package_name: str) -> bool:
        """注入单个应用数据。"""
        return inject_app_data(self.env, package_name)
    
    def inject_all_app_data(self) -> dict:
        """
        按参考实现顺序注入所有应用数据（全量注入）。
        
        此方法确保随机种子消耗顺序与参考实现一致，从而保证数据一致性。
        
        Returns:
            每个数据类型的注入结果字典
        """
        return inject_all_app_data(self.env, self.config)
    
    def reset_app_for_task(self, package_name: str, inject_data: bool = True, use_full_injection: bool = False) -> bool:
        """
        重置单个应用数据。
        
        Args:
            package_name: 应用包名
            inject_data: 是否注入数据
            use_full_injection: 是否使用全量注入模式
        """
        return reset_app_for_task(self.env, package_name, inject_data, use_full_injection)
    
    def reset_apps_for_task(self, packages: List[str], inject_data: bool = True, use_full_injection: bool = False) -> Dict[str, bool]:
        """
        重置多个应用的数据。
        
        Args:
            packages: 应用包名列表
            inject_data: 是否注入数据
            use_full_injection: 是否使用全量注入模式（推荐，确保数据与参考实现一致）
        """
        if use_full_injection and inject_data:
            # 使用全量注入模式
            return reset_apps_for_task_with_full_injection(self.env, packages, self.config)
        else:
            # 逐个注入模式
            results = {}
            for package in packages:
                results[package] = self.reset_app_for_task(package, inject_data, use_full_injection=False)
            return results
    
    def reset_apps_with_full_injection(self, packages: List[str]) -> Dict[str, bool]:
        """
        使用全量注入模式重置多个应用（推荐方法）。
        
        此方法确保：
        1. 清空所有指定应用的数据
        2. 按参考实现顺序全量注入所有应用数据
        3. 随机种子消耗顺序与参考实现一致
        
        Args:
            packages: 需要重置的应用包名列表
            
        Returns:
            每个应用的重置结果字典
        """
        return reset_apps_for_task_with_full_injection(self.env, packages, self.config)


def get_app_package(app_name: str) -> Optional[str]:
    """获取应用的包名。"""
    return APP_PACKAGES.get(app_name.lower())


def get_app_name(package_name: str) -> Optional[str]:
    """获取包名对应的应用名称。"""
    return PACKAGE_TO_APP_NAME.get(package_name)
