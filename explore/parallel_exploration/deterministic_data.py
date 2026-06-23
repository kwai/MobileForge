# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Deterministic data module for parallel exploration setup.

This module provides fixed, deterministic data for all app data injection.
All data is sourced directly from AndroidWorld's original data definitions
to ensure consistency with the task evaluation system.

Key features:
- No random sampling - all data is injected in full
- Fixed ordering for reproducibility
- Direct reference to AndroidWorld's original data structures
- Additional data for contacts, SMS, call history, gallery, and files
"""

import datetime
from typing import Dict, List, Any, Optional

# 尝试导入 android_world 模块，如果不可用则使用本地备份数据
try:
    from android_world.env import device_constants
    from android_world.task_evals.single import recipe as recipe_module
    from android_world.task_evals.information_retrieval import task_app_utils
    from android_world.task_evals.information_retrieval import joplin_app_utils
    from android_world.task_evals.information_retrieval import activity_app_utils
    from android_world.task_evals.utils import sqlite_schema_utils
    ANDROID_WORLD_AVAILABLE = True
except ImportError:
    ANDROID_WORLD_AVAILABLE = False
    # 定义基准时间
    class device_constants:
        DT = datetime.datetime(2023, 10, 15, 10, 0, 0)


# ============================================================================
# RECIPE DATA (Broccoli App) - 39 固定食谱
# ============================================================================

def get_all_recipes() -> List[Any]:
    """获取所有39个固定食谱数据。
    
    数据源: android_world/task_evals/single/recipe.py 的 _RECIPES 列表
    
    Returns:
        包含所有39个食谱的列表，每个食谱都有完整的字段填充
    """
    if not ANDROID_WORLD_AVAILABLE:
        return []
    
    recipes = []
    
    # 固定的描述、配料说明和准备时间，确保确定性
    descriptions = [
        'A quick and easy meal, perfect for busy weekdays.',
        'A delicious and healthy choice for any time of the day.',
        'An ideal recipe for experimenting with different flavors and ingredients.',
    ]
    
    servings_options = ['1 serving', '2 servings', '3-4 servings', '6 servings', '8 servings']
    prep_time_options = ['10 mins', '20 mins', '30 mins', '45 mins', '1 hrs', '2 hrs']
    
    ingredient_descriptors = [
        'see directions', 'as per recipe', 'varies', 'to preference',
        'quantities to taste', 'as needed', 'optional ingredients',
    ]
    
    directions_additions = [
        'Try adding a pinch of your favorite spices for extra flavor.',
        'Feel free to substitute with ingredients you have on hand.',
        'Garnish with fresh herbs for a more vibrant taste.',
    ]
    
    # 遍历所有原始食谱，添加固定的附加字段
    for i, base_recipe in enumerate(recipe_module._RECIPES):
        # 使用确定性索引选择附加属性
        desc_idx = i % len(descriptions)
        serv_idx = i % len(servings_options)
        prep_idx = i % len(prep_time_options)
        ingr_idx = i % len(ingredient_descriptors)
        dir_idx = i % len(directions_additions)
        
        recipe = sqlite_schema_utils.Recipe(
            title=base_recipe.title,
            description=descriptions[desc_idx],
            servings=servings_options[serv_idx],
            preparationTime=prep_time_options[prep_idx],
            ingredients=ingredient_descriptors[ingr_idx],
            directions=f'{base_recipe.directions} {directions_additions[dir_idx]}',
        )
        recipes.append(recipe)
    
    return recipes


# ============================================================================
# TASKS DATA (Tasks App) - 20 固定任务
# ============================================================================

def get_all_tasks() -> List[Dict[str, str]]:
    """获取所有20个固定任务数据。
    
    数据源: android_world/task_evals/information_retrieval/task_app_utils.py 的 _TASKS 字典
    
    Returns:
        包含所有20个任务的列表，每个任务包含 title 和 notes
    """
    if not ANDROID_WORLD_AVAILABLE:
        return []
    
    tasks = []
    for title, notes in task_app_utils._TASKS.items():
        tasks.append({
            'title': title,
            'notes': notes
        })
    return tasks


def create_deterministic_tasks(base_timestamp_ms: int = None) -> List[Any]:
    """创建确定性的任务数据对象。
    
    Args:
        base_timestamp_ms: 基准时间戳（毫秒），默认使用 device_constants.DT
        
    Returns:
        sqlite_schema_utils.Task 对象列表
    """
    if not ANDROID_WORLD_AVAILABLE:
        return []
    
    import uuid
    
    if base_timestamp_ms is None:
        # 使用 AndroidWorld 的标准时间: 2023-10-15
        base_dt = device_constants.DT
        base_timestamp_ms = int(base_dt.timestamp() * 1000)
    
    tasks = []
    task_data = get_all_tasks()
    
    # 固定的重要性级别分配
    importance_pattern = [0, 1, 2, 3, 2, 1, 0, 3, 2, 1]
    
    for i, task_info in enumerate(task_data):
        # 确定性时间戳：基于索引计算
        days_offset = (i % 7) + 1  # 1-7天前创建
        hours_offset = (i % 12) + 8  # 8-19点
        
        created_ts = base_timestamp_ms - (days_offset * 24 * 3600 * 1000) - (hours_offset * 3600 * 1000)
        modified_ts = created_ts + (i * 3600 * 1000)  # 每个任务修改时间递增1小时
        
        # 前6个任务标记为完成 (30%)
        completed_ts = 0
        if i < 6:
            completed_ts = created_ts + (2 * 24 * 3600 * 1000)  # 创建后2天完成
        
        # 确定性的due date：基于索引
        due_offset_days = (i % 14) - 3  # -3 到 +10 天
        due_ts = base_timestamp_ms + (due_offset_days * 24 * 3600 * 1000)
        
        task = sqlite_schema_utils.Task(
            title=task_info['title'],
            importance=importance_pattern[i % len(importance_pattern)],
            dueDate=due_ts,
            hideUntil=0,  # 不隐藏
            completed=completed_ts,
            created=created_ts,
            modified=modified_ts,
            notes=task_info['notes'],
            remoteId=str(uuid.UUID(int=i + 1000).int),  # 确定性UUID
            recurrence=None,
        )
        tasks.append(task)
    
    return tasks


# ============================================================================
# JOPLIN DATA - 12个文件夹，300+ 笔记
# ============================================================================

def get_all_joplin_folders() -> List[str]:
    """获取所有12个 Joplin 文件夹名称。
    
    数据源: android_world/task_evals/information_retrieval/joplin_app_utils.py 的 _FOLDERS
    
    Returns:
        文件夹名称列表
    """
    if not ANDROID_WORLD_AVAILABLE:
        return []
    return list(joplin_app_utils._FOLDERS.keys())


def get_all_joplin_notes() -> Dict[str, List[Dict[str, str]]]:
    """获取所有 Joplin 笔记数据，按文件夹分组。
    
    数据源: android_world/task_evals/information_retrieval/joplin_app_utils.py 的 _FOLDERS
    
    Returns:
        字典，键为文件夹名，值为该文件夹下所有笔记的列表
    """
    if not ANDROID_WORLD_AVAILABLE:
        return {}
    return joplin_app_utils._FOLDERS.copy()


def get_total_joplin_notes_count() -> int:
    """获取 Joplin 笔记总数。"""
    if not ANDROID_WORLD_AVAILABLE:
        return 0
    total = 0
    for folder_notes in joplin_app_utils._FOLDERS.values():
        total += len(folder_notes)
    return total


# ============================================================================
# OPENTRACKS DATA - 16个运动类别
# ============================================================================

def get_all_activity_categories() -> List[str]:
    """获取所有16个运动类别。
    
    数据源: android_world/task_evals/information_retrieval/activity_app_utils.py
    
    Returns:
        运动类别名称列表
    """
    if not ANDROID_WORLD_AVAILABLE:
        return []
    return list(activity_app_utils._CATEGORY_TO_ACTIVITY_NAMES.keys())


def get_activity_names_by_category() -> Dict[str, List[str]]:
    """获取每个运动类别下的所有活动名称。
    
    Returns:
        字典，键为类别名，值为该类别下所有活动名称列表
    """
    if not ANDROID_WORLD_AVAILABLE:
        return {}
    return activity_app_utils._CATEGORY_TO_ACTIVITY_NAMES.copy()


def create_deterministic_activities(base_timestamp_ms: int = None) -> List[Any]:
    """创建确定性的运动活动数据。
    
    为每个运动类别创建固定数量的活动记录。
    
    Args:
        base_timestamp_ms: 基准时间戳（毫秒）
        
    Returns:
        SportsActivity 对象列表
    """
    if not ANDROID_WORLD_AVAILABLE:
        return []
    
    if base_timestamp_ms is None:
        base_dt = device_constants.DT
        base_timestamp_ms = int(base_dt.timestamp() * 1000)
    
    activities = []
    categories = get_all_activity_categories()
    activity_names = get_activity_names_by_category()
    
    # 为每个类别创建活动
    activity_index = 0
    for cat_idx, category in enumerate(categories):
        names = activity_names[category]
        # 每个类别创建2-3个活动
        num_activities = 2 + (cat_idx % 2)
        
        for name_idx in range(min(num_activities, len(names))):
            name = names[name_idx]
            
            # 确定性时间：基于索引
            days_ago = activity_index + 1  # 1, 2, 3... 天前
            start_hour = 6 + (activity_index % 12)  # 6-17点开始
            
            start_ts = base_timestamp_ms - (days_ago * 24 * 3600 * 1000)
            start_ts += start_hour * 3600 * 1000
            
            # 确定性持续时间：30-180分钟
            duration_minutes = 30 + (activity_index * 10) % 150
            duration_ms = duration_minutes * 60 * 1000
            stop_ts = start_ts + duration_ms
            
            # 确定性距离：1000-15000米
            distance = 1000 + (activity_index * 500) % 14000
            
            # 确定性海拔
            elevation_gain = (activity_index * 20) % 500
            elevation_loss = (activity_index * 15) % 400
            
            # 计算速度
            avg_speed = distance / (duration_ms / 1000) if duration_ms > 0 else 0
            
            activity = sqlite_schema_utils.SportsActivity(
                name=name,
                category=category,
                activity_type=category,
                description=f'Deterministic {category} activity #{activity_index + 1}',
                totaldistance=float(distance),
                starttime=start_ts,
                stoptime=stop_ts,
                totaltime=duration_ms,
                movingtime=duration_ms,
                avgspeed=avg_speed,
                avgmovingspeed=avg_speed,
                elevationgain=elevation_gain,
                elevationloss=elevation_loss,
            )
            activities.append(activity)
            activity_index += 1
    
    return activities


# ============================================================================
# CALENDAR DATA - 固定日历事件
# ============================================================================

def create_deterministic_calendar_events(base_timestamp: int = None) -> List[Any]:
    """创建确定性的日历事件数据。
    
    Args:
        base_timestamp: 基准Unix时间戳（秒）
        
    Returns:
        CalendarEvent 对象列表
    """
    if not ANDROID_WORLD_AVAILABLE:
        return []
    
    if base_timestamp is None:
        base_dt = device_constants.DT
        base_timestamp = int(base_dt.timestamp())
    
    events = []
    
    # 固定的事件模板
    event_templates = [
        ('Team Meeting', 'meeting', 60, 'Conference Room A'),
        ('Project Review', 'meeting', 90, 'Meeting Room B'),
        ('Client Call', 'meeting', 30, 'Phone'),
        ('Doctor Appointment', 'appointment', 45, 'Medical Center'),
        ('Dentist Checkup', 'appointment', 60, 'Dental Clinic'),
        ('Birthday Party', 'birthday', 1440, 'Home'),  # 全天
        ('Anniversary', 'birthday', 1440, ''),  # 全天
        ('Pay Bills', 'reminder', 0, ''),
        ('Submit Report', 'reminder', 0, ''),
        ('Gym Session', 'task', 90, 'Fitness Center'),
        ('Weekly Review', 'meeting', 60, 'Office'),
        ('Sprint Planning', 'meeting', 120, 'Conference Room'),
        ('Code Review', 'meeting', 45, 'Virtual'),
        ('Lunch Meeting', 'meeting', 60, 'Restaurant'),
        ('Training Session', 'meeting', 180, 'Training Room'),
        ('Interview', 'appointment', 60, 'HR Office'),
        ('Car Service', 'appointment', 120, 'Auto Shop'),
        ('Mom Birthday', 'birthday', 1440, ''),
        ('Project Deadline', 'reminder', 0, ''),
        ('Team Building', 'meeting', 240, 'Event Hall'),
        ('Quarterly Review', 'meeting', 90, 'Board Room'),
        ('Standup Meeting', 'meeting', 15, 'Virtual'),
        ('Design Review', 'meeting', 60, 'Design Lab'),
        ('Performance Review', 'appointment', 45, 'Manager Office'),
        ('Yoga Class', 'task', 60, 'Yoga Studio'),
    ]
    
    for i, (title, event_type, duration_minutes, location) in enumerate(event_templates):
        # 确定性时间：基于索引分布在前后两周
        day_offset = (i % 21) - 7  # -7 到 +13 天
        hour = 8 + (i % 10)  # 8-17点
        
        start_ts = base_timestamp + (day_offset * 24 * 3600) + (hour * 3600)
        
        if duration_minutes == 1440:  # 全天事件
            end_ts = start_ts + duration_minutes * 60
        else:
            end_ts = start_ts + duration_minutes * 60
        
        event = sqlite_schema_utils.CalendarEvent(
            start_ts=start_ts,
            end_ts=end_ts,
            title=title,
            location=location,
            description=f'Deterministic {event_type} event #{i + 1}',
            repeat_interval=0,
            repeat_rule=0,
            reminder_1_minutes=-1,
            reminder_2_minutes=-1,
            reminder_3_minutes=-1,
            reminder_1_type=0,
            reminder_2_type=0,
            reminder_3_type=0,
            repeat_limit=0,
            repetition_exceptions='[]',
            attendees='',
            import_id='',
            flags=0,
            event_type=1,
            parent_id=0,
            last_updated=0,
            source='comprehensive-setup',
            availability=0,
            color=0,
            type=0
        )
        events.append(event)
    
    return events


# ============================================================================
# EXPENSE DATA (Pro Expense) - 固定费用记录
# ============================================================================

def create_deterministic_expenses(base_timestamp_ms: int = None) -> List[Any]:
    """创建确定性的费用记录数据。
    
    与 reference/MobileForge Rollout/framework/deterministic_data.py 完全一致，
    返回 sqlite_schema_utils.Expense 对象列表，以确保与 sqlite_utils.insert_rows_to_remote_db 兼容。
    
    Category ID 映射:
        1: Food, 2: Transport, 3: Entertainment, 4: Health, 5: Bills,
        6: Shopping, 7: Education, 8: Other, 9: Savings
    
    Args:
        base_timestamp_ms: 基准时间戳（毫秒）
        
    Returns:
        sqlite_schema_utils.Expense 对象列表（当 android_world 可用时）
        或字典列表（当 android_world 不可用时作为回退）
    """
    if base_timestamp_ms is None:
        base_dt = device_constants.DT
        base_timestamp_ms = int(base_dt.timestamp() * 1000)
    
    expenses = []
    
    # 固定的费用模板（与 reference 完全一致）
    expense_templates = [
        ('Groceries', 1, 45.99),
        ('Gas', 2, 52.30),
        ('Restaurant', 1, 28.50),
        ('Coffee', 1, 5.25),
        ('Movie Tickets', 3, 24.00),
        ('Gym Membership', 4, 49.99),
        ('Phone Bill', 5, 85.00),
        ('Internet', 5, 65.00),
        ('Electricity', 5, 120.50),
        ('Water Bill', 5, 35.00),
        ('Uber', 2, 18.75),
        ('Amazon', 6, 156.99),
        ('Netflix', 3, 15.99),
        ('Spotify', 3, 9.99),
        ('Lunch', 1, 12.50),
        ('Dinner', 1, 35.00),
        ('Books', 7, 29.99),
        ('Clothing', 6, 89.99),
        ('Medicine', 4, 22.50),
        ('Haircut', 8, 25.00),
        ('Parking', 2, 8.00),
        ('Subway', 2, 2.75),
        ('Snacks', 1, 6.50),
        ('Office Supplies', 7, 34.99),
        ('Gift', 8, 50.00),
        ('Dry Cleaning', 8, 18.00),
        ('Pet Food', 8, 42.00),
        ('Insurance', 5, 150.00),
        ('Rent', 5, 1500.00),
        ('Savings', 9, 500.00),
    ]
    
    for i, (name, category_id, amount) in enumerate(expense_templates):
        # 确定性时间：过去30天内
        days_ago = i % 30
        
        created_date = base_timestamp_ms - (days_ago * 24 * 3600 * 1000)
        modified_date = created_date
        
        if ANDROID_WORLD_AVAILABLE:
            # 返回 sqlite_schema_utils.Expense 对象（与 reference 一致）
            expense = sqlite_schema_utils.Expense(
                name=name,
                amount=int(amount * 100),  # 以分为单位
                category=category_id,
                note=f'Deterministic expense #{i + 1}',
                created_date=created_date,
                modified_date=modified_date,
            )
        else:
            # 回退方式：返回字典
            expense = {
                'name': name,
                'amount': int(amount * 100),  # 以分为单位
                'category': category_id,
                'note': f'Deterministic expense #{i + 1}',
                'created_date': created_date,
                'modified_date': modified_date,
            }
        expenses.append(expense)
    
    return expenses


# ============================================================================
# CONTACTS DATA - 固定联系人
# ============================================================================

# 固定的联系人数据
DETERMINISTIC_CONTACTS = [
    {'name': 'Alice Johnson', 'phone': '+1-555-0101'},
    {'name': 'Bob Smith', 'phone': '+1-555-0102'},
    {'name': 'Carol Williams', 'phone': '+1-555-0103'},
    {'name': 'David Brown', 'phone': '+1-555-0104'},
    {'name': 'Emily Davis', 'phone': '+1-555-0105'},
    {'name': 'Frank Miller', 'phone': '+1-555-0106'},
    {'name': 'Grace Wilson', 'phone': '+1-555-0107'},
    {'name': 'Henry Moore', 'phone': '+1-555-0108'},
    {'name': 'Ivy Taylor', 'phone': '+1-555-0109'},
    {'name': 'Jack Anderson', 'phone': '+1-555-0110'},
    {'name': 'Kate Thomas', 'phone': '+1-555-0111'},
    {'name': 'Leo Jackson', 'phone': '+1-555-0112'},
    {'name': 'Mia White', 'phone': '+1-555-0113'},
    {'name': 'Noah Harris', 'phone': '+1-555-0114'},
    {'name': 'Olivia Martin', 'phone': '+1-555-0115'},
    {'name': 'Peter Garcia', 'phone': '+1-555-0116'},
    {'name': 'Quinn Martinez', 'phone': '+1-555-0117'},
    {'name': 'Rachel Robinson', 'phone': '+1-555-0118'},
    {'name': 'Sam Clark', 'phone': '+1-555-0119'},
    {'name': 'Tina Rodriguez', 'phone': '+1-555-0120'},
    {'name': 'Uma Lewis', 'phone': '+1-555-0121'},
    {'name': 'Victor Lee', 'phone': '+1-555-0122'},
    {'name': 'Wendy Walker', 'phone': '+1-555-0123'},
    {'name': 'Xavier Hall', 'phone': '+1-555-0124'},
    {'name': 'Yolanda Allen', 'phone': '+1-555-0125'},
    {'name': 'Zack Young', 'phone': '+1-555-0126'},
    {'name': 'Amy King', 'phone': '+1-555-0127'},
    {'name': 'Brian Wright', 'phone': '+1-555-0128'},
    {'name': 'Cindy Scott', 'phone': '+1-555-0129'},
    {'name': 'Daniel Green', 'phone': '+1-555-0130'},
    {'name': 'Eva Adams', 'phone': '+1-555-0131'},
    {'name': 'Fred Baker', 'phone': '+1-555-0132'},
    {'name': 'Gina Nelson', 'phone': '+1-555-0133'},
    {'name': 'Howard Hill', 'phone': '+1-555-0134'},
    {'name': 'Iris Ramirez', 'phone': '+1-555-0135'},
    {'name': 'James Campbell', 'phone': '+1-555-0136'},
    {'name': 'Kelly Mitchell', 'phone': '+1-555-0137'},
    {'name': 'Larry Roberts', 'phone': '+1-555-0138'},
    {'name': 'Mary Carter', 'phone': '+1-555-0139'},
    {'name': 'Nick Phillips', 'phone': '+1-555-0140'},
    {'name': 'Oscar Evans', 'phone': '+1-555-0141'},
    {'name': 'Paula Turner', 'phone': '+1-555-0142'},
    {'name': 'Quentin Torres', 'phone': '+1-555-0143'},
    {'name': 'Rita Parker', 'phone': '+1-555-0144'},
    {'name': 'Steve Collins', 'phone': '+1-555-0145'},
    {'name': 'Tracy Edwards', 'phone': '+1-555-0146'},
    {'name': 'Ulysses Stewart', 'phone': '+1-555-0147'},
    {'name': 'Vera Sanchez', 'phone': '+1-555-0148'},
    {'name': 'Will Morris', 'phone': '+1-555-0149'},
    {'name': 'Xena Rogers', 'phone': '+1-555-0150'},
]


def get_all_contacts() -> List[Dict[str, str]]:
    """获取所有固定联系人数据。
    
    Returns:
        联系人字典列表
    """
    return DETERMINISTIC_CONTACTS.copy()


# ============================================================================
# SMS DATA - 固定短信对话
# ============================================================================

def get_deterministic_sms_conversations(base_timestamp_ms: int = None) -> List[Dict[str, Any]]:
    """获取确定性的短信对话数据。
    
    Args:
        base_timestamp_ms: 基准时间戳（毫秒）
        
    Returns:
        短信对话列表
    """
    if base_timestamp_ms is None:
        base_dt = device_constants.DT
        base_timestamp_ms = int(base_dt.timestamp() * 1000)
    
    conversations = []
    
    # 使用前10个联系人创建对话
    sms_templates = [
        ("Hey, how are you?", True),
        ("I'm good, thanks! How about you?", False),
        ("Great! Want to grab lunch tomorrow?", True),
        ("Sure, sounds good. Where?", False),
        ("How about the Italian place on Main St?", True),
        ("Perfect! See you at noon.", False),
        ("Don't forget about the meeting today", True),
        ("Thanks for the reminder!", False),
        ("Can you send me the report?", True),
        ("Sending it now.", False),
        ("Got it, thanks!", True),
        ("Happy birthday!", True),
        ("Thank you so much!", False),
        ("Are you free this weekend?", True),
        ("Yes, what did you have in mind?", False),
    ]
    
    for i, contact in enumerate(DETERMINISTIC_CONTACTS[:10]):
        # 每个联系人创建3-5条消息
        num_messages = 3 + (i % 3)
        for j in range(num_messages):
            msg_idx = (i * 3 + j) % len(sms_templates)
            message, is_incoming = sms_templates[msg_idx]
            
            # 确定性时间戳
            days_ago = (i + j) % 14
            hours_ago = (i * 2 + j) % 24
            timestamp = base_timestamp_ms - (days_ago * 24 * 3600 * 1000) - (hours_ago * 3600 * 1000)
            
            conversations.append({
                'address': contact['phone'],
                'contact_name': contact['name'],
                'body': message,
                'type': 1 if is_incoming else 2,  # 1=incoming, 2=outgoing
                'date': timestamp,
                'read': 1,
            })
    
    return conversations


# ============================================================================
# CALL HISTORY DATA - 固定通话记录
# ============================================================================

def get_deterministic_call_history(base_timestamp_ms: int = None) -> List[Dict[str, Any]]:
    """获取确定性的通话记录数据。
    
    Args:
        base_timestamp_ms: 基准时间戳（毫秒）
        
    Returns:
        通话记录列表
    """
    if base_timestamp_ms is None:
        base_dt = device_constants.DT
        base_timestamp_ms = int(base_dt.timestamp() * 1000)
    
    call_history = []
    
    # 通话类型: 1=incoming, 2=outgoing, 3=missed
    call_types = [1, 2, 1, 3, 2, 1, 2, 3, 1, 2]
    durations = [120, 300, 60, 0, 180, 240, 90, 0, 150, 420]  # 未接电话时长为0
    
    for i, contact in enumerate(DETERMINISTIC_CONTACTS[:15]):
        # 每个联系人1-2条通话记录
        num_calls = 1 + (i % 2)
        for j in range(num_calls):
            call_idx = (i + j) % len(call_types)
            call_type = call_types[call_idx]
            duration = durations[call_idx] if call_type != 3 else 0
            
            # 确定性时间戳
            days_ago = (i * 2 + j) % 21
            hours_ago = (i + j * 3) % 24
            timestamp = base_timestamp_ms - (days_ago * 24 * 3600 * 1000) - (hours_ago * 3600 * 1000)
            
            call_history.append({
                'number': contact['phone'],
                'contact_name': contact['name'],
                'type': call_type,
                'duration': duration,
                'date': timestamp,
            })
    
    return call_history


# ============================================================================
# GALLERY DATA - 固定相册图片
# ============================================================================

def get_deterministic_gallery_images() -> List[Dict[str, Any]]:
    """获取确定性的相册图片数据。
    
    Returns:
        图片数据列表
    """
    categories = [
        ('Family', 8),
        ('Travel', 10),
        ('Work', 5),
        ('Screenshots', 6),
        ('Food', 5),
        ('Events', 4),
    ]
    
    images = []
    image_idx = 0
    
    for category, count in categories:
        for i in range(count):
            images.append({
                'filename': f'{category.lower()}_{i+1:03d}.jpg',
                'category': category,
                'text': f'{category} Photo {i+1}',
                'directory': f'/storage/emulated/0/DCIM/{category}',
            })
            image_idx += 1
    
    return images


# ============================================================================
# FILE SYSTEM DATA - 固定文件系统结构
# ============================================================================

def get_deterministic_file_structure() -> Dict[str, List[str]]:
    """获取确定性的文件系统结构。
    
    Returns:
        目录和文件映射
    """
    return {
        'Documents/Work': [
            'report_2023.txt',
            'meeting_notes.txt',
            'project_plan.txt',
        ],
        'Documents/Personal': [
            'todo_list.txt',
            'shopping_list.txt',
            'notes.txt',
        ],
        'Downloads': [
            'readme.txt',
            'user_manual.pdf',
            'install_guide.txt',
        ],
        'Music': [
            'song_01.mp3',
            'song_02.mp3',
            'song_03.mp3',
        ],
        'Videos': [
            'clip_01.mp4',
            'recording_01.mp4',
        ],
        'Pictures': [
            'photo_001.jpg',
            'photo_002.jpg',
            'screenshot_001.png',
        ],
    }


def get_deterministic_file_contents() -> List[Dict[str, str]]:
    """获取确定性的文件内容数据。
    
    Returns:
        文件内容列表
    """
    return [
        {
            'path': '/storage/emulated/0/Documents/Work/report_2023.txt',
            'content': 'Annual Report 2023\n\nKey findings:\n- Revenue increased by 15%\n- Customer satisfaction improved\n- New markets opened'
        },
        {
            'path': '/storage/emulated/0/Documents/Work/meeting_notes.txt',
            'content': 'Meeting Notes - Project Alpha\n\nAttendees: John, Sarah, Mike\nDate: 2023-10-15\n\nAgenda:\n1. Project status review\n2. Resource allocation\n3. Next steps'
        },
        {
            'path': '/storage/emulated/0/Documents/Personal/todo_list.txt',
            'content': 'Personal Todo List\n\n1. Buy groceries\n2. Call dentist\n3. Pay bills\n4. Schedule car service\n5. Plan weekend trip'
        },
        {
            'path': '/storage/emulated/0/Downloads/readme.txt',
            'content': 'README\n\nThis is a downloaded text file.\nVersion: 1.0\nDate: 2023-10-15'
        },
        {
            'path': '/storage/emulated/0/Downloads/install_guide.txt',
            'content': 'Installation Guide\n\n1. Extract the archive\n2. Run setup\n3. Follow the prompts'
        },
    ]


# ============================================================================
# MARKOR DATA - 固定 Markdown 文件
# ============================================================================

DETERMINISTIC_MARKOR_DOCUMENTS = [
    {
        'filename': 'meeting_notes.md',
        'content': '''# Meeting Notes

## Project Status Meeting - October 2023

### Attendees
- Alice Johnson
- Bob Smith
- Carol Williams

### Agenda
1. Project timeline review
2. Resource allocation
3. Next steps

### Action Items
- [ ] Complete design review by Friday
- [ ] Schedule client call
- [ ] Update documentation
'''
    },
    {
        'filename': 'todo_list.md',
        'content': '''# To-Do List

## Work Tasks
- [x] Review pull request
- [ ] Update API documentation
- [ ] Fix bug in login module
- [ ] Prepare presentation slides

## Personal Tasks
- [ ] Buy groceries
- [ ] Schedule dentist appointment
- [ ] Call mom
'''
    },
    {
        'filename': 'project_ideas.md',
        'content': '''# Project Ideas

## Mobile App Concepts

### 1. Fitness Tracker
Track daily workouts, calories, and progress

### 2. Recipe Manager
Save and organize favorite recipes

### 3. Budget Tracker
Monitor expenses and savings goals

## Web Projects
- Portfolio website
- Blog platform
- E-commerce template
'''
    },
    {
        'filename': 'shopping_list.md',
        'content': '''# Shopping List

## Groceries
- Milk
- Eggs
- Bread
- Cheese
- Fruits
- Vegetables

## Household Items
- Soap
- Toothpaste
- Paper towels
- Cleaning supplies

## Electronics
- USB cable
- Phone charger
'''
    },
    {
        'filename': 'study_notes.md',
        'content': '''# Study Notes

## Python Programming

### Data Types
- int, float, str, bool
- list, tuple, dict, set

### Control Flow
- if/elif/else
- for loops
- while loops

### Functions
```python
def greet(name):
    return f"Hello, {name}!"
```

## Important Concepts
1. Object-Oriented Programming
2. Error Handling
3. File I/O
'''
    },
    {
        'filename': 'health_log.md',
        'content': '''# Health Log

## Weekly Exercise
| Day | Activity | Duration |
|-----|----------|----------|
| Mon | Running  | 30 min   |
| Wed | Gym      | 45 min   |
| Fri | Swimming | 60 min   |
| Sat | Hiking   | 2 hours  |

## Nutrition Goals
- Drink 8 glasses of water daily
- Eat 5 servings of fruits/vegetables
- Limit processed foods
'''
    },
    {
        'filename': 'travel_plans.md',
        'content': '''# Travel Plans

## Japan Trip - Spring 2024

### Itinerary
- Day 1-3: Tokyo
- Day 4-5: Kyoto
- Day 6: Nara
- Day 7: Osaka

### Budget
- Flights: $1,200
- Accommodation: $800
- Food: $500
- Activities: $300

### Packing List
- Passport
- Adapters
- Comfortable shoes
'''
    },
    {
        'filename': 'book_reviews.md',
        'content': '''# Book Reviews

## Currently Reading
**The Pragmatic Programmer** by David Thomas

Rating: 5/5 stars

Key Takeaways:
1. DRY - Don't Repeat Yourself
2. Orthogonality in design
3. Tracer bullets

## To Read
- Clean Code
- Design Patterns
- Refactoring
'''
    },
    {
        'filename': 'recipes.md',
        'content': '''# Favorite Recipes

## Pasta Carbonara

### Ingredients
- 400g spaghetti
- 200g pancetta
- 4 eggs
- 100g parmesan
- Black pepper

### Instructions
1. Cook pasta al dente
2. Fry pancetta until crispy
3. Mix eggs and cheese
4. Combine all ingredients
5. Season with pepper
'''
    },
    {
        'filename': 'work_notes.md',
        'content': '''# Work Notes

## Q4 Goals
1. Complete mobile app v2.0
2. Improve test coverage to 80%
3. Reduce bug backlog by 50%

## Team Updates
- New developer joining next week
- Sprint review on Friday
- Holiday schedule reminder

## Technical Debt
- Refactor authentication module
- Update deprecated dependencies
- Improve error logging
'''
    },
]


def get_all_markor_documents() -> List[Dict[str, str]]:
    """获取所有固定 Markor 文档数据。
    
    Returns:
        文档字典列表，每个包含 filename 和 content
    """
    return DETERMINISTIC_MARKOR_DOCUMENTS.copy()


# ============================================================================
# MUSIC DATA (Retro Music) - 固定音乐数据
# ============================================================================

def get_deterministic_music_files() -> List[Dict[str, Any]]:
    """获取确定性的音乐文件数据。
    
    Returns:
        音乐文件数据列表
    """
    artists = ['Artist Alpha', 'Artist Beta', 'Artist Gamma', 'Artist Delta']
    albums = ['Album One', 'Album Two', 'Album Three']
    
    music_files = []
    song_idx = 0
    
    for artist_idx, artist in enumerate(artists):
        album = albums[artist_idx % len(albums)]
        # 每个艺术家3-4首歌
        num_songs = 3 + (artist_idx % 2)
        for i in range(num_songs):
            music_files.append({
                'filename': f'{artist.replace(" ", "_")}_song_{i+1:02d}.mp3',
                'title': f'Song {song_idx + 1}',
                'artist': artist,
                'album': album,
                'duration_ms': 180000 + (song_idx * 30000) % 120000,  # 3-5分钟
                'directory': '/storage/emulated/0/Music',
            })
            song_idx += 1
    
    return music_files


# ============================================================================
# VIDEO DATA (VLC) - 固定视频数据
# ============================================================================

def get_deterministic_video_files() -> List[Dict[str, Any]]:
    """获取确定性的视频文件数据。
    
    Returns:
        视频文件数据列表
    """
    video_types = ['clip', 'footage', 'scene', 'recording', 'highlight']
    
    videos = []
    for i in range(15):
        video_type = video_types[i % len(video_types)]
        videos.append({
            'filename': f'{video_type}_{i+1:02d}.mp4',
            'title': f'{video_type.title()} {i+1}',
            'duration_seconds': 20 + (i * 10) % 160,  # 20-180秒
            'directory': '/storage/emulated/0/VLCVideos',
            'messages': [f'Video Content {i+1}', 'Sample Video'],
        })
    
    return videos


# ============================================================================
# PACKAGE NAME TO DATA TYPE MAPPING
# ============================================================================

# 包名到数据类型的映射
PACKAGE_DATA_MAPPING = {
    # 系统级应用
    'com.google.android.contacts': ['contacts'],
    'com.simplemobiletools.smsmessenger': ['sms', 'contacts'],
    'com.google.android.dialer': ['call_history', 'contacts'],
    'com.simplemobiletools.gallery.pro': ['gallery'],
    'com.google.android.documentsui': ['files'],
    
    # 第三方应用
    'com.flauschcode.broccoli': ['recipes'],
    'org.tasks': ['tasks'],
    'net.cozic.joplin': ['joplin'],
    'de.dennisguse.opentracks': ['activities'],
    'com.simplemobiletools.calendar.pro': ['calendar'],
    'net.gsantner.markor': ['markor'],
    'com.arduia.expense': ['expenses'],
    'code.name.monkey.retromusic': ['music'],
    'org.videolan.vlc': ['videos'],
}


def get_data_types_for_package(package_name: str) -> List[str]:
    """获取指定包名需要注入的数据类型。
    
    Args:
        package_name: 应用包名
        
    Returns:
        数据类型列表
    """
    return PACKAGE_DATA_MAPPING.get(package_name, [])


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_data_summary():
    """打印所有固定数据的统计摘要。"""
    print("=" * 50)
    print("Deterministic Data Summary")
    print("=" * 50)
    
    if ANDROID_WORLD_AVAILABLE:
        print(f"Recipes (Broccoli):     {len(recipe_module._RECIPES)} items")
        print(f"Tasks:                  {len(task_app_utils._TASKS)} items")
        print(f"Joplin Folders:         {len(get_all_joplin_folders())} folders")
        print(f"Joplin Notes:           {get_total_joplin_notes_count()} notes")
        print(f"Activity Categories:    {len(get_all_activity_categories())} categories")
    else:
        print("(android_world not available - some data unavailable)")
    
    print(f"Contacts:               {len(DETERMINISTIC_CONTACTS)} contacts")
    print(f"Markor Documents:       {len(DETERMINISTIC_MARKOR_DOCUMENTS)} documents")
    print(f"Calendar Events:        25 events")
    print(f"Expense Records:        30 records")
    print(f"SMS Conversations:      ~30 messages")
    print(f"Call History:           ~20 calls")
    print(f"Gallery Images:         {sum(c[1] for c in [('Family', 8), ('Travel', 10), ('Work', 5), ('Screenshots', 6), ('Food', 5), ('Events', 4)])} images")
    print(f"Music Files:            ~14 files")
    print(f"Video Files:            15 files")
    print("=" * 50)


if __name__ == '__main__':
    print_data_summary()

