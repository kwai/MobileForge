"""
任务调度器模块

负责智能地将任务分配到不同的emulator设备上，
实现相同app_name的任务尽可能分配到不同的emulator。
"""


def assign_device_for_task(work_item, devices, device_assignments):
    """
    智能分配设备：尽可能将相同app_name的任务分配到不同的emulator。
    如果无法满足，则将相同app_name下相同original_goal的任务分配到不同的emulator。
    
    分配策略优先级：
    1. 优先选择尚未处理过该app_name的设备
    2. 如果所有设备都已处理过该app，选择尚未处理过该(app_name, original_goal)组合的设备
    3. 都无法满足时，选择处理该app_name次数最少的设备（负载均衡）
    
    Args:
        work_item: 工作项，包含task信息
            - task: 任务对象，应包含app_name和original_goal属性
            - agent_name: 代理名称
            - attempt: 尝试次数
        devices: 可用设备列表
            - 每个设备是dict，包含"serial"键
        device_assignments: 每个设备已分配任务的统计信息
            {device_serial: {"app_counts": {app_name: count}, "goal_counts": {(app_name, original_goal): count}}}
    
    Returns:
        dict: 最优设备
    """
    task = work_item["task"]
    app_name = getattr(task, "app_name", None)
    original_goal = getattr(task, "original_goal", None)
    
    # 如果没有app_name信息，退化为简单负载均衡
    if app_name is None:
        return _select_least_loaded_device(devices, device_assignments)
    
    # 策略1: 优先选择尚未处理过该app_name的设备
    devices_without_app = _get_devices_without_app(devices, device_assignments, app_name)
    
    if devices_without_app:
        # 从未处理过该app的设备中选择总任务数最少的
        return _select_least_loaded_device(devices_without_app, device_assignments)
    
    # 策略2: 如果所有设备都已处理过该app，优先选择尚未处理过该(app_name, original_goal)组合的设备
    if original_goal is not None:
        devices_without_goal = _get_devices_without_goal(
            devices, device_assignments, app_name, original_goal
        )
        
        if devices_without_goal:
            # 从未处理过该goal的设备中选择该app处理次数最少的
            return _select_device_with_least_app_count(
                devices_without_goal, device_assignments, app_name
            )
    
    # 策略3: 都无法满足，选择处理该app_name次数最少的设备
    return _select_device_with_least_app_count(devices, device_assignments, app_name)


def update_device_assignments(device, work_item, device_assignments):
    """
    更新设备分配统计信息
    
    Args:
        device: 被分配的设备
        work_item: 工作项
        device_assignments: 设备分配统计字典（会被原地修改）
    """
    task = work_item["task"]
    app_name = getattr(task, "app_name", None)
    original_goal = getattr(task, "original_goal", None)
    serial = device["serial"]
    
    if serial not in device_assignments:
        device_assignments[serial] = {"app_counts": {}, "goal_counts": {}}
    
    if app_name is not None:
        if app_name not in device_assignments[serial]["app_counts"]:
            device_assignments[serial]["app_counts"][app_name] = 0
        device_assignments[serial]["app_counts"][app_name] += 1
        
        if original_goal is not None:
            goal_key = (app_name, original_goal)
            if goal_key not in device_assignments[serial]["goal_counts"]:
                device_assignments[serial]["goal_counts"][goal_key] = 0
            device_assignments[serial]["goal_counts"][goal_key] += 1


def print_device_assignment_stats(device_assignments):
    """
    打印设备分配统计信息
    
    Args:
        device_assignments: 设备分配统计字典
    """
    print("=== Device assignment statistics ===")
    for serial, stats in device_assignments.items():
        app_counts = stats.get("app_counts", {})
        if app_counts:
            print(f"  {serial}: {dict(app_counts)}")


# ============ 内部辅助函数 ============

def _select_least_loaded_device(devices, device_assignments):
    """选择总任务数最少的设备"""
    min_total = float("inf")
    best_device = devices[0]
    for device in devices:
        serial = device["serial"]
        total = sum(device_assignments.get(serial, {}).get("app_counts", {}).values())
        if total < min_total:
            min_total = total
            best_device = device
    return best_device


def _get_devices_without_app(devices, device_assignments, app_name):
    """获取尚未处理过指定app的设备列表"""
    devices_without_app = []
    for device in devices:
        serial = device["serial"]
        app_counts = device_assignments.get(serial, {}).get("app_counts", {})
        if app_name not in app_counts or app_counts[app_name] == 0:
            devices_without_app.append(device)
    return devices_without_app


def _get_devices_without_goal(devices, device_assignments, app_name, original_goal):
    """获取尚未处理过指定(app_name, original_goal)组合的设备列表"""
    goal_key = (app_name, original_goal)
    devices_without_goal = []
    for device in devices:
        serial = device["serial"]
        goal_counts = device_assignments.get(serial, {}).get("goal_counts", {})
        if goal_key not in goal_counts or goal_counts[goal_key] == 0:
            devices_without_goal.append(device)
    return devices_without_goal


def _select_device_with_least_app_count(devices, device_assignments, app_name):
    """选择处理指定app次数最少的设备"""
    min_app_count = float("inf")
    best_device = devices[0]
    for device in devices:
        serial = device["serial"]
        app_count = device_assignments.get(serial, {}).get("app_counts", {}).get(app_name, 0)
        if app_count < min_app_count:
            min_app_count = app_count
            best_device = device
    return best_device

