import concurrent.futures
import queue
import threading
from framework import utils


def run_task_with_multi_devices_rollout(run_task_rollout, task_arg_list, devices, emulator_exe=None, source_avd_name=None):
    """
    使用rollout逻辑并发执行任务：每个任务的所有attempts都要执行完
    """
    # 按照task_identifier排序，从前往后逐个执行任务
    task_arg_list_sorted = sorted(task_arg_list, key=lambda x: x[1].task_identifier)
    
    # 创建任务队列
    task_queue = queue.Queue()
    for task_arg in task_arg_list_sorted:
        task_queue.put(task_arg)
    
    def worker(device):
        while not task_queue.empty():
            try:
                # 获取下一个任务
                current_task_arg = task_queue.get_nowait()
            except queue.Empty:
                break
            
            max_retries = 2  # 最大重试次数
            success = False
            
            for attempt in range(max_retries + 1):
                try:
                    print(f"Executing task {current_task_arg[1].task_identifier} on device {device['serial']} (attempt {attempt + 1})")
                    
                    # 在任务执行前检查设备状态
                    if emulator_exe and source_avd_name:
                        if not utils.check_and_restart_device_if_needed(device, emulator_exe, source_avd_name):
                            print(f"Device {device['serial']} is not ready, skipping task {current_task_arg[1].task_identifier}")
                            break
                    
                    # 执行任务的rollout逻辑（包含所有attempts）
                    run_task_rollout(*current_task_arg, devices)  # 传递所有设备供rollout使用
                    success = True
                    break
                    
                except Exception as e:
                    print(f"Error executing task {current_task_arg[1].task_identifier} on {device['serial']} (attempt {attempt + 1}): {e}")
                    
                    # 如果还有重试机会，检查并重启设备
                    if attempt < max_retries:
                        if emulator_exe and source_avd_name:
                            print(f"Checking device {device['serial']} status after task failure...")
                            if not utils.check_and_restart_device_if_needed(device, emulator_exe, source_avd_name):
                                print(f"Failed to restart device {device['serial']}, skipping remaining retries")
                                break
                        else:
                            # 如果没有重启能力，简单等待一下再重试
                            import time
                            time.sleep(5)
                    else:
                        print(f"Max retries exceeded for task {current_task_arg[1].task_identifier} on {device['serial']}")
            
            if not success:
                print(f"Task {current_task_arg[1].task_identifier} failed on device {device['serial']} after all retries")
            
            # 标记任务完成
            task_queue.task_done()

    # 使用ThreadPoolExecutor管理并发执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        # 为每个设备提交一个worker
        futures = [executor.submit(worker, device) for device in devices]
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)


# 保留原函数以稳定调用（已弃用，但保留以防其他地方调用）
def run_task_with_multi_devices(run_task, task_arg_list, devices, emulator_exe=None, source_avd_name=None):
    """
    旧版本的多设备执行函数（已弃用，建议使用run_task_with_multi_devices_rollout）
    """
    print("Warning: run_task_with_multi_devices is deprecated. Use run_task_with_multi_devices_rollout instead.")
    
    # 按照task_identifier排序
    task_arg_list_sorted = sorted(task_arg_list, key=lambda x: x[1].task_identifier)
    
    # 创建任务队列（移除mirror task概念）
    task_queue = queue.Queue()
    for task_arg in task_arg_list_sorted:
        task_queue.put(task_arg)
    
    def worker(device):
        while not task_queue.empty():
            try:
                # 获取下一个任务
                current_task_arg = task_queue.get_nowait()
            except queue.Empty:
                break
            
            max_retries = 2  # 最大重试次数
            success = False
            
            for attempt in range(max_retries + 1):
                try:
                    print(f"Executing task {current_task_arg[1].task_identifier} on device {device['serial']} (attempt {attempt + 1})")
                    
                    # 在任务执行前检查设备状态
                    if emulator_exe and source_avd_name:
                        if not utils.check_and_restart_device_if_needed(device, emulator_exe, source_avd_name):
                            print(f"Device {device['serial']} is not ready, skipping task {current_task_arg[1].task_identifier}")
                            break
                    
                    # 执行任务
                    run_task(*current_task_arg, device)
                    success = True
                    break
                    
                except Exception as e:
                    print(f"Error executing task {current_task_arg[1].task_identifier} on {device['serial']} (attempt {attempt + 1}): {e}")
                    
                    # 如果还有重试机会，检查并重启设备
                    if attempt < max_retries:
                        if emulator_exe and source_avd_name:
                            print(f"Checking device {device['serial']} status after task failure...")
                            if not utils.check_and_restart_device_if_needed(device, emulator_exe, source_avd_name):
                                print(f"Failed to restart device {device['serial']}, skipping remaining retries")
                                break
                        else:
                            # 如果没有重启能力，简单等待一下再重试
                            import time
                            time.sleep(5)
                    else:
                        print(f"Max retries exceeded for task {current_task_arg[1].task_identifier} on {device['serial']}")
            
            if not success:
                print(f"Task {current_task_arg[1].task_identifier} failed on device {device['serial']} after all retries")
            
            # 标记任务完成
            task_queue.task_done()

    # 使用ThreadPoolExecutor管理并发执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        # 为每个设备提交一个worker
        futures = [executor.submit(worker, device) for device in devices]
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)
