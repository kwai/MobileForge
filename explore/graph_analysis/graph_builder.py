import os
import sys
sys.setrecursionlimit(2000000) # 增加递归深度限制
# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import logging
from typing import Dict, List, Any, Optional

# 导入 autodroid 中必要的类
from autodroid.utg import UTG
from autodroid.device_state import DeviceState
from autodroid.input_event import TouchEvent, LongTouchEvent, ScrollEvent, SetTextEvent, KeyEvent, UIEvent
from MLLM_Agent.json_action import JSONAction # 导入 JSONAction 类

# 用于加载 .pkl.zst 文件
import zstandard
import pickle
from PIL import Image

# 创建 MockDevice 类
class MockDevice:
    def __init__(self, serial=None, output_dir=None, screen_width=1080, screen_height=1920):
        self.serial = serial
        self.output_dir = output_dir
        self._width = screen_width
        self._height = screen_height
        self.logger = logging.getLogger(self.__class__.__name__)
        self.minicap = None  # 模拟 minicap 不存在
        self.adapters = {} # 模拟 adapters
        self.humanoid = None # 模拟 humanoid 不存在

    def get_width(self, refresh=False):
        return self._width

    def get_height(self, refresh=False):
        return self._height

    def get_display_info(self):
        return {"width": self._width, "height": self._height}

    def get_model_number(self):
        return "MockModel"

    def get_sdk_version(self):
        return "MockSDK"

# 创建 MockApp 类
class MockApp:
    def __init__(self, package_name, main_activity=None, activities=None):
        self.package_name = package_name
        self.main_activity = main_activity if main_activity else f"{package_name}.MainActivity"
        self.activities = activities if activities else [self.main_activity]
        self.hashes = ["", "", "mock_sha256"] # 模拟 hash 值

    def get_app_name(self):
        return self.package_name.split('.')[-1]

    def get_package(self):
        return self.package_name

    def get_main_activity(self):
        return self.main_activity

    def get_activities(self):
        return self.activities

    def get_androidversion_code(self):
        return "1"

    def get_androidversion_name(self):
        return "1.0"

def load_zst_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            data = pickle.load(reader)
    return data

def _bbox_contains(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> bool:
    """
    检查 bbox1 是否包含 bbox2。
    :param bbox1: 包含 'x_min', 'y_min', 'x_max', 'y_max' 的字典。
    :param bbox2: 包含 'x_min', 'y_min', 'x_max', 'y_max' 的字典。
    :return: 如果 bbox1 包含 bbox2，则返回 True，否则返回 False。
    """
    return (
        bbox1['x_min'] <= bbox2['x_min'] and
        bbox1['y_min'] <= bbox2['y_min'] and
        bbox1['x_max'] >= bbox2['x_max'] and
        bbox1['y_max'] >= bbox2['y_max']
    )

def _convert_gui_explorer_elements_to_autodroid_views(
    gui_explorer_elements: List[Dict[str, Any]],
    screen_width: int,
    screen_height: int,
) -> List[Dict[str, Any]]:
    # Step 1: Create initial autodroid_views list with temp_id == index
    autodroid_views = []
    for i, element in enumerate(gui_explorer_elements):
        view_dict = {
            "temp_id": i,  # temp_id directly corresponds to its index in this list
            "class": element.get("class_name"),
            "resource_id": element.get("resource_id"),
            "text": element.get("text"),
            "content_description": element.get("content_description"),
            "bounds": [
                [int(element["bbox_pixels"]['x_min']), int(element["bbox_pixels"]['y_min'])],
                [int(element["bbox_pixels"]['x_max']), int(element["bbox_pixels"]['y_max'])]
            ],
            "bbox": element["bbox"],
            "clickable": element.get("is_clickable", False),
            "long_clickable": element.get("is_long_clickable", False),
            "scrollable": element.get("is_scrollable", False),
            "editable": element.get("is_editable", False),
            "checkable": element.get("is_checkable", False),
            "checked": element.get("is_checked", False),
            "selected": element.get("is_selected", False),
            "enabled": element.get("is_enabled", False),
            "focused": element.get("is_focused", False),
            "focusable": False, # Temporarily set to False to avoid potential recursion issues if autodroid's logic uses this recursively
            "visible": element.get("is_visible", False),
            "parent": -1,  # Initialize parent to -1 (no parent)
            "children": [],  # Initialize children as empty list
            "package_name": element.get("package_name") # Keep package_name for system UI filtering
        }
        autodroid_views.append(view_dict)

    # Step 2: Determine direct parent for each view based on smallest enclosing bbox
    for i, child_view in enumerate(autodroid_views):
        if child_view.get("package_name") == "com.android.systemui":
            continue

        best_parent_index = -1
        min_parent_area = float('inf')

        for j, potential_parent_view in enumerate(autodroid_views):
            if i == j: # A view cannot be its own parent
                continue
            if potential_parent_view.get("package_name") == "com.android.systemui":
                continue

            # Check if potential_parent_view geometrically contains child_view
            if _bbox_contains(potential_parent_view["bbox"], child_view["bbox"]):
                parent_bbox = potential_parent_view["bbox"]
                parent_area = (parent_bbox['x_max'] - parent_bbox['x_min']) * \
                              (parent_bbox['y_max'] - parent_bbox['y_min'])

                # If this potential parent is smaller (closer) than previous best, update
                if parent_area < min_parent_area:
                    min_parent_area = parent_area
                    best_parent_index = j
        
        child_view["parent"] = best_parent_index

    # Step 2.5: Detect and break cycles in parent relationships
    # This is a crucial step to prevent RecursionError in get_all_ancestors
    for i, view_data in enumerate(autodroid_views):
        current_id = i
        path = set()
        while current_id != -1:
            if current_id in path: # Cycle detected
                logging.warning(f"Cycle detected for view {i}. Breaking cycle.")
                view_data["parent"] = -1 # Detach from the cycle
                break
            path.add(current_id)
            parent_id = autodroid_views[current_id]["parent"]
            if parent_id == -1: # Reached a root
                break
            current_id = parent_id
            
            # Ensure parent_id is within bounds, to prevent IndexError during cycle detection
            if not (0 <= current_id < len(autodroid_views)):
                logging.warning(f"Invalid parent_id {current_id} encountered for view {view_data['temp_id']}. Breaking path.")
                view_data["parent"] = -1
                break

    # Step 3: Populate children lists based on the assigned parents
    for i, parent_view in enumerate(autodroid_views):
        parent_view["children"] = [] # Clear any previous children assignment (important for safety)
        for j, child_view in enumerate(autodroid_views):
            if child_view["parent"] == i: # If this child's parent is the current parent_view
                parent_view["children"].append(j) # Add child's index to parent's children list

    return autodroid_views

def build_utg_from_trajectories(trajectory_dir: str, package_name: str, output_dir: str):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 创建 MockDevice 和 MockApp
    mock_device = MockDevice(serial="mock_device", output_dir=output_dir)
    mock_app = MockApp(package_name=package_name)

    # 初始化 UTG
    utg = UTG(device=mock_device, app=mock_app, random_input=False, output_dir=output_dir)

    # 用于保存前一个设备状态，以便构建边
    previous_device_state = None

    # 遍历轨迹文件
    trajectory_files = []
    for root, _, files in os.walk(trajectory_dir):
        for file in files:
            if file.endswith(".pkl.zst"):
                trajectory_files.append(os.path.join(root, file))

    if not trajectory_files:
        logger.warning(f"No trajectory files found in {trajectory_dir}")
        return

    logger.info(f"Found {len(trajectory_files)} trajectory files. Building UTG...")

    for file_path in sorted(trajectory_files):
        logger.info(f"Processing trajectory file: {file_path}")
        trajectory_data = load_zst_pkl_file(file_path)
        logger.info(f"Type of trajectory_data: {type(trajectory_data)}")
        logger.info(f"Length of trajectory_data: {len(trajectory_data) if hasattr(trajectory_data, '__len__') else 'N/A'}")

        if not trajectory_data:
            logger.warning(f"Trajectory data in {file_path} is empty. Skipping.")
            continue

        logger.info(f"Entering for step_data in trajectory_data loop for {file_path}")
        for step_data in trajectory_data:
            # 注意: foreground_activity 和 activity_stack 需要从 step_data 中提取
            # 如果 step_data 中没有这些信息，可能需要从文件路径或假定默认值
            foreground_activity = package_name + "/" + mock_app.main_activity.split('.')[-1] # 确保格式为 package_name/activity_name

            # 检查 step_data 中是否存在 ui_elements 和 logical_screen_size
            if "ui_elements" not in step_data or "logical_screen_size" not in step_data or "raw_screenshot" not in step_data:
                logger.warning(f"Missing 'ui_elements', 'logical_screen_size' or 'raw_screenshot' in step_data from {file_path}. Skipping this step.")
                continue

            # 调用转换函数，获取 autodroid 期望的 views 格式
            processed_ui_elements = _convert_gui_explorer_elements_to_autodroid_views(
                step_data["ui_elements"],
                step_data["logical_screen_size"][0],
                step_data["logical_screen_size"][1]
            )

            # 使用 step_data 中的屏幕尺寸更新 MockDevice
            mock_device._width = step_data["logical_screen_size"][0]
            mock_device._height = step_data["logical_screen_size"][1]

            # 保存原始截图到指定目录
            raw_screenshot_np = step_data["raw_screenshot"]
            pil_image = Image.fromarray(raw_screenshot_np)

            screenshot_filename = f"screen_{os.path.basename(file_path).split('.')[0]}.png"
            actual_screenshot_dir = os.path.join(output_dir, "utg_results", "states")
            os.makedirs(actual_screenshot_dir, exist_ok=True)
            actual_screenshot_path = os.path.join(actual_screenshot_dir, screenshot_filename)
            pil_image.save(actual_screenshot_path)
            
            logger.info(f"Screenshot path: {actual_screenshot_path}, Exists: {os.path.exists(actual_screenshot_path)}")

            current_device_state = DeviceState(
                device=mock_device,
                views=processed_ui_elements,
                foreground_activity=foreground_activity,
                activity_stack=[foreground_activity], # 简化处理
                background_services=[], # 简化处理
                tag=os.path.basename(file_path).split(".")[0], # 使用文件名作为tag
                screenshot_path=actual_screenshot_path # 指向实际的截图文件
            )
            
            # 显式将当前状态添加到UTG，以确保所有状态都被记录
            utg.add_node(current_device_state)

            # 构建事件对象
            converted_action = step_data.get("converted_action")
            event = None
            logger.info(f"Converted action from step_data: {converted_action}")

            if isinstance(converted_action, JSONAction): # converted_action 是 JSONAction 类的实例
                action_type = converted_action.action_type
                # 尝试从 converted_action 中获取事件相关的 view 信息
                # 注意：这里假设 JSONAction 包含一个 'index' 字段来引用 ui_elements 列表的索引
                event_view_id = getattr(converted_action, "index", None) # 使用 getattr 安全地获取属性
                event_view = None
                if event_view_id is not None and 0 <= event_view_id < len(processed_ui_elements):
                    event_view = processed_ui_elements[event_view_id]
                logger.info(f"Event view for action (id={event_view_id}): {event_view}")

                if action_type == "click":
                    event = TouchEvent(view=event_view)
                elif action_type == "long_press":
                    event = LongTouchEvent(view=event_view)
                elif action_type == "scroll":
                    # ScrollEvent 需要 direction
                    direction = getattr(converted_action, "direction", "UNKNOWN")
                    event = ScrollEvent(view=event_view, direction=direction)
                elif action_type == "input_text":
                    # SetTextEvent 需要 text
                    text = getattr(converted_action, "text", "")
                    event = SetTextEvent(view=event_view, text=text)
                elif action_type == "navigate_back":
                    event = KeyEvent(name="BACK")
                elif action_type == "navigate_home":
                    event = KeyEvent(name="HOME")
                # 可以根据需要添加更多事件类型，例如 open_app
                elif action_type == "open_app":
                    app_name = getattr(converted_action, "app_name", "")
                    # 对于 open_app 事件，可以创建一个自定义的 UIEvent 或一个特殊的 KeyEvent
                    # 这里我们暂时将其视为一个特殊的 KeyEvent, 或者可以扩展 input_event.py
                    event = KeyEvent(name=f"OPEN_APP_{app_name.upper().replace(' ', '_')}")
                elif action_type == "status":
                    # status action 不会生成实际的 UI 事件，因此我们忽略它
                    event = None

            logger.info(f"Generated event object: {event}")

            if previous_device_state and event:
                logger.info(f"Attempting to add transition from state: {previous_device_state.state_str} to {current_device_state.state_str} with event type: {event.__class__.__name__}, event details: {event.__dict__}")
                utg.add_transition(event, previous_device_state, current_device_state)
                logger.info(f"Added transition from {previous_device_state.state_str} to {current_device_state.state_str} with event {event.__class__.__name__}")
            
            previous_device_state = current_device_state

    # 导出 UTG 为 utg.js
    utg._UTG__output_utg() # 直接调用私有方法

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build UI Transition Graph from exploration trajectories.")
    parser.add_argument("-trajectory_dir", required=True, help="Path to the directory containing exploration trajectory .pkl.zst files.")
    parser.add_argument("-package_name", required=True, help="Package name of the app being analyzed.")
    parser.add_argument("-output_dir", default="./utg_results", help="Output directory for the UTG visualization files.")

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    build_utg_from_trajectories(args.trajectory_dir, args.package_name, args.output_dir)
