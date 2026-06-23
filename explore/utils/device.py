import dataclasses
from typing import Any, Optional
import xml.etree.ElementTree as ET
from dataclasses import asdict
import time
from PIL import Image
import imagehash
import numpy as np
import logging


@dataclasses.dataclass
class BoundingBox:
    """Class for representing a bounding box."""

    x_min: float | int
    x_max: float | int
    y_min: float | int
    y_max: float | int

    @property
    def center(self) -> tuple[float, float]:
        """Gets center of bounding box."""
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float | int:
        """Gets width of bounding box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float | int:
        """Gets height of bounding box."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float | int:
        return self.width * self.height


@dataclasses.dataclass
class UIElement:
    """Represents a UI element."""

    text: Optional[str] = None  # 这个就是文本框里面的内容
    content_description: Optional[str] = None
    class_name: Optional[str] = None
    bbox: Optional[BoundingBox] = None  # 根据屏幕尺寸进行归一化的bbox
    bbox_pixels: Optional[BoundingBox] = None
    hint_text: Optional[str] = None
    is_checked: Optional[bool] = None
    is_checkable: Optional[bool] = None
    is_clickable: Optional[bool] = None
    is_editable: Optional[bool] = None
    is_enabled: Optional[bool] = None
    is_focused: Optional[bool] = None
    is_focusable: Optional[bool] = None
    is_long_clickable: Optional[bool] = None
    is_scrollable: Optional[bool] = None
    is_selected: Optional[bool] = None
    is_visible: Optional[bool] = None
    package_name: Optional[str] = None
    resource_name: Optional[str] = None
    tooltip: Optional[str] = None
    resource_id: Optional[str] = None

    element_id: Optional[str] = None
    # Make the actual storage field private and part of the dataclass init
    _image_hash: Optional[str] = dataclasses.field(
        default=None, repr=False
    )  # Use field to control repr if needed
    uid: Optional[str] = None

    def _get_element_id(self) -> str:
        """Generates a unique ID for the UI element based on its properties."""
        if self.element_id is None and self.bbox_pixels:
            elem_w, elem_h = self.bbox_pixels.width, self.bbox_pixels.height
            if self.resource_id:
                self.element_id = self.resource_id.replace(":", ".").replace("/", "_")
            else:
                self.element_id = f"{self.class_name}_{elem_w}_{elem_h}"
            if self.content_description and len(self.content_description) < 20:
                content_desc = (
                    self.content_description.replace("/", "_")
                    .replace(" ", "")
                    .replace(":", "_")
                )
                self.element_id += f"_{content_desc}"
        return self.element_id

    def _maybe_is_editable(self) -> bool:
        """Checks if the UI element is editable based on its properties."""
        if self.is_editable:
            return True
        if self.is_clickable and self.class_name:
            for s in [
                "EditText",
                "TextView",
                "AutoCompleteTextView",
                "MultiAutoCompleteTextView",
            ]:
                if s in self.class_name:
                    return True
        return False

    def _update_uid(self) -> None:  # WARNING:在更新image_hash之后记得调用这个函数
        if self.element_id is not None and self._image_hash is not None:
            self.uid = f"{self.element_id}_{self._image_hash }"
        elif self._image_hash is not None:
            self.uid = f"{self._image_hash }"
        elif self.element_id is not None:
            self.uid = f"{self.element_id}"

    def set_image_hash(self, image: Image.Image) -> None:
        """Set the image hash for the UI element."""
        self.image_hash = str(
            imagehash.phash(image, hash_size=16, highfreq_factor=8)
        ).upper()

    # Define the property getter and setter
    @property
    def image_hash(self) -> Optional[str]:
        """Getter for the image hash."""
        return self._image_hash

    # 用setter实现更新image_hash之后自动更新uid
    @image_hash.setter
    def image_hash(self, value: Optional[str]) -> None:
        """Setter for the image hash that also updates the UID."""
        if self._image_hash != value:
            self._image_hash = value
            self._update_uid()

    def __post_init__(self):
        """Post-initialization to ensure element_id is set."""
        if self.bbox_pixels is not None and isinstance(self.bbox_pixels, dict):
            self.bbox_pixels = BoundingBox(**self.bbox_pixels)
        if self.bbox is not None and isinstance(self.bbox, dict):
            self.bbox = BoundingBox(**self.bbox)
        self.element_id = self._get_element_id()
        self.is_editable = self._maybe_is_editable()
        self._update_uid()


def _normalize_bounding_box(
    node_bbox: BoundingBox,
    screen_width_height_px: tuple[int, int],
) -> BoundingBox:
    width, height = screen_width_height_px
    return BoundingBox(
        node_bbox.x_min / width,
        node_bbox.x_max / width,
        node_bbox.y_min / height,
        node_bbox.y_max / height,
    )


def _parse_ui_hierarchy(xml_string: str) -> dict[str, Any]:
    """Parses the UI hierarchy XML into a dictionary structure."""
    root = ET.fromstring(xml_string)

    def parse_node(node):
        result = node.attrib
        result["children"] = [parse_node(child) for child in node]
        return result

    return parse_node(root)


def xml_dump_to_ui_elements(
    xml_string: str,
    exclude_invisible_elements: bool = False,
    screen_size: Optional[tuple[int, int]] = None,
    screenshot: Optional[Image.Image] = None,
) -> list[UIElement]:
    """Converts a UI hierarchy XML dump from uiautomator dump to UIElements.
    Args:
        xml_string: The XML string containing the UI hierarchy dump.
        exclude_invisible_elements: True if invisible elements should not be
      returned.
        screen_size: The size of the device screen in pixels (width, height).

    Returns:
        The extracted UI elements.
    """

    def text_or_none(text: Optional[str]) -> Optional[str]:
        """Returns None if text is None or 0 length."""
        return text if text else None

    parsed_hierarchy = _parse_ui_hierarchy(xml_string)
    ui_elements = []

    def process_node(node, screen_size=None, is_root=False, parent_node=None):
        bounds = node.get("bounds")
        bbox_pixels, bbox_normalized = None, None
        if bounds:
            x_min, y_min, x_max, y_max = map(
                int, bounds.strip("[]").replace("][", ",").split(",")
            )
            bbox_pixels = BoundingBox(x_min, x_max, y_min, y_max)
            if screen_size is not None:
                bbox_normalized = _normalize_bounding_box(bbox_pixels, screen_size)

        ui_element = UIElement(
            text=text_or_none(node.get("text")),
            content_description=text_or_none(node.get("content-desc")),
            class_name=text_or_none(node.get("class")),
            bbox=bbox_normalized,
            bbox_pixels=bbox_pixels,
            hint_text=text_or_none(node.get("hint")),
            is_checked=node.get("checked") == "true",
            is_checkable=node.get("checkable") == "true",
            is_clickable=node.get("clickable") == "true",
            is_enabled=node.get("enabled") == "true",
            is_focused=node.get("focused") == "true",
            is_focusable=node.get("focusable") == "true",
            is_long_clickable=node.get("long-clickable") == "true",
            is_scrollable=node.get("scrollable") == "true",
            is_selected=node.get("selected") == "true",
            package_name=text_or_none(node.get("package")),
            resource_id=text_or_none(node.get("resource-id")),
            is_visible=node.get("visible-to-user") == "true",
        )
        if parent_node and parent_node.element_id:
            # ui_element.element_id = f"{parent_node.element_id}_{ui_element.element_id}"
            pass
        if not is_root:
            if (
                not (node.get("children", None) is not None)
                or (text_or_none(node.get("content-desc")) is not None)
                or (node.get("scrollable", "false") == "true")
                or (node.get("clickable", "false") == "true")
            ):
                if exclude_invisible_elements and not (
                    node.get("visible-to-user", "false") == "true"
                ):
                    # continue
                    pass
                else:
                    if screen_size is None or validate_ui_element(
                        ui_element, screen_size
                    ):
                        if ui_element.bbox_pixels and screenshot:
                            image_hash = (
                                imagehash.phash(  # NOTE:使用phash来计算图片的hash值
                                    screenshot.crop(
                                        (
                                            ui_element.bbox_pixels.x_min,
                                            ui_element.bbox_pixels.y_min,
                                            ui_element.bbox_pixels.x_max,
                                            ui_element.bbox_pixels.y_max,
                                        )
                                    ),
                                    hash_size=16,
                                    highfreq_factor=8,
                                )
                            )
                            ui_element.image_hash = str(image_hash).upper()
                        ui_elements.append(ui_element)

        for child in node.get("children", []):
            process_node(
                child, screen_size=screen_size, is_root=False, parent_node=ui_element
            )

    process_node(parsed_hierarchy, screen_size=screen_size, is_root=True)
    return ui_elements


def _generate_ui_element_description(ui_element: UIElement, index: int) -> str:
    """Generate a description for a given UI element with important information.

    Args:
      ui_element: UI elements for the current screen.
      index: The numeric index for the UI element.

    Returns:
      The description for the UI element.
    """
    element_description = f'UI element {index}: {{"index": {index}, '
    if ui_element.text:
        element_description += f'"text": "{ui_element.text}", '
    if ui_element.content_description:
        element_description += (
            f'"content_description": "{ui_element.content_description}", '
        )
    if ui_element.hint_text:
        element_description += f'"hint_text": "{ui_element.hint_text}", '
    if ui_element.tooltip:
        element_description += f'"tooltip": "{ui_element.tooltip}", '
    element_description += (
        f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
    )
    element_description += (
        '"is_long_clickable":'
        f' {"True" if ui_element.is_long_clickable else "False"}, '
    )
    element_description += (
        f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
    )
    if ui_element.is_scrollable:
        element_description += '"is_scrollable": True, '
    if ui_element.is_focusable:
        element_description += '"is_focusable": True, '
    element_description += (
        f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
    )
    element_description += (
        f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
    )
    return element_description[:-2] + "}"  # 这里的[:-2]是为了去掉', '


def validate_ui_element(
    ui_element: UIElement,
    screen_width_height_px: tuple[int, int],
) -> bool:
    """Used to filter out invalid UI element."""
    screen_width, screen_height = screen_width_height_px

    # Filters out invisible element.
    if not ui_element.is_visible:
        return False

    # Filters out element with invalid bounding box.
    if ui_element.bbox_pixels:
        x_min = ui_element.bbox_pixels.x_min
        x_max = ui_element.bbox_pixels.x_max
        y_min = ui_element.bbox_pixels.y_min
        y_max = ui_element.bbox_pixels.y_max

        if (
            x_min >= x_max
            or x_min >= screen_width
            or x_max <= 0
            or y_min >= y_max
            or y_min >= screen_height
            or y_max <= 0
        ):
            return False

    # if not is_element_available(ui_element):
    #     return False

    return True


def _generate_ui_elements_description_list(
    ui_elements: list[UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
    """Generate concise information for a list of UIElement.

    Args:
      ui_elements: UI elements for the current screen.
      screen_width_height_px: The height and width of the screen in pixels.

    Returns:
      Concise information for each UIElement.
    """
    tree_info = ""
    for index, ui_element in enumerate(ui_elements):
        if validate_ui_element(ui_element, screen_width_height_px):
            tree_info += _generate_ui_element_description(ui_element, index) + "\n"
    return tree_info


import base64
import re
from typing import Any, Optional
import cv2
import numpy as np


def _logical_to_physical(
    logical_coordinates: tuple[int, int],
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
) -> tuple[int, int]:
    """Convert logical coordinates to physical coordinates.

    Args:
      logical_coordinates: The logical coordinates for the point.
      logical_screen_size: The logical screen size.
      physical_frame_boundary: The physical coordinates in portrait orientation
        for the upper left and lower right corner for the frame.
      orientation: The current screen orientation.

    Returns:
      The physical coordinate for the point in portrait orientation.

    Raises:
      ValueError: If the orientation is not valid.
    """
    x, y = logical_coordinates
    px0, py0, px1, py1 = physical_frame_boundary
    px, py = px1 - px0, py1 - py0
    lx, ly = logical_screen_size
    if orientation == 0:
        return (int(x * px / lx) + px0, int(y * py / ly) + py0)
    if orientation == 1:
        return (px - int(y * px / ly) + px0, int(x * py / lx) + py0)
    if orientation == 2:
        return (px - int(x * px / lx) + px0, py - int(y * py / ly) + py0)
    if orientation == 3:
        return (int(y * px / ly) + px0, py - int(x * py / lx) + py0)
    print("Invalid orientation.")
    raise ValueError("Unsupported orientation.")


def _ui_element_logical_corner(
    ui_element: UIElement, orientation: int
) -> list[tuple[int, int]]:
    """Get logical coordinates for corners of a given UI element.

    Args:
      ui_element: The corresponding UI element.
      orientation: The current orientation.

    Returns:
      Logical coordinates for upper left and lower right corner for the UI
      element.

    Raises:
      ValueError: If bounding box is missing.
      ValueError: If orientation is not valid.
    """
    if ui_element.bbox_pixels is None:
        raise ValueError("UI element does not have bounding box.")
    if orientation == 0:
        return [
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_min)),
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_max)),
        ]
    if orientation == 1:
        return [
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_max)),
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_min)),
        ]
    if orientation == 2:
        return [
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_max)),
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_min)),
        ]
    if orientation == 3:
        return [
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_min)),
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_max)),
        ]
    raise ValueError("Unsupported orientation.")


def add_ui_element_mark(
    screenshot: np.ndarray,
    ui_element: UIElement,
    index: int | str,
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
):
    """Add mark (a bounding box plus index) for a UI element in the screenshot.

    Args:
      screenshot: The screenshot as a numpy ndarray.
      ui_element: The UI element to be marked.
      index: The index for the UI element.
      logical_screen_size: The logical screen size.
      physical_frame_boundary: The physical coordinates in portrait orientation
        for the upper left and lower right corner for the frame.
      orientation: The current screen orientation.
    """
    if ui_element.bbox_pixels:
        upper_left_logical, lower_right_logical = _ui_element_logical_corner(
            ui_element, orientation
        )
        upper_left_physical = _logical_to_physical(
            upper_left_logical,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
        lower_right_physical = _logical_to_physical(
            lower_right_logical,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )

        cv2.rectangle(
            screenshot,
            upper_left_physical,
            lower_right_physical,
            color=(0, 255, 0),
            # thickness=2,
            thickness=3,
        )
        screenshot[
            upper_left_physical[1] + 1 : upper_left_physical[1] + 25,
            upper_left_physical[0] + 1 : upper_left_physical[0] + 35,
            :,
        ] = (255, 255, 255)
        cv2.putText(
            screenshot,
            str(index),
            (
                upper_left_physical[0] + 1,
                upper_left_physical[1] + 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            thickness=2,
            # thickness=3,#这个会导致标签的数字变成一团
        )


def add_screenshot_label(screenshot: np.ndarray, label: str):
    """Add a text label to the right bottom of the screenshot.

    Args:
      screenshot: The screenshot as a numpy ndarray.
      label: The text label to add, just a single word.
    """
    if len(label) > 8:
        print(f"Label {label} is too long, please use a shorter one.")
    height, width, _ = screenshot.shape
    screenshot[height - 30 : height, width - 150 : width, :] = (255, 255, 255)
    cv2.putText(
        screenshot,
        label,
        (width - 135, height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        # thickness=2,
        thickness=3,
    )


import logging
import uiautomator2 as u2
from PIL import Image
from typing import List, Tuple, Union
import time
import re


def get_available_devices() -> list[str]:
    """
    Get a list of device serials connected via adb
    :return: list of str, each str is a device serial number
    """
    import subprocess

    r = subprocess.check_output(["adb", "devices"])
    if not isinstance(r, str):
        r = r.decode()
    devices = []
    for line in r.splitlines():
        segs = line.strip().split()
        if len(segs) == 2 and segs[1] == "device":
            devices.append(segs[0])
    return devices


class Device(object):

    def __init__(self, device_serial: str = None) -> None:
        """
        Initialize a device connection with the bare minimum requirements.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        if device_serial is None:
            all_devices = get_available_devices()
            if len(all_devices) == 0:
                raise Exception("No device connected.")
            device_serial = all_devices[0]
        self.logger.info(f"Using device {device_serial}")
        self.device_serial = device_serial
        self.u2d = None
        self.connect()

    def __del__(self) -> None:
        self.disconnect()

    def connect(self) -> None:
        """
        Connect to the device.
        """
        self._prior_ui_elements_state = None
        if self.u2d is None:
            self.u2d = u2.connect(self.device_serial)
        self.logger.info(f"Connected to device.\n{self.u2d.info}")

    def disconnect(self) -> None:
        """
        Disconnect from the device.
        """
        if self.u2d is not None:
            self.u2d.stop_uiautomator()
            self.u2d = None
        self.logger.info("Disconnected from device.")

    def run_shell_command(
        self, cmdargs: Union[str, List[str]], timeout=60
    ) -> tuple[str, int]:
        """
        Run shell command on device

        Args:
            cmdargs: str or list, example: "ls -l" or ["ls", "-l"]
            timeout: seconds of command run, works on when stream is False

        Returns:
            return type is `namedtuple("ShellResponse", ("output", "exit_code"))`

        Raises:
            AdbShellError
        """
        return self.u2d.shell(cmdargs, timeout=timeout)

    def launch_app(
        self,
        package_name: str,
        use_monkey: bool = False,
        timeout: float = 20.0,
        front: bool = True,
        activity: str = None,
    ) -> None:
        """
        Args:
            package_name (str): package name
            use_monkey (bool): use monkey command to start app when activity is not given
            timeout (float): maxium wait time, 0 means no wait
            front (bool): wait until app is current app
        """
        self.u2d.app_start(package_name, use_monkey=use_monkey, activity=activity)
        if timeout > 0:
            self.u2d.app_wait(package_name, front=front, timeout=timeout)

    def stop_app(self, package_name: str):
        """Stop one application"""
        self.u2d.app_stop(package_name)

    def stop_all_apps(self, excludes: list = []) -> List[str]:
        """Stop all third party applications
        Args:
            excludes (list): apps that do not want to kill

        Returns:
            a list of killed apps
        """
        return self.u2d.app_stop_all(excludes=excludes)

    def list_running_app(self) -> List[str]:
        """
        Returns:
            list of running apps
        """
        return self.u2d.app_list_running()

    def list_installed_app(self, filter: str = None) -> List[str]:
        """
        List installed app package names

        Args:
            filter: [-f] [-d] [-e] [-s] [-3] [-i] [-u] [--user USER_ID] [FILTER]

        Returns:
            list of apps by filter
        """
        return self.u2d.app_list(filter)

    def get_viewhierachy(self) -> str:
        viewhierachy = self.u2d.dump_hierarchy(
            compressed=False, pretty=False, max_depth=50
        )
        return viewhierachy

    def get_screenshot(self) -> Image.Image:
        return self.u2d.screenshot().convert("RGB")

    def get_screen_size(self) -> Tuple[int, int]:
        """
        Returns:
            screen width and height
        """
        return self.u2d.window_size()

    def get_top_activity_name(self) -> str:
        current = self.u2d.app_current()
        return current["activity"]

    def get_top_package_name(self) -> str:
        current = self.u2d.app_current()
        return current["package"]

    def get_installed_apps(self) -> List[str]:
        return self.u2d.app_list()

    def click(self, x: int, y: int):
        self.u2d.click(x, y)

    def long_click(self, x: int, y: int, duration: float = 2.0):
        self.u2d.long_click(x, y, duration)

    def double_click(self, x: int, y: int):
        self.u2d.double_click(x, y)

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        self.u2d.drag(x1, y1, x2, y2, duration)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        self.u2d.swipe(x1, y1, x2, y2, duration)

    def swip_up(self):
        self.u2d.swipe_ext("up")

    def swip_down(self):
        self.u2d.swipe_ext("down")

    def swip_left(self):
        self.u2d.swipe_ext("left")

    def swip_right(self):
        self.u2d.swipe_ext("right")

    def is_keyboard_shown(self) -> bool:
        output, exit_code = self.u2d.shell(
            "dumpsys input_method | grep mInputShown", timeout=120
        )
        # If the output shows mInputShown=true,
        # it means that the current input method is in the display state,
        # which usually means that there is an input box focused.
        return "mInputShown=true" in output

    def input_text(self, text: str, smart_enter=True, clear_first=True):
        """
        Input text with enhanced error handling for ADB Keyboard issues.
        
        Args:
            text: Text to input
            smart_enter: Whether to automatically execute enter/search actions
            clear_first: Whether to clear the input field before typing
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.u2d.send_keys(text, clear=clear_first)
                break  # Success, exit retry loop
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"Input text attempt {attempt + 1} failed: {error_msg}")
                
                # Handle ADB_KEYBOARD_CLEAR_TEXT specific error
                if "ADB_KEYBOARD_CLEAR_TEXT failed" in error_msg and clear_first:
                    # First retry: try without clearing
                    if attempt == 0:
                        try:
                            self.logger.info("Retrying input without clearing first")
                            self.u2d.send_keys(text, clear=False)
                            break
                        except Exception as retry_e:
                            self.logger.warning(f"Retry without clear failed: {retry_e}")
                            continue
                    
                    # Second retry: manual clear then input
                    elif attempt == 1:
                        try:
                            self.logger.info("Attempting manual text selection and deletion")
                            # Try to select all text and delete
                            self.u2d.send_action("select_all")
                            time.sleep(0.1)
                            self.u2d.press("del")
                            time.sleep(0.1)
                            self.u2d.send_keys(text, clear=False)
                            break
                        except Exception as manual_e:
                            self.logger.warning(f"Manual clear failed: {manual_e}")
                            continue
                
                # For other errors or final retry
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to input text '{text}' after {max_retries} attempts. "
                                    f"Last error: {error_msg}")
                    # Don't raise exception to avoid breaking the exploration flow
                    # Just log the error and continue
                    return
                
                # Wait before next retry
                time.sleep(0.5)

        if smart_enter:
            try:
                self.u2d.send_action()
                # Automatically execute enter,
                # search and other commands according to the needs of the input box,
                # Added in version 3.1
            except Exception as e:
                self.logger.warning(f"Smart enter action failed: {e}")
                # Continue without smart enter if it fails

    def enter(self):
        self.u2d.press("enter")

    def home(self):
        self.u2d.press("home")

    def back(self):
        self.u2d.press("back")

    def _get_ui_elements(
        self, exclude_invisible_elements: bool = True
    ) -> list[UIElement]:
        """Get the current UI elements from the device.

        Args:
            exclude_invisible_elements: If True, invisible elements will be excluded from the result.
        Returns:
            list[UIElement] - The extracted UI elements.
        """
        return xml_dump_to_ui_elements(
            self.get_viewhierachy(),
            exclude_invisible_elements=exclude_invisible_elements,
            screen_size=self.get_screen_size(),
            screenshot=self.get_screenshot(),
        )

    def wait_to_stabilize(
        self,
        stability_threshold: int = 3,
        sleep_duration: float = 0.5,
        timeout: float = 6.0,
    ) -> list[UIElement]:
        """Checks if the UI elements remain stable over a number of checks and returns the state.

        Args:
            stability_threshold: Number of consecutive checks where UI elements must
            remain the same to consider UI stable.
            sleep_duration: Minimum time in seconds between each check.
            timeout: Maximum time in seconds to wait for UI to become stable before
            giving up.

        Returns:
            The current state of the UI if stability is achieved within the timeout.
        """
        if not self._prior_ui_elements_state:
            self._prior_ui_elements_state = self._get_ui_elements()
        if stability_threshold <= 0:
            raise ValueError("Stability threshold must be a positive integer.")

        stable_checks = 1
        start_time = time.time()
        deadline = start_time + timeout
        current_ui_elements_state = []
        while stable_checks < stability_threshold and time.time() < deadline:
            iteration_start_time = time.time()
            current_ui_elements_state = self._get_ui_elements()

            if self._prior_ui_elements_state == current_ui_elements_state:
                stable_checks += 1
                if stable_checks == stability_threshold:
                    break  # Exit early if stability is achieved.
            else:
                stable_checks = 1  # Reset if any change is detected
                self._prior_ui_elements_state = current_ui_elements_state

            elapsed_time = time.time() - iteration_start_time
            remaining_sleep = sleep_duration - elapsed_time
            if remaining_sleep > 0:
                sleep_time = min(remaining_sleep, deadline - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
            # If remaining_sleep <= 0, proceed immediately to the next iteration
        return current_ui_elements_state

    def get_orientation(self) -> int:
        """Returns the current screen orientation.
        0: natural, 1: left, 2: right, 3: upside down.
        """
        # return self.u2d.orientation #natural, left, right, upsidedown
        return self.u2d.info.get("displayRotation", 0)  # 0, 1, 2, 3

    def get_physical_frame_boundary(self) -> tuple[int, int, int, int]:
        """Returns the physical frame boundary.

        Returns:
            First two integers are the coordinates for top left corner, last two are for
            lower right corner. All coordinates are given in portrait orientation.
        """
        response = self.run_shell_command("dumpsys input | grep physicalFrame")
        if response.exit_code == 0:
            raw_output = response.output
            pattern = r"physicalFrame=\[(\d+), (\d+), (\d+), (\d+)\]"
            matches = re.findall(pattern, raw_output)
            orientation = self.get_orientation()
            for m in matches:
                if (
                    int(m[0]) == 0
                    and int(m[1]) == 0
                    and int(m[2]) == 0
                    and int(m[3]) == 0
                ):
                    continue
                if orientation == 0 or orientation == 2:
                    return (int(m[0]), int(m[1]), int(m[2]), int(m[3]))
                return (int(m[1]), int(m[0]), int(m[3]), int(m[2]))
        raise ValueError("Failed to get physical frame boundary.")


def is_element_available(ele: UIElement) -> bool:
    """
    Check if the UI element is available for interaction.
    """
    return (
        ele.is_clickable
        or ele.is_scrollable
        or ele.is_long_clickable
        or ele.is_editable
    )
