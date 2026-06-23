from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import cv2
import os
import json
import math
import textwrap


def visualize_click_opencv(
    img_np: np.ndarray,
    click_pos: Tuple[int, int],
    circle_radius: int = 35,
    square_size: int = 100,
    alpha: float = 0.7,
) -> Image:
    """
    This function visualizes the click position with a red circle and a green square frame,
    and labels it with a "C" at the top-right corner of the square.
    """
    # Convert RGB to BGR for OpenCV processing
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Create an overlay image
    overlay = img.copy()
    x, y = click_pos

    # Draw a red circle with a white border. OpenCV uses BGR, so red is (0, 0, 255)
    cv2.circle(overlay, (x, y), circle_radius, (0, 0, 255), 6)

    # Draw a green square frame around the red circle
    top_left = (x - square_size, y - square_size)
    bottom_right = (x + square_size, y + square_size)
    cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), 8)

    # Add the "C" label in the top-right corner of the green square
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "C"
    font_scale = 1.5
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = bottom_right[0] - text_size[0] - 10
    text_y = top_left[1] + text_size[1] + 10

    # Draw the "C" label with a black background and white text
    cv2.rectangle(
        overlay,
        (text_x - 10, text_y - text_size[1] - 10),
        (text_x + text_size[0] + 10, text_y + 10),
        (0, 0, 0),
        -1,
    )  # Black background
    cv2.putText(
        overlay,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )

    # Combine the overlay with the base image using alpha transparency
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Convert the final result back to RGB format for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_rgb)


def create_frame(image: Image, label: str) -> Image:
    """创建带边框和标签的子图"""
    BORDER_WIDTH = 14  # 设置边框宽度
    LABEL_HEIGHT = 140  # 增加标签高度，避免与图像重叠
    frame_w = image.width + BORDER_WIDTH * 2
    frame_h = image.height + BORDER_WIDTH * 2 + LABEL_HEIGHT  # 包括标签的高度

    # 创建画布（白色背景）
    frame = Image.new("RGB", (frame_w, frame_h), "white")
    draw = ImageDraw.Draw(frame)

    # 绘制四周的边框
    draw.rectangle(
        [
            (BORDER_WIDTH, LABEL_HEIGHT),
            (frame_w - BORDER_WIDTH, frame_h - BORDER_WIDTH),
        ],
        outline="#808080",  # 灰色边框
        width=BORDER_WIDTH,
    )

    # 将原始图像粘贴到框架中
    frame.paste(image, (BORDER_WIDTH, LABEL_HEIGHT + BORDER_WIDTH))

    # 添加标签文本
    try:
        font = ImageFont.truetype("Arial.ttf", 120)
    except:
        font = ImageFont.load_default(120)

    # 标签文本居中
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_x = (frame_w - (text_bbox[2] - text_bbox[0])) // 2
    draw.text((text_x, 10), label, fill="#404040", font=font)

    return frame


def _calculate_characters_per_line(image_width, font):
    # Measure the width of a sample of characters
    sample_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    total_width = sum(
        font.getbbox(char)[2] - font.getbbox(char)[0] for char in sample_text
    )
    if total_width == 0:
        return 20  # Fallback
    average_char_width = total_width / len(sample_text)

    # Calculate the number of characters that fit in one line
    characters_per_line = image_width // average_char_width
    return int(characters_per_line)


def _add_strip_with_text(
    image: Image, text_segments: list, font_size=60, line_spacing=10
) -> Image:
    width, height = image.size

    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default(size=font_size)

    # --- New logic to handle multi-color lines ---

    # 1. Group segments into logical lines
    logical_lines = []
    current_line_segments = []
    for segment in text_segments:
        text = segment["text"]
        if text.startswith("\n"):
            if current_line_segments:
                logical_lines.append(current_line_segments)
            current_line_segments = [
                {"text": text.lstrip("\n"), "color": segment["color"]}
            ]
        else:
            current_line_segments.append(segment)
    if current_line_segments:
        logical_lines.append(current_line_segments)

    # 2. Calculate layout for all lines
    characters_per_line = _calculate_characters_per_line(width, font)
    all_wrapped_lines = []
    total_text_height = 0
    dummy_draw = ImageDraw.Draw(image)

    for line_segments in logical_lines:
        full_line_text = "".join([s["text"] for s in line_segments])
        wrapped_lines = textwrap.wrap(full_line_text, width=characters_per_line)
        all_wrapped_lines.append(wrapped_lines)
        for line in wrapped_lines:
            text_bbox = dummy_draw.textbbox((0, 0), line, font=font)
            total_text_height += text_bbox[3] - text_bbox[1] + line_spacing

    # 3. Create the new image with the correctly sized strip
    strip_height = total_text_height + 20  # Add padding
    blue_line_height = 10
    new_height = height + blue_line_height + strip_height
    new_image = Image.new("RGB", (width, int(new_height)), "white")
    new_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_image)

    # 4. Draw the blue line and the text
    draw.rectangle([(0, height), (width, height + blue_line_height)], fill="blue")
    y_text = height + blue_line_height + 10

    for i, wrapped_lines in enumerate(all_wrapped_lines):
        line_segments = logical_lines[i]
        char_cursor = 0
        full_line_text = "".join([s["text"] for s in line_segments])

        for line_text in wrapped_lines:
            # Center the line
            line_bbox = draw.textbbox((0, 0), line_text, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            x_text = (width - line_width) // 2

            # Draw each part of the line with its color
            line_char_ptr = 0
            while line_char_ptr < len(line_text):
                # Find which segment the global cursor is in
                segment_start_char = 0
                for segment in line_segments:
                    segment_end_char = segment_start_char + len(segment["text"])
                    if char_cursor < segment_end_char:
                        # This is the correct segment
                        chars_to_draw_from_segment = segment_end_char - char_cursor
                        chars_to_draw_on_line = len(line_text) - line_char_ptr

                        draw_len = min(
                            chars_to_draw_from_segment, chars_to_draw_on_line
                        )
                        text_piece = line_text[line_char_ptr : line_char_ptr + draw_len]

                        draw.text(
                            (x_text, y_text),
                            text_piece,
                            font=font,
                            fill=segment["color"],
                        )

                        piece_bbox = draw.textbbox((0, 0), text_piece, font=font)
                        x_text += piece_bbox[2] - piece_bbox[0]

                        char_cursor += draw_len
                        line_char_ptr += draw_len

                    segment_start_char = segment_end_char

            line_height_bbox = draw.textbbox((0, 0), line_text, font=font)
            y_text += line_height_bbox[3] - line_height_bbox[1] + line_spacing

    return new_image


def _wrap_action_text(action, action_detail):
    text_segments = [
        {"text": "Action:", "color": "red"},
        {"text": f" {action}", "color": "black"},
        {"text": "\nDetail:", "color": "red"},
        {"text": f" {action_detail}", "color": "black"},
    ]
    return text_segments


def combine_images(
    before_img: np.ndarray,
    after_img: np.ndarray,
    action_type: str,
    action_detail: str,
    click_position: Optional[Tuple[int, int]] = None,
) -> Image:
    """
    优化版图像拼接函数，确保标签、边框和描述不重叠
    """

    GAP_WIDTH = 40  # 图片之间的间隙

    # 将 numpy 数组转换为 PIL Image
    if click_position is not None:
        before_base = visualize_click_opencv(before_img, click_position)
    else:
        before_base = Image.fromarray(before_img)

    after_base = Image.fromarray(after_img)

    # 创建带边框和标签的子图
    before_frame = create_frame(before_base, "Before Action")
    after_frame = create_frame(after_base, "After Action")

    # 计算拼接的总宽度和最大高度
    total_width = before_frame.width + after_frame.width + GAP_WIDTH
    max_height = max(before_frame.height, after_frame.height)

    # 创建一个不带底部文字区域的临时画布
    combined_temp = Image.new("RGB", (total_width, max_height), "white")

    # 拼接两个子图
    combined_temp.paste(before_frame, (0, 0))
    combined_temp.paste(after_frame, (before_frame.width + GAP_WIDTH, 0))

    # 在底部添加带格式的动作描述
    text_segments = _wrap_action_text(action_type, action_detail)
    combined_final = _add_strip_with_text(combined_temp, text_segments)

    return combined_final


def _create_puzzle_layout(image_folder: str, task_title: str, output_path: str):
    """
    将文件夹中的所有图片合并成一张大图。
    """
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found: {image_folder}")
        return

    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(".png")],
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )
    if not image_files:
        print(f"Error: No images found in {image_folder}")
        return

    # Load first image to get dimensions
    first_image = Image.open(os.path.join(image_folder, image_files[0]))
    img_width, img_height = first_image.size
    first_image.close()

    # Calculate layout
    num_images = len(image_files)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # Font settings
    try:
        title_font = ImageFont.truetype("Arial.ttf", 80)
        number_font = ImageFont.truetype("Arial.ttf", 60)
    except IOError:
        title_font = ImageFont.load_default(size=80)
        number_font = ImageFont.load_default(size=60)

    # Calculate title height
    task_title_text = task_title
    temp_img = Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    # Set max title width to be slightly less than the image content area
    max_title_width = cols * img_width * 0.95

    # --- New robust text wrapping logic ---
    words = task_title_text.split(" ")
    wrapped_text = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        # Use textlength for more accurate width calculation
        line_width = temp_draw.textlength(test_line, font=title_font)
        if line_width <= max_title_width:
            current_line = test_line
        else:
            wrapped_text.append(current_line)
            current_line = word
    if current_line:
        wrapped_text.append(current_line)
    # --- End of new logic ---

    line_height = title_font.getbbox("A")[3] - title_font.getbbox("A")[1]
    top_padding, bottom_padding, line_spacing = 60, 60, 30
    title_height = (
        (len(wrapped_text) * line_height)
        + ((len(wrapped_text) - 1) * line_spacing if len(wrapped_text) > 1 else 0)
        + top_padding
        + bottom_padding
    )

    # Create puzzle canvas with horizontal padding for the title
    horizontal_padding = int(cols * img_width * 0.025)
    puzzle_width = cols * img_width + horizontal_padding * 2
    puzzle_height = rows * img_height + title_height
    puzzle_image = Image.new("RGB", (int(puzzle_width), int(puzzle_height)), "white")
    draw = ImageDraw.Draw(puzzle_image)

    # Draw title centered within the full canvas width
    y_position = top_padding
    for line in wrapped_text:
        line_bbox = draw.textbbox((0, 0), line, font=title_font)
        line_width = line_bbox[2] - line_bbox[0]
        x_position = (puzzle_width - line_width) // 2
        draw.text((x_position, y_position), line, font=title_font, fill="black")
        y_position += line_height + line_spacing

    # Paste images
    for i, image_file in enumerate(image_files):
        row, col = i // cols, i % cols
        x_base = col * img_width + horizontal_padding
        y_base = row * img_height + title_height
        img = Image.open(os.path.join(image_folder, image_file))
        puzzle_image.paste(img, (int(x_base), int(y_base)))
        img.close()

        # Add number to top-left corner with taller background
        number_text = str(i + 1)
        number_bbox = draw.textbbox((0, 0), number_text, font=number_font)
        text_w = number_bbox[2] - number_bbox[0]
        text_h = number_bbox[3] - number_bbox[1]
        bg_x_padding = 10
        bg_y_padding = 12  # Increased vertical padding
        bg_rect = (
            x_base + 5,
            y_base + 5,
            x_base + 5 + text_w + bg_x_padding * 2,
            y_base + 15 + text_h + bg_y_padding * 2,
        )
        overlay = Image.new("RGBA", puzzle_image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(bg_rect, fill=(255, 255, 255, 180))
        puzzle_image = Image.alpha_composite(
            puzzle_image.convert("RGBA"), overlay
        ).convert("RGB")
        draw = ImageDraw.Draw(puzzle_image)
        draw.text(
            (x_base + 5 + bg_x_padding, y_base + 5 + bg_y_padding),
            number_text,
            font=number_font,
            fill="red",
        )

    # Save puzzle image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    puzzle_image.save(output_path)
    # print(f"Puzzle image saved to {output_path}")


def _add_llm_description_to_image(
    image: Image, action_text: str, ui_text: str
) -> Image:
    """Helper to draw LLM descriptions below an image using a small font."""
    try:
        font = ImageFont.truetype("Arial.ttf", 28)  # Smaller font
    except IOError:
        font = ImageFont.load_default(size=28)

    image_width, image_height = image.size
    padding = 20  # Horizontal padding for text

    # Prepare text
    action_text = "Action: " + (action_text or "N/A")
    ui_text = "UI: " + (ui_text or "N/A")

    # Wrap text using a more robust average character width calculation
    temp_draw = ImageDraw.Draw(image)
    sample_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    try:
        # Get the bounding box for the entire sample text for better accuracy
        total_width = temp_draw.textlength(sample_text, font=font)
        avg_char_width = total_width / len(sample_text)
    except Exception:
        # Fallback for older PIL versions
        total_width = sum(font.getbbox(c)[2] for c in sample_text)
        avg_char_width = total_width / len(sample_text)

    if avg_char_width > 0:
        chars_per_line = int((image_width - padding * 2) / avg_char_width)
    else:  # Fallback
        chars_per_line = 50

    action_lines = textwrap.wrap(action_text, width=chars_per_line)
    ui_lines = textwrap.wrap(ui_text, width=chars_per_line)

    # Calculate height needed for the text strip
    line_height = (
        temp_draw.textbbox((0, 0), "A", font=font)[3]
        - temp_draw.textbbox((0, 0), "A", font=font)[1]
    )
    line_spacing = 8
    vertical_padding = 15

    strip_height = (len(action_lines) + len(ui_lines)) * (
        line_height + line_spacing
    ) + vertical_padding * 2

    # Create new image with space for the strip
    new_image = Image.new(
        "RGB", (image_width, image_height + int(strip_height)), "white"
    )
    new_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_image)

    # Draw the text
    y_text = image_height + vertical_padding
    for line in action_lines:
        draw.text((padding, y_text), line, font=font, fill="red")
        y_text += line_height + line_spacing

    for line in ui_lines:
        draw.text((padding, y_text), line, font=font, fill="black")
        y_text += line_height + line_spacing

    return new_image


def create_llm_puzzle(
    log_dir: str,
    task_identifier: str,
    task_description: str,
    llm_step_descriptions: dict,  # Changed to dict
    log_data: list,
):
    """
    Creates single-step images with LLM descriptions and combines them into a puzzle.
    """
    # Create output directory
    llm_actions_output_dir = os.path.join(log_dir, "llm_described_actions")
    os.makedirs(llm_actions_output_dir, exist_ok=True)

    for step_data in log_data:
        step_num = step_data.get("step")
        if not step_num:
            continue

        # Use the raw screenshot before the action as the base
        raw_img_path = os.path.join(log_dir, f"{step_num - 1}.png")
        if not os.path.exists(raw_img_path):
            continue

        # Get the LLM description for this specific step from the dictionary
        desc_obj = llm_step_descriptions.get(step_num)
        if not desc_obj:  # Skip if no description was generated for this step
            continue

        raw_img_np = np.array(Image.open(raw_img_path).convert("RGB"))

        # Check for click position in log data to add visualization
        action_details = step_data.get("action", [])
        click_position = None
        if (
            action_details
            and isinstance(action_details, list)
            and len(action_details) > 1
            and isinstance(action_details[1], dict)
        ):
            if action_details[1].get("detail_type") == "coordinates":
                detail = action_details[1].get("detail")
                if isinstance(detail, list) and len(detail) == 2:
                    click_position = tuple(detail)

        # Create base image with click visualization on the raw screenshot
        if click_position:
            base_img = visualize_click_opencv(raw_img_np, click_position)
        else:
            base_img = Image.fromarray(raw_img_np)

        # Get text from the description object
        action_text = desc_obj.get("action_description", "N/A")
        ui_text = desc_obj.get("ui_description", "N/A")

        # Add LLM text to the base image
        final_img = _add_llm_description_to_image(base_img, action_text, ui_text)

        # Save the new image
        output_path = os.path.join(llm_actions_output_dir, f"step_{step_num}.png")
        final_img.save(output_path)
        print(f"Saved LLM-described image for step {step_num} to {output_path}")

    # Create the puzzle from the new directory of images
    task_info = f"{task_identifier}: {task_description} (LLM Analysis)"
    puzzle_output_path = os.path.join(log_dir, "puzzle", "puzzle_llm.png")

    if os.path.exists(llm_actions_output_dir) and os.listdir(llm_actions_output_dir):
        print("Creating LLM-based puzzle image...")
        _create_puzzle_layout(
            image_folder=llm_actions_output_dir,
            task_title=task_info,
            output_path=puzzle_output_path,
        )


def visualize_single_action(
    img: np.ndarray,
    action_type: str,
    action_detail: str,
    click_position: Optional[Tuple[int, int]] = None,
) -> Image:
    """
    为单个动作生成可视化图像，包括点击位置标记和动作描述。

    Args:
        img: 输入图像数组
        action_type: 动作类型
        action_detail: 动作详情
        click_position: 可选的点击位置坐标

    Returns:
        带有动作标记和描述的PIL Image
    """
    # 将numpy数组转换为PIL Image
    if click_position is not None:
        base_img = visualize_click_opencv(img, click_position)
    else:
        base_img = Image.fromarray(img)

    # 添加动作描述
    text_segments = _wrap_action_text(action_type, action_detail)
    final_img = _add_strip_with_text(base_img, text_segments)

    return final_img


def visualize_and_save_actions(
    log_dir: str, task_identifier: str, task_description: str
):
    """
    读取日志文件，为每个动作生成可视化，并保存结果。

    Args:
        log_dir: 包含log.json和截图的目录路径
        task_identifier: 任务标识符
        task_description: 任务描述
    """
    log_file_path = os.path.join(log_dir, "log.json")
    if not os.path.exists(log_file_path):
        print(f"Error: log.json not found in {log_dir}")
        return

    with open(log_file_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    # 创建两个输出目录
    actions_output_dir = os.path.join(log_dir, "visualize_actions")
    single_actions_output_dir = os.path.join(log_dir, "single_actions")
    os.makedirs(actions_output_dir, exist_ok=True)
    os.makedirs(single_actions_output_dir, exist_ok=True)

    # 处理所有步骤
    for step_data in log_data:
        step = step_data.get("step")
        if not step:
            continue

        # print(f"Processing step {step}...")

        # The screenshot *before* the action of the current step
        before_img_path = os.path.join(log_dir, f"{step - 1}.png")
        if not os.path.exists(before_img_path):
            print(f"Warning: Screenshot for step {step} not found. Skipping.")
            continue

        try:
            before_img_np = np.array(Image.open(before_img_path).convert("RGB"))
        except Exception as e:
            print(f"Error loading image for step {step}: {e}. Skipping.")
            continue

        # 提取动作信息
        action_details = step_data.get("action", [])
        click_position = None
        action_type_str = "N/A"
        action_detail_str = "N/A"

        if (
            action_details
            and isinstance(action_details, list)
            and len(action_details) > 0
        ):
            action_type_str = action_details[0]
            if len(action_details) > 1 and isinstance(action_details[1], dict):
                detail_type = action_details[1].get("detail_type")
                detail = action_details[1].get("detail")
                if detail_type == "coordinates":
                    action_detail_str = "Click position is marked with a red circle."
                    if isinstance(detail, list) and len(detail) == 2:
                        click_position = tuple(detail)
                else:
                    action_detail_str = str(detail)

        # --- 生成 `single_actions` 图片 (基于log.json) ---
        single_action_img = visualize_single_action(
            before_img_np, action_type_str, action_detail_str, click_position
        )
        single_action_path = os.path.join(single_actions_output_dir, f"step_{step}.png")
        single_action_img.save(single_action_path)
        # print(
        #     f"Saved single action visualization for step {step} to {single_action_path}"
        # )

        # --- 生成 `visualize_actions` 图片 (给LLM分析) ---
        # The screenshot *after* the action. If it doesn't exist, it's the last step.
        after_img_path = os.path.join(log_dir, f"{step}.png")
        if not os.path.exists(after_img_path):
            # print(
            #     f"  - Step {step} appears to be the last action. Skipping before/after visualization as no 'after' screenshot exists."
            # )
            continue  # Skip creating a before/after image for the last step

        after_img_np = np.array(Image.open(after_img_path).convert("RGB"))

        try:
            combined_img = combine_images(
                before_img=before_img_np,
                after_img=after_img_np,
                action_type=action_type_str,
                action_detail=action_detail_str,
                click_position=click_position,
            )
            output_path = os.path.join(actions_output_dir, f"step_{step}.png")
            combined_img.save(output_path)
            # print(f"Saved before/after visualization for step {step} to {output_path}")
        except Exception as e:
            print(f"Error creating before/after visualization for step {step}: {e}")

    # 使用单个动作的可视化创建原始puzzle图
    task_info = f"{task_identifier}: {task_description}"
    # print("Creating puzzle image from log.json...")
    puzzle_path = os.path.join(log_dir, "puzzle", "puzzle.png")
    if os.path.exists(single_actions_output_dir) and os.listdir(
        single_actions_output_dir
    ):
        _create_puzzle_layout(
            image_folder=single_actions_output_dir,
            task_title=task_info,
            output_path=puzzle_path,
        )

    return log_data
