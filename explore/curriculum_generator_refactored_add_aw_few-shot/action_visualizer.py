"""
动作可视化器模块

负责生成带有动作标记和步骤序号的可视化截图
基于reference/mobilegym_critic/utils/visualize_actions.py中的visualize_single_action函数
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2


class ActionVisualizer:
    """动作可视化器类"""
    
    def __init__(self):
        """初始化动作可视化器"""
        self.circle_radius = 35
        self.square_size = 100
        self.alpha = 0.7
    
    def create_visualized_screenshots(self, trajectory_data: Dict[str, Any], 
                                    screenshot_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        为轨迹创建可视化截图
        
        Args:
            trajectory_data: 轨迹数据
            screenshot_info: 截图信息
            
        Returns:
            可视化截图信息列表
        """
        visualized_screenshots = []
        steps = trajectory_data.get("steps", [])
        screenshot_steps = screenshot_info.get("steps", {})
        
        for i, step in enumerate(steps):
            step_index = step.get("step_index", i)
            
            # 获取对应的截图
            step_screenshot_info = screenshot_steps.get(step_index)
            if not step_screenshot_info:
                print(f"步骤 {step_index} 没有对应的截图")
                continue
            
            before_screenshot_path = step_screenshot_info["before_screenshot"]
            
            # 加载原始截图
            try:
                original_image = Image.open(before_screenshot_path).convert("RGB")
                img_np = np.array(original_image)
            except Exception as e:
                print(f"加载截图失败 {before_screenshot_path}: {e}")
                continue
            
            # 提取动作信息
            action_type, action_detail, click_position = self._extract_action_info(step)
            
            # 生成可视化图像
            visualized_image = self._visualize_single_action(
                img_np, action_type, action_detail, click_position, step_index
            )
            
            # 构建可视化截图信息
            visualized_info = {
                "step_index": step_index,
                "original_screenshot_path": str(before_screenshot_path),
                "visualized_image": visualized_image,
                "action_type": action_type,
                "action_detail": action_detail,
                "click_position": click_position,
                "step_summary": step.get("summary", "")
            }
            
            visualized_screenshots.append(visualized_info)
        
        return visualized_screenshots
    
    def _extract_action_info(self, step: Dict[str, Any]) -> Tuple[str, str, Optional[Tuple[int, int]]]:
        """
        从步骤数据中提取动作信息
        
        Args:
            step: 步骤数据
            
        Returns:
            动作类型、动作详情、点击位置
        """
        metadata = step.get("metadata", {})
        
        # 提取动作坐标
        actual_coordinates = metadata.get("actual_action_coordinates")
        click_position = None
        if actual_coordinates and len(actual_coordinates) == 2:
            click_position = tuple(actual_coordinates)
        
        # 提取动作类型和详情
        summary = step.get("summary", "")
        
        # 根据summary解析动作类型
        action_type = "unknown"
        if "click" in summary.lower():
            action_type = "click"
        elif "swipe" in summary.lower():
            action_type = "swipe"
        elif "type" in summary.lower() or "input" in summary.lower():
            action_type = "type"
        elif "scroll" in summary.lower():
            action_type = "scroll"
        elif "complete" in summary.lower():
            action_type = "complete"
        
        # 构建动作详情
        if click_position:
            action_detail = f"Action performed at coordinates {click_position}"
        else:
            action_detail = summary if summary else "Action performed"
        
        return action_type, action_detail, click_position
    
    def _visualize_single_action(self, img_np: np.ndarray, action_type: str, 
                               action_detail: str, click_position: Optional[Tuple[int, int]] = None,
                               step_number: int = 0) -> Image.Image:
        """
        为单个动作生成可视化图像，包括点击位置标记和动作描述
        
        Args:
            img_np: 输入图像数组
            action_type: 动作类型
            action_detail: 动作详情
            click_position: 可选的点击位置坐标
            step_number: 步骤序号
            
        Returns:
            带有动作标记和描述的PIL Image
        """
        # 如果有点击位置，添加可视化标记
        if click_position is not None:
            base_img = self._visualize_click_opencv(img_np, click_position)
        else:
            base_img = Image.fromarray(img_np)
        
        # 添加步骤序号
        base_img = self._add_step_number(base_img, step_number)
        
        # 添加动作描述
        text_segments = self._wrap_action_text(action_type, action_detail)
        final_img = self._add_strip_with_text(base_img, text_segments)
        
        return final_img
    
    def _visualize_click_opencv(self, img_np: np.ndarray, 
                              click_pos: Tuple[int, int]) -> Image.Image:
        """
        使用OpenCV在图像上可视化点击位置
        
        Args:
            img_np: 图像数组
            click_pos: 点击位置坐标
            
        Returns:
            带有点击标记的PIL Image
        """
        # 转换RGB到BGR用于OpenCV处理
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 创建叠加层
        overlay = img.copy()
        x, y = click_pos
        
        # 绘制红色圆圈，OpenCV使用BGR，所以红色是(0, 0, 255)
        cv2.circle(overlay, (x, y), self.circle_radius, (0, 0, 255), 6)
        
        # 绘制绿色方框
        top_left = (x - self.square_size, y - self.square_size)
        bottom_right = (x + self.square_size, y + self.square_size)
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), 8)
        
        # 在方框右上角添加"C"标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "C"
        font_scale = 1.5
        font_thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = bottom_right[0] - text_size[0] - 10
        text_y = top_left[1] + text_size[1] + 10
        
        # 绘制带黑色背景的白色"C"标签
        cv2.rectangle(
            overlay,
            (text_x - 10, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 10, text_y + 10),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            overlay,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )
        
        # 使用alpha透明度合并叠加层和基础图像
        img = cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0)
        
        # 转换回RGB格式用于PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(img_rgb)
    
    def _add_step_number(self, image: Image.Image, step_number: int) -> Image.Image:
        """
        在图像左上角添加步骤序号
        
        Args:
            image: 输入图像
            step_number: 步骤序号
            
        Returns:
            带有步骤序号的图像
        """
        draw = ImageDraw.Draw(image)
        
        # 设置字体
        try:
            font = ImageFont.truetype("Arial.ttf", 60)
        except:
            font = ImageFont.load_default(size=60)
        
        # 步骤文本
        step_text = f"Step {step_number}"
        
        # 计算文本尺寸
        text_bbox = draw.textbbox((0, 0), step_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # 设置背景矩形参数
        bg_x_padding = 15
        bg_y_padding = 10
        bg_rect = (
            5,
            5,
            5 + text_w + bg_x_padding * 2,
            5 + text_h + bg_y_padding * 2,
        )
        
        # 创建半透明背景
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(bg_rect, fill=(255, 255, 255, 200))
        
        # 合并图像
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # 绘制步骤序号文本
        draw.text(
            (5 + bg_x_padding, 5 + bg_y_padding),
            step_text,
            font=font,
            fill="red",
        )
        
        return image
    
    def _wrap_action_text(self, action: str, action_detail: str) -> List[Dict[str, str]]:
        """
        包装动作文本为带颜色的段落
        
        Args:
            action: 动作类型
            action_detail: 动作详情
            
        Returns:
            文本段落列表
        """
        text_segments = [
            {"text": "Action:", "color": "red"},
            {"text": f" {action}", "color": "black"},
            {"text": "\nDetail:", "color": "red"},
            {"text": f" {action_detail}", "color": "black"},
        ]
        return text_segments
    
    def _add_strip_with_text(self, image: Image.Image, 
                           text_segments: List[Dict[str, str]], 
                           font_size: int = 50, line_spacing: int = 8) -> Image.Image:
        """
        在图像底部添加带格式文本条，确保文字不超出边界
        
        Args:
            image: 输入图像
            text_segments: 文本段落列表
            font_size: 字体大小
            line_spacing: 行间距
            
        Returns:
            带有文本条的图像
        """
        width, height = image.size
        
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default(size=font_size)
        
        # 动态调整字体大小以适应图像宽度
        max_width = width - 40  # 保留左右边距
        dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
        
        # 将段落组合成逻辑行
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
        
        # 检查并自动换行过长的文本
        wrapped_lines = []
        for line_segments in logical_lines:
            full_line_text = "".join([s["text"] for s in line_segments])
            
            # 检查文本宽度
            text_bbox = dummy_draw.textbbox((0, 0), full_line_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            
            if text_width <= max_width:
                # 文本适合一行
                wrapped_lines.append(line_segments)
            else:
                # 需要自动换行
                wrapped_segments = self._wrap_text_segments(line_segments, font, max_width, dummy_draw)
                wrapped_lines.extend(wrapped_segments)
        
        # 计算总文本高度
        total_text_height = 0
        for line_segments in wrapped_lines:
            full_line_text = "".join([s["text"] for s in line_segments])
            text_bbox = dummy_draw.textbbox((0, 0), full_line_text, font=font)
            total_text_height += text_bbox[3] - text_bbox[1] + line_spacing
        
        # 创建带有文本条的新图像
        padding = 20
        strip_height = total_text_height + padding * 2
        blue_line_height = 10
        new_height = height + blue_line_height + strip_height
        new_image = Image.new("RGB", (width, int(new_height)), "white")
        new_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(new_image)
        
        # 绘制蓝色分隔线
        draw.rectangle([(0, height), (width, height + blue_line_height)], fill="blue")
        y_text = height + blue_line_height + padding
        
        # 绘制文本
        for line_segments in wrapped_lines:
            full_line_text = "".join([s["text"] for s in line_segments])
            
            # 计算居中位置
            line_bbox = draw.textbbox((0, 0), full_line_text, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            x_text = max(20, (width - line_width) // 2)  # 确保不小于左边距
            
            # 绘制每个段落的不同颜色
            for segment in line_segments:
                text_piece = segment["text"]
                color = segment["color"]
                
                # 确保文本不超出右边界
                piece_bbox = draw.textbbox((0, 0), text_piece, font=font)
                piece_width = piece_bbox[2] - piece_bbox[0]
                
                if x_text + piece_width > width - 20:
                    # 如果文本会超出边界，截断或调整
                    remaining_width = width - 20 - x_text
                    if remaining_width > 0:
                        # 尝试截断文本
                        truncated_text = self._truncate_text_to_fit(text_piece, font, remaining_width, draw)
                        draw.text((x_text, y_text), truncated_text, font=font, fill=color)
                    break
                else:
                    draw.text((x_text, y_text), text_piece, font=font, fill=color)
                    x_text += piece_width
            
            line_height_bbox = draw.textbbox((0, 0), full_line_text, font=font)
            y_text += line_height_bbox[3] - line_height_bbox[1] + line_spacing
        
        return new_image
    
    def _wrap_text_segments(self, line_segments: List[Dict[str, str]], font: ImageFont.ImageFont, 
                          max_width: int, draw: ImageDraw.ImageDraw) -> List[List[Dict[str, str]]]:
        """
        将过长的文本段落自动换行
        
        Args:
            line_segments: 文本段落列表
            font: 字体
            max_width: 最大宽度
            draw: 绘图对象
            
        Returns:
            换行后的文本段落列表
        """
        wrapped_lines = []
        current_line = []
        current_width = 0
        
        for segment in line_segments:
            text = segment["text"]
            color = segment["color"]
            
            # 按单词分割
            words = text.split()
            
            for word in words:
                word_with_space = word + " "
                word_bbox = draw.textbbox((0, 0), word_with_space, font=font)
                word_width = word_bbox[2] - word_bbox[0]
                
                if current_width + word_width <= max_width:
                    # 单词适合当前行
                    if current_line and current_line[-1]["color"] == color:
                        # 与前一个段落颜色相同，合并
                        current_line[-1]["text"] += word_with_space
                    else:
                        # 创建新段落
                        current_line.append({"text": word_with_space, "color": color})
                    current_width += word_width
                else:
                    # 单词不适合当前行，换行
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = [{"text": word_with_space, "color": color}]
                    current_width = word_width
        
        if current_line:
            wrapped_lines.append(current_line)
        
        return wrapped_lines
    
    def _truncate_text_to_fit(self, text: str, font: ImageFont.ImageFont, 
                            max_width: int, draw: ImageDraw.ImageDraw) -> str:
        """
        截断文本以适应指定宽度
        
        Args:
            text: 原始文本
            font: 字体
            max_width: 最大宽度
            draw: 绘图对象
            
        Returns:
            截断后的文本
        """
        if not text:
            return ""
        
        # 添加省略号的宽度
        ellipsis = "..."
        ellipsis_bbox = draw.textbbox((0, 0), ellipsis, font=font)
        ellipsis_width = ellipsis_bbox[2] - ellipsis_bbox[0]
        
        available_width = max_width - ellipsis_width
        
        if available_width <= 0:
            return ellipsis
        
        # 二分查找最佳截断位置
        left, right = 0, len(text)
        best_length = 0
        
        while left <= right:
            mid = (left + right) // 2
            test_text = text[:mid]
            test_bbox = draw.textbbox((0, 0), test_text, font=font)
            test_width = test_bbox[2] - test_bbox[0]
            
            if test_width <= available_width:
                best_length = mid
                left = mid + 1
            else:
                right = mid - 1
        
        if best_length == 0:
            return ellipsis
        
        return text[:best_length] + ellipsis
    
    def save_visualized_screenshots(self, visualized_screenshots: List[Dict[str, Any]], 
                                  output_dir: Path, trajectory_id: str) -> None:
        """
        保存可视化截图到文件
        
        Args:
            visualized_screenshots: 可视化截图列表
            output_dir: 输出目录
            trajectory_id: 轨迹ID
        """
        trajectory_output_dir = output_dir / f"visualized_{trajectory_id}"
        trajectory_output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, screenshot_info in enumerate(visualized_screenshots):
            step_index = screenshot_info["step_index"]
            visualized_image = screenshot_info["visualized_image"]
            
            # 保存图像
            output_path = trajectory_output_dir / f"step_{step_index:02d}_visualized.png"
            visualized_image.save(output_path)
            
            print(f"保存可视化截图: {output_path}")
