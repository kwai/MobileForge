from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional


def stitch_images_vertically(image_paths: List[str]) -> Image:
    """
    Stitches a list of images together vertically.

    Args:
        image_paths (List[str]): A list of paths to the images to stitch.

    Returns:
        Image: A PIL Image object of the stitched images.
    """
    images = [Image.open(p) for p in image_paths]
    if not images:
        raise ValueError("Image path list is empty")

    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    return new_im


def stitch_images_horizontally(
    image_paths: List[str], labels: Optional[List[str]] = None
) -> Image:
    """
    Stitches a list of images together horizontally and adds optional labels.

    Args:
        image_paths (List[str]): A list of paths to the images to stitch.
        labels (Optional[List[str]]): A list of labels to draw on top of each image.

    Returns:
        Image: A PIL Image object of the stitched images with labels.
    """
    images = [Image.open(p) for p in image_paths]
    if not images:
        raise ValueError("Image path list is empty")

    if labels and len(images) != len(labels):
        raise ValueError("Number of images and labels must be the same.")

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    # Add some space at the top for the labels
    label_height = 80 if labels else 0
    new_im = Image.new("RGB", (total_width, max_height + label_height), "white")
    draw = ImageDraw.Draw(new_im)

    try:
        # Use a truetype font if available, otherwise default.
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=60)
    except IOError:
        font = ImageFont.load_default(size=40)

    x_offset = 0
    for i, im in enumerate(images):
        new_im.paste(im, (x_offset, label_height))
        if labels:
            label = labels[i]
            # Calculate text size and position
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x_offset + (im.width - text_width) // 2
            text_y = 10
            draw.text((text_x, text_y), label, fill="black", font=font)
        x_offset += im.size[0]

    return new_im
