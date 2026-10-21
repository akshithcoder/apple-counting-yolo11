"""Conversion utilities for MinneApple instance masks and YOLO boxes."""

from typing import TypeAlias

import numpy as np

Box: TypeAlias = tuple[float, float, float, float]


def mask_to_boxes(mask: np.ndarray) -> list[Box]:
    """Return one ``(left, top, width, height)`` box per nonzero instance ID."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D instance mask, received shape {mask.shape}.")

    boxes: list[Box] = []
    for instance_id in np.unique(mask):
        if instance_id == 0:
            continue
        rows, columns = np.where(mask == instance_id)
        left, right = int(columns.min()), int(columns.max())
        top, bottom = int(rows.min()), int(rows.max())
        # Pixel bounds are inclusive, so one pixel is added to each extent.
        boxes.append((left, top, right - left + 1, bottom - top + 1))
    return boxes


def to_yolo_coordinates(
    box: Box,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert a pixel ``(left, top, width, height)`` box to normalized YOLO form."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    left, top, width, height = box
    if left < 0 or top < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid pixel box: {box}.")
    if left + width > image_width or top + height > image_height:
        raise ValueError(f"Box {box} lies outside a {image_width}x{image_height} image.")

    return (
        (left + width / 2) / image_width,
        (top + height / 2) / image_height,
        width / image_width,
        height / image_height,
    )


def to_yolo_line(
    box: Box,
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> str:
    """Serialize a pixel box as one six-decimal YOLO label line."""
    x_center, y_center, width, height = to_yolo_coordinates(
        box,
        image_width,
        image_height,
    )
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
