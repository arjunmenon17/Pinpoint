"""Spatial reasoning: region classification and direction strings."""

from __future__ import annotations

from typing import Literal

from app.schemas import (
    DistanceLabel,
    HorizontalRegion,
    SpatialGuidance,
    VerticalRegion,
)


def get_horizontal_region(center_x: float, width: float) -> HorizontalRegion:
    """Classify horizontal position into left/center/right third."""
    if width <= 0:
        return "center"
    third = center_x / width
    if third < 1 / 3:
        return "left"
    if third > 2 / 3:
        return "right"
    return "center"


def get_vertical_region(center_y: float, height: float) -> VerticalRegion:
    """Classify vertical position into top/middle/bottom third."""
    if height <= 0:
        return "middle"
    third = center_y / height
    if third < 1 / 3:
        return "top"
    if third > 2 / 3:
        return "bottom"
    return "middle"


def get_normalized_offset_from_center(
    center_x: float, center_y: float, width: float, height: float
) -> tuple[float, float]:
    """
    Return (dx, dy) in [-1, 1]: positive = right/down from image center.
    Clamped to [-1, 1].
    """
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    cx_img = width / 2.0
    cy_img = height / 2.0
    dx = (center_x - cx_img) / (width / 2.0)
    dy = (center_y - cy_img) / (height / 2.0)
    dx = max(-1.0, min(1.0, dx))
    dy = max(-1.0, min(1.0, dy))
    return dx, dy


def format_direction_string(
    label: str,
    horizontal: HorizontalRegion,
    vertical: VerticalRegion,
    distance_label: DistanceLabel | None = None,
) -> str:
    """
    Produce natural language direction, e.g. "The remote is bottom-right of the frame."
    """
    parts: list[str] = []
    if horizontal == "left" and vertical == "top":
        parts.append("top-left")
    elif horizontal == "center" and vertical == "top":
        parts.append("top")
    elif horizontal == "right" and vertical == "top":
        parts.append("top-right")
    elif horizontal == "left" and vertical == "middle":
        parts.append("left")
    elif horizontal == "center" and vertical == "middle":
        parts.append("center")
    elif horizontal == "right" and vertical == "middle":
        parts.append("right")
    elif horizontal == "left" and vertical == "bottom":
        parts.append("bottom-left")
    elif horizontal == "center" and vertical == "bottom":
        parts.append("bottom")
    elif horizontal == "right" and vertical == "bottom":
        parts.append("bottom-right")
    else:
        parts.append(f"{vertical}-{horizontal}")

    region_str = parts[0]
    sentence = f"The {label} is {region_str} of the frame."
    if distance_label:
        sentence = f"The {label} is {region_str} of the frame ({distance_label})."
    return sentence


def get_distance_label(area_percentile: float) -> DistanceLabel:
    """
    Heuristic distance from bbox area percentile (0–100) in image.
    Smaller area -> farther; larger area -> nearer.
    """
    if area_percentile < 33.33:
        return "far"
    if area_percentile < 66.66:
        return "medium"
    return "near"


def compute_spatial_guidance(
    center_x: float,
    center_y: float,
    bbox_area_percentile: float,
    width: float,
    height: float,
    label: str,
    include_distance: bool = True,
) -> SpatialGuidance:
    """Build full SpatialGuidance for one detection."""
    horizontal = get_horizontal_region(center_x, width)
    vertical = get_vertical_region(center_y, height)
    dx, dy = get_normalized_offset_from_center(center_x, center_y, width, height)
    distance_label: DistanceLabel | None = (
        get_distance_label(bbox_area_percentile) if include_distance else None
    )
    direction_string = format_direction_string(label, horizontal, vertical, distance_label)
    return SpatialGuidance(
        horizontal=horizontal,
        vertical=vertical,
        dx=round(dx, 4),
        dy=round(dy, 4),
        direction_string=direction_string,
        distance_label=distance_label,
    )
