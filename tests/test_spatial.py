"""Unit tests for spatial region classification and direction formatting."""

from __future__ import annotations

import pytest

from app.spatial import (
    format_direction_string,
    get_distance_label,
    get_horizontal_region,
    get_normalized_offset_from_center,
    get_vertical_region,
    compute_spatial_guidance,
)


class TestHorizontalRegion:
    def test_left_third(self):
        assert get_horizontal_region(10, 100) == "left"
        assert get_horizontal_region(0, 90) == "left"

    def test_center_third(self):
        assert get_horizontal_region(50, 100) == "center"
        assert get_horizontal_region(45, 90) == "center"

    def test_right_third(self):
        assert get_horizontal_region(80, 100) == "right"
        assert get_horizontal_region(100, 100) == "right"

    def test_zero_width_returns_center(self):
        assert get_horizontal_region(0, 0) == "center"


class TestVerticalRegion:
    def test_top_third(self):
        assert get_vertical_region(10, 100) == "top"
        assert get_vertical_region(0, 90) == "top"

    def test_middle_third(self):
        assert get_vertical_region(50, 100) == "middle"
        assert get_vertical_region(45, 90) == "middle"

    def test_bottom_third(self):
        assert get_vertical_region(80, 100) == "bottom"
        assert get_vertical_region(100, 100) == "bottom"

    def test_zero_height_returns_middle(self):
        assert get_vertical_region(0, 0) == "middle"


class TestNormalizedOffset:
    def test_center_is_zero(self):
        dx, dy = get_normalized_offset_from_center(50, 50, 100, 100)
        assert dx == 0.0 and dy == 0.0

    def test_right_positive_dx(self):
        dx, dy = get_normalized_offset_from_center(100, 50, 100, 100)
        assert dx == 1.0 and dy == 0.0

    def test_left_negative_dx(self):
        dx, dy = get_normalized_offset_from_center(0, 50, 100, 100)
        assert dx == -1.0 and dy == 0.0

    def test_bottom_positive_dy(self):
        dx, dy = get_normalized_offset_from_center(50, 100, 100, 100)
        assert dx == 0.0 and dy == 1.0

    def test_clamped(self):
        dx, dy = get_normalized_offset_from_center(200, -50, 100, 100)
        assert dx == 1.0 and dy == -1.0


class TestDirectionString:
    def test_bottom_right(self):
        s = format_direction_string("remote", "right", "bottom", None)
        assert "remote" in s and "bottom-right" in s and "frame" in s

    def test_center(self):
        s = format_direction_string("keys", "center", "middle", None)
        assert "keys" in s and "center" in s

    def test_with_distance(self):
        s = format_direction_string("phone", "left", "top", "near")
        assert "phone" in s and "top-left" in s and "near" in s


class TestDistanceLabel:
    def test_far(self):
        assert get_distance_label(10.0) == "far"
        assert get_distance_label(0.0) == "far"

    def test_medium(self):
        assert get_distance_label(50.0) == "medium"
        assert get_distance_label(33.34) == "medium"

    def test_near(self):
        assert get_distance_label(80.0) == "near"
        assert get_distance_label(100.0) == "near"


class TestComputeSpatialGuidance:
    def test_full_guidance(self):
        g = compute_spatial_guidance(10, 10, 5.0, 100, 100, "remote", include_distance=True)
        assert g.horizontal == "left"
        assert g.vertical == "top"
        assert g.direction_string
        assert "remote" in g.direction_string
        assert g.distance_label == "far"

    def test_no_distance_label(self):
        g = compute_spatial_guidance(50, 50, 50.0, 100, 100, "keys", include_distance=False)
        assert g.distance_label is None
        assert g.horizontal == "center"
        assert g.vertical == "middle"
