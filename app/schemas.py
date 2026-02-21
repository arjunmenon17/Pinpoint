"""Pydantic models for API request/response."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --- Spatial / detection output ---

HorizontalRegion = Literal["left", "center", "right"]
VerticalRegion = Literal["top", "middle", "bottom"]
DistanceLabel = Literal["near", "medium", "far"]


class SpatialGuidance(BaseModel):
    """Spatial description of a detection."""

    horizontal: HorizontalRegion = Field(..., description="Left/center/right third")
    vertical: VerticalRegion = Field(..., description="Top/middle/bottom third")
    dx: float = Field(..., description="Normalized offset from center X (-1 to 1)")
    dy: float = Field(..., description="Normalized offset from center Y (-1 to 1)")
    direction_string: str = Field(..., description="Natural language direction")
    distance_label: DistanceLabel | None = Field(
        default=None, description="Heuristic near/medium/far from bbox area"
    )


class Detection(BaseModel):
    """Single object detection with spatial guidance."""

    label: str = Field(..., description="Class name (COCO)")
    class_id: int = Field(..., description="COCO class index")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    bbox: tuple[float, float, float, float] = Field(
        ..., description="xyxy in pixel coordinates"
    )
    spatial: SpatialGuidance = Field(..., description="Spatial guidance")


class TimingBreakdown(BaseModel):
    """Per-stage timing in milliseconds."""

    decode_ms: float = Field(0.0, description="Image decode time")
    preprocess_ms: float = Field(0.0, description="Resize/preprocess time")
    infer_ms: float = Field(0.0, description="Model inference time")
    post_ms: float = Field(0.0, description="Postprocess time")
    annotate_ms: float = Field(0.0, description="Draw/encode annotated image time")
    total_ms: float = Field(0.0, description="Total request time")


class DetectResponse(BaseModel):
    """Response for POST /detect."""

    detections: list[Detection] = Field(
        default_factory=list, description="Detections (filtered/sorted per request)"
    )
    inference_latency_ms: float = Field(
        ..., description="Inference time in ms (same as timing.infer_ms)"
    )
    timing: TimingBreakdown | None = Field(
        default=None, description="Per-stage timing breakdown"
    )
    annotated_image_b64: str | None = Field(
        default=None, description="PNG image as base64 string"
    )
    target_requested: str | None = Field(
        default=None, description="Target label from query param if provided"
    )
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")
