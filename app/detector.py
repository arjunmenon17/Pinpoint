"""YOLO model loading and inference."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from ultralytics import YOLO

from app.schemas import Detection, SpatialGuidance
from app.spatial import compute_spatial_guidance
from app.utils import decode_image_from_bytes

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_model: YOLO | None = None


def get_model() -> YOLO:
    """Load YOLO model once (singleton)."""
    global _model
    if _model is None:
        model_name = os.environ.get("MODEL_NAME", "yolov8n.pt")
        logger.info("Loading YOLO model: %s", model_name)
        _model = YOLO(model_name)
        logger.info("Model loaded")
    return _model


def run_detection(
    image_bgr: NDArray,
    conf_threshold: float,
    target_label: str | None,
    top_k: int = 5,
) -> tuple[list[Detection], float]:
    """
    Run YOLO inference and build list of Detection with spatial guidance.
    Returns (detections, inference_latency_ms).
    Selection:
      - If target_label: best match (highest conf) for that class, then top-k if multiple.
      - If no target: all detections above conf_threshold, sorted by confidence.
    """
    model = get_model()
    h, w = image_bgr.shape[:2]
    image_area = h * w

    t0 = time.perf_counter()
    results = model.predict(
        image_bgr,
        conf=conf_threshold,
        verbose=False,
        device="cpu",
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    detections: list[Detection] = []
    if not results:
        return detections, latency_ms

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return detections, latency_ms

    # Collect all above threshold with spatial info
    names = model.names
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)

    # Area percentiles: we need all areas first to compute percentile
    areas = []
    for (x1, y1, x2, y2) in boxes:
        area = (x2 - x1) * (y2 - y1)
        pct = (area / image_area) * 100.0 if image_area > 0 else 0.0
        areas.append(pct)
    # Simple percentile: rank among these detections (0–100)
    sorted_areas = sorted(areas)
    n_a = len(sorted_areas)
    area_percentiles = []
    for a in areas:
        idx = sum(1 for s in sorted_areas if s < a)
        pct = (idx / n_a * 100.0) if n_a else 0.0
        area_percentiles.append(pct)

    target_lower = target_label.strip().lower() if target_label else None

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        conf = float(confs[i])
        cls_id = int(cls_ids[i])
        label = names.get(cls_id, f"class_{cls_id}")
        area_pct = area_percentiles[i]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        spatial = compute_spatial_guidance(cx, cy, area_pct, w, h, label, include_distance=True)
        det = Detection(
            label=label,
            class_id=cls_id,
            confidence=conf,
            bbox=(x1, y1, x2, y2),
            spatial=spatial,
        )
        if target_lower is None:
            detections.append(det)
        elif label.lower() == target_lower:
            detections.append(det)
        # else skip (we only want target class when target is set)

    # Sort by confidence descending
    detections.sort(key=lambda d: d.confidence, reverse=True)

    if target_lower is not None:
        detections = detections[:top_k]
    else:
        detections = detections[: max(top_k, 50)]  # cap when returning all

    return detections, latency_ms
