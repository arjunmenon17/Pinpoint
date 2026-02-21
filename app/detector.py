"""YOLO model loading and inference with device-aware presets and timing."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING

from ultralytics import YOLO

from app.config import (
    CONF_THRESHOLD,
    get_imgsz,
    get_max_dim,
    get_model_name,
    get_preset_name,
    resolve_device,
)
from app.schemas import Detection, SpatialGuidance
from app.spatial import compute_spatial_guidance
from app.utils import resize_to_max_dim

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_model: YOLO | None = None
_device: str | None = None

# Limit concurrent CPU inference to avoid pileups (1 = single-threaded inference)
_CPU_SEMAPHORE = threading.Semaphore(1)


def _get_device() -> str:
    global _device
    if _device is None:
        _device = resolve_device()
        if os.environ.get("MODEL_DEVICE", "auto").strip().lower() == "cuda" and _device == "cpu":
            logger.warning("MODEL_DEVICE=cuda but CUDA not available; falling back to CPU")
    return _device


def get_model() -> YOLO:
    """Load YOLO model once (singleton) on the resolved device."""
    global _model
    if _model is None:
        device = _get_device()
        model_name = get_model_name(device)
        logger.info("Loading YOLO model: %s on device=%s", model_name, device)
        _model = YOLO(model_name)
        logger.info("Model loaded | device=%s | preset=%s", device, get_preset_name(device))
    return _model


def get_device() -> str:
    """Return the resolved device (cuda or cpu). Ensures model is loaded."""
    get_model()
    return _get_device()


def warmup() -> None:
    """Run one tiny dummy inference to avoid first-request spike. Safe on CPU."""
    try:
        import numpy as np
        model = get_model()
        device = _get_device()
        # Tiny BGR image (64x64) to avoid OOM and keep CPU fast
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        imgsz = get_imgsz(device)
        half = device == "cuda"
        model.predict(
            dummy,
            conf=CONF_THRESHOLD,
            verbose=False,
            device=device,
            imgsz=min(imgsz, 128),
            half=half,
        )
        logger.info("Model warmup done (device=%s)", device)
    except Exception as e:
        logger.warning("Model warmup failed (non-fatal): %s", e)


def run_detection(
    image_bgr: NDArray,
    conf_threshold: float,
    target_label: str | None,
    top_k: int = 5,
    imgsz_override: int | None = None,
) -> tuple[list[Detection], dict]:
    """
    Run YOLO inference and build list of Detection with spatial guidance.
    Returns (detections, timing_dict) with keys: preprocess_ms, infer_ms, post_ms.
    """
    model = get_model()
    device = _get_device()
    imgsz = imgsz_override if imgsz_override is not None else get_imgsz(device)
    imgsz = max(160, min(1920, imgsz))  # clamp for safety
    max_dim = get_max_dim(device)
    half = device == "cuda"

    # Preprocess: downscale only when max_dim > 0 (GPU default 0 = no downscale)
    t_pre = time.perf_counter()
    if max_dim > 0 and max(image_bgr.shape[:2]) > max_dim:
        image_bgr = resize_to_max_dim(image_bgr, max_dim)
    preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

    h, w = image_bgr.shape[:2]
    image_area = h * w

    # Inference (optionally limit concurrency on CPU)
    if device == "cpu":
        _CPU_SEMAPHORE.acquire()
    try:
        t_infer = time.perf_counter()
        results = model.predict(
            image_bgr,
            conf=conf_threshold,
            verbose=False,
            device=device,
            imgsz=imgsz,
            half=half,
        )
        infer_ms = (time.perf_counter() - t_infer) * 1000.0
    finally:
        if device == "cpu":
            _CPU_SEMAPHORE.release()

    t_post = time.perf_counter()
    detections: list[Detection] = []
    if not results:
        post_ms = (time.perf_counter() - t_post) * 1000.0
        return detections, {"preprocess_ms": preprocess_ms, "infer_ms": infer_ms, "post_ms": post_ms}

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        post_ms = (time.perf_counter() - t_post) * 1000.0
        return detections, {"preprocess_ms": preprocess_ms, "infer_ms": infer_ms, "post_ms": post_ms}

    names = model.names
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)

    areas = []
    for (x1, y1, x2, y2) in boxes:
        area = (x2 - x1) * (y2 - y1)
        pct = (area / image_area) * 100.0 if image_area > 0 else 0.0
        areas.append(pct)
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

    detections.sort(key=lambda d: d.confidence, reverse=True)
    if target_lower is not None:
        detections = detections[:top_k]
    else:
        detections = detections[: max(top_k, 50)]

    post_ms = (time.perf_counter() - t_post) * 1000.0
    return detections, {"preprocess_ms": preprocess_ms, "infer_ms": infer_ms, "post_ms": post_ms}
