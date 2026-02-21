"""FastAPI application and routes for Pinpoint."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_imgsz, get_preset_name
from app.detector import get_device, get_model, run_detection, warmup
from app.schemas import DetectResponse, TimingBreakdown
from app.utils import (
    decode_image_from_bytes,
    draw_detections,
    encode_image_to_png_b64,
    validate_file_upload,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pinpoint")

# --- App ---
app = FastAPI(
    title="Pinpoint",
    description="Low-latency assistive object-localization API: detect objects and return spatial guidance.",
    version="0.1.0",
)

# Static files (mount after routes so /detect is not shadowed)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def preload_model():
    """Load YOLO at startup and warmup with a dummy inference."""
    try:
        get_model()
        device = get_device()
        preset = get_preset_name(device)
        logger.info("Model preloaded at startup | device=%s | preset=%s", device, preset)
        warmup()
    except Exception as e:
        logger.warning("Startup model preload failed (will load on first request): %s", e)


def _get_conf_threshold(query_conf: float | None) -> float:
    """Confidence threshold: query param overrides env."""
    env_val = os.environ.get("CONF_THRESHOLD", "0.25")
    try:
        default = float(env_val)
    except ValueError:
        default = 0.25
    if query_conf is not None and 0 <= query_conf <= 1:
        return query_conf
    return default


@app.get("/")
async def root():
    """Redirect or serve demo."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Pinpoint API", "docs": "/docs", "detect": "POST /detect"}


def _get_imgsz_override(imgsz_query: int | None) -> int | None:
    """Validate imgsz query (160–1920); return None to use preset default."""
    if imgsz_query is None:
        return None
    return max(160, min(1920, imgsz_query))


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile | None = File(None),
    target: str | None = Query(None, description="Target class label (e.g. 'cell phone')"),
    conf: float | None = Query(None, ge=0, le=1, description="Confidence threshold override"),
    top_k: int = Query(5, ge=1, le=50, description="Max detections to return when target set"),
    annotate: int = Query(1, ge=0, le=1, description="1=include annotated image, 0=JSON only (faster)"),
    imgsz: int | None = Query(None, ge=160, le=1920, description="Inference size override"),
):
    """
    Detect objects in an image and return detections with spatial guidance.
    Upload image via multipart form; optionally filter by target label.
    """
    request_start = time.perf_counter()
    logger.info("detect request start | target=%s", target)

    if file is None:
        raise HTTPException(status_code=400, detail="Missing file: upload an image")

    try:
        content = await file.read()
    except Exception as e:
        logger.warning("read upload failed: %s", e)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file") from e

    try:
        validate_file_upload(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    t_decode = time.perf_counter()
    try:
        image_bgr = decode_image_from_bytes(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    decode_ms = (time.perf_counter() - t_decode) * 1000.0

    height, width = image_bgr.shape[:2]
    conf_threshold = _get_conf_threshold(conf)
    target_normalized = target.strip() if target else None
    if target_normalized == "":
        target_normalized = None
    imgsz_override = _get_imgsz_override(imgsz)

    try:
        detections, timing_inner = run_detection(
            image_bgr,
            conf_threshold,
            target_normalized,
            top_k=top_k,
            imgsz_override=imgsz_override,
        )
    except Exception as e:
        logger.exception("detection failed: %s", e)
        raise HTTPException(status_code=500, detail="Detection failed") from e

    preprocess_ms = timing_inner.get("preprocess_ms", 0.0)
    infer_ms = timing_inner.get("infer_ms", 0.0)
    post_ms = timing_inner.get("post_ms", 0.0)

    annotated_b64 = None
    annotate_ms = 0.0
    if annotate == 1:
        t_ann = time.perf_counter()
        boxes = [d.bbox for d in detections]
        labels = [d.label for d in detections]
        confs = [d.confidence for d in detections]
        annotated_img = draw_detections(image_bgr, boxes, labels, confs)
        try:
            annotated_b64 = encode_image_to_png_b64(annotated_img)
        except Exception as e:
            logger.warning("encode annotated image: %s", e)
        annotate_ms = (time.perf_counter() - t_ann) * 1000.0

    total_ms = (time.perf_counter() - request_start) * 1000.0
    timing_breakdown = TimingBreakdown(
        decode_ms=decode_ms,
        preprocess_ms=preprocess_ms,
        infer_ms=infer_ms,
        post_ms=post_ms,
        annotate_ms=annotate_ms,
        total_ms=total_ms,
    )
    logger.info(
        "detect request end | total_ms=%.2f | infer_ms=%.2f | detections=%d",
        total_ms, infer_ms, len(detections),
    )

    return DetectResponse(
        detections=detections,
        inference_latency_ms=infer_ms,
        timing=timing_breakdown,
        annotated_image_b64=annotated_b64,
        target_requested=target_normalized,
        image_width=width,
        image_height=height,
    )


@app.get("/classes")
async def list_classes():
    """Return the list of object classes the model can detect (COCO 80 classes)."""
    model = get_model()
    classes = list(model.names.values())
    return {"classes": classes}


@app.get("/health")
async def health():
    """Health check: status, device, model, and active preset."""
    try:
        device = get_device()
        preset = get_preset_name(device)
        return {
            "status": "ok",
            "device": device,
            "model": os.environ.get("MODEL_NAME", "yolov8n.pt"),
            "preset": preset,
        }
    except Exception as e:
        logger.warning("health check error: %s", e)
        return {"status": "ok", "device": None, "model": None, "preset": None}
