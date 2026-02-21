"""FastAPI application and routes for Pinpoint."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.detector import get_model, run_detection
from app.schemas import DetectResponse, Detection
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


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile | None = File(None),
    target: str | None = Query(None, description="Target class label (e.g. 'cell phone')"),
    conf: float | None = Query(None, ge=0, le=1, description="Confidence threshold"),
    top_k: int = Query(5, ge=1, le=50, description="Max detections to return when target set"),
    include_annotated: bool = Query(True, description="Include base64 annotated image"),
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

    try:
        image_bgr = decode_image_from_bytes(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    height, width = image_bgr.shape[:2]
    conf_threshold = _get_conf_threshold(conf)
    target_normalized = target.strip() if target else None
    if target_normalized == "":
        target_normalized = None

    try:
        detections, inference_latency_ms = run_detection(
            image_bgr, conf_threshold, target_normalized, top_k=top_k
        )
    except Exception as e:
        logger.exception("detection failed: %s", e)
        raise HTTPException(status_code=500, detail="Detection failed") from e

    # Optional: "target not found" when user asked for a target and we have no matches
    if target_normalized and not detections:
        logger.info("target not found: %s", target_normalized)
        # Still return 200 with empty detections and latency
        annotated_b64 = None
        if include_annotated:
            try:
                annotated = draw_detections(
                    image_bgr, [], [], []
                )
                annotated_b64 = encode_image_to_png_b64(annotated)
            except Exception:
                pass
        request_latency = (time.perf_counter() - request_start) * 1000
        logger.info("detect request end | latency_ms=%.2f | detections=0", request_latency)
        return DetectResponse(
            detections=[],
            inference_latency_ms=inference_latency_ms,
            annotated_image_b64=annotated_b64,
            target_requested=target_normalized,
            image_width=width,
            image_height=height,
        )

    annotated_b64 = None
    if include_annotated and detections:
        boxes = [d.bbox for d in detections]
        labels = [d.label for d in detections]
        confs = [d.confidence for d in detections]
        annotated = draw_detections(image_bgr, boxes, labels, confs)
        try:
            annotated_b64 = encode_image_to_png_b64(annotated)
        except Exception as e:
            logger.warning("encode annotated image: %s", e)

    request_latency = (time.perf_counter() - request_start) * 1000
    logger.info(
        "detect request end | latency_ms=%.2f | detections=%d",
        request_latency,
        len(detections),
    )
    return DetectResponse(
        detections=detections,
        inference_latency_ms=inference_latency_ms,
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
    """Health check for deployment."""
    return {"status": "ok"}
