"""Image I/O, decoding, encoding, and drawing."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Allowed MIME / extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def decode_image_from_bytes(data: bytes) -> NDArray[np.uint8]:
    """
    Decode image bytes to BGR numpy array for OpenCV.
    Raises ValueError on invalid or unsupported image.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback: try PIL then convert to BGR
        try:
            pil_img = Image.open(io.BytesIO(data))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Could not decode image: {e}") from e
    if img is None or img.size == 0:
        raise ValueError("Decoded image is empty")
    return img


def encode_image_to_png_b64(img: NDArray[np.uint8]) -> str:
    """Encode BGR image to base64 PNG string."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def validate_file_upload(
    filename: str | None, content: bytes | None
) -> tuple[str, bytes]:
    """
    Validate upload: file type and size. Returns (filename, content).
    Raises ValueError on validation failure.
    """
    if not content:
        raise ValueError("No file content")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large (max {MAX_FILE_SIZE_BYTES // (1024*1024)} MB)"
        )
    name = (filename or "image").strip()
    if not name:
        name = "image"
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    return name, content


def draw_detections(
    img: NDArray[np.uint8],
    boxes_xyxy: list[tuple[float, float, float, float]],
    labels: list[str],
    confidences: list[float],
    box_color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> NDArray[np.uint8]:
    """
    Draw bounding boxes and labels on image (in-place style; returns same image).
    """
    out = img.copy()
    for (x1, y1, x2, y2), label, conf in zip(boxes_xyxy, labels, confidences):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thickness)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), box_color, -1)
        cv2.putText(
            out,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out
