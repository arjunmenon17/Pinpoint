"""Environment-based config and device selection."""

from __future__ import annotations

import os


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# Device: auto | cpu | cuda
MODEL_DEVICE_RAW: str = os.environ.get("MODEL_DEVICE", "auto").strip().lower()
MODEL_NAME: str = os.environ.get("MODEL_NAME", "yolov8n.pt").strip()
# GPU (local): best model by default; CPU (deploy): lightweight
GPU_MODEL_NAME: str = os.environ.get("GPU_MODEL_NAME", "yolov8x.pt").strip()
CPU_IMG_SIZE: int = _env_int("CPU_IMG_SIZE", 640)
GPU_IMG_SIZE: int = _env_int("GPU_IMG_SIZE", 1280)
CONF_THRESHOLD: float = _env_float("CONF_THRESHOLD", 0.25)
MAX_UPLOAD_MB: int = max(1, min(50, _env_int("MAX_UPLOAD_MB", 8)))
# Max dimension before downscale. GPU: 0 = no cap (full quality); CPU: keep fast
GPU_MAX_DIM: int = _env_int("GPU_MAX_DIM", 0)
CPU_MAX_DIM: int = _env_int("CPU_MAX_DIM", 1280)

# Resolved device after probe
_resolved_device: str | None = None


def resolve_device() -> str:
    """Resolve MODEL_DEVICE to 'cuda' or 'cpu'. Cached after first call."""
    global _resolved_device
    if _resolved_device is not None:
        return _resolved_device
    if MODEL_DEVICE_RAW == "cpu":
        _resolved_device = "cpu"
        return _resolved_device
    if MODEL_DEVICE_RAW == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                _resolved_device = "cuda"
                return _resolved_device
        except Exception:
            pass
        _resolved_device = "cpu"
        return _resolved_device
    # auto
    try:
        import torch
        if torch.cuda.is_available():
            _resolved_device = "cuda"
            return _resolved_device
    except Exception:
        pass
    _resolved_device = "cpu"
    return _resolved_device


def get_model_name(device: str) -> str:
    """Model file to load: best (x) for GPU, lightweight (n) for CPU unless overridden."""
    return GPU_MODEL_NAME if device == "cuda" else MODEL_NAME


def get_imgsz(device: str) -> int:
    return GPU_IMG_SIZE if device == "cuda" else CPU_IMG_SIZE


def get_max_dim(device: str) -> int:
    return GPU_MAX_DIM if device == "cuda" else CPU_MAX_DIM


def get_preset_name(device: str) -> str:
    return "gpu" if device == "cuda" else "cpu"
