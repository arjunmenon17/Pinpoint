# Pinpoint

**Pinpoint** is a low-latency assistive object-localization web API that helps you find an item in an image or webcam frame by detecting objects and returning interpretable spatial guidance (e.g., *"keys are bottom-left"*). 

- **Who it's for:** Everyday users who misplace small items (keys, wallet, phone, remote) and want quick, explainable location guidance from a photo or webcam. Also built to demonstrate production ML systems skills.
- **Stack:** Python 3.11, Ultralytics YOLOv8 (GPU or CPU), OpenCV, FastAPI, single-page Tailwind UI.
- **Live demo:** [**https://pinpoint-21aa.onrender.com/**](https://pinpoint-21aa.onrender.com/) — try it in the browser. The deployed app runs on **CPU only**, so it is **slower and less accurate** than running locally with a GPU; see [GPU vs CPU](#gpu-vs-cpu-latency-and-accuracy) below.

## Architecture overview

```
┌─────────────┐     POST /detect      ┌──────────────┐     ┌─────────────┐
│  Browser    │ ───────────────────► │  FastAPI     │ ──► │  YOLOv8     │
│  (static)   │   multipart image    │  app/main    │     │  (GPU: x,   │
└─────────────┘                      └──────┬───────┘     │   CPU: n)   │
       ▲                                    │             └──────┬──────┘
       │ JSON + base64 annotated image     │                    │
       │ timing (infer_ms, total_ms)        │                    ▼
       └───────────────────────────────────┘             ┌─────────────┐
                                                         │  spatial     │
                                                         │  (regions,   │
                                                         │   direction) │
                                                         └─────────────┘
```

- **app/main.py** – FastAPI app, `POST /detect`, file validation, static serve.
- **app/detector.py** – YOLO model singleton, inference, selection (target / top-k / all).
- **app/spatial.py** – Region (thirds), normalized offset from center, natural-language direction, distance heuristic.
- **app/utils.py** – Image decode/encode, validation, bounding-box drawing.
- **app/schemas.py** – Pydantic request/response models.

## Endpoint documentation

### `POST /detect`

Detect objects in an image and return detections with spatial guidance.

| Item | Description |
|------|-------------|
| **Request** | `multipart/form-data` with `file` = image (JPG, PNG, WebP, BMP; max size from `MAX_UPLOAD_MB`). |
| **Query params** | `target` (optional): class label to filter. `conf` (optional): confidence override 0–1. `top_k` (default 5): max detections when target set. `annotate` (default 1): 1 = include base64 annotated image, 0 = JSON only (faster). `imgsz` (optional): inference size override (160–1920). |
| **Response** | JSON: `detections`, `inference_latency_ms`, `timing` (decode_ms, preprocess_ms, infer_ms, post_ms, annotate_ms, total_ms), `annotated_image_b64` (if annotate=1), `target_requested`, `image_width`, `image_height`. |

**Selection logic:**

- If `target` is provided: return up to `top_k` detections of that class (highest confidence first).
- If no target: return all detections above confidence threshold, sorted by confidence.
- If target requested but none found: 200 with empty `detections` and optional annotated image.

### `GET /`

Serves the demo UI (single-page at `static/index.html`).

### `GET /health`

Health check; returns `status`, `device` (cuda/cpu), `model` (model filename), and `preset` (gpu/cpu).

## Run locally

**Prerequisites:** Python 3.11, pip.

```bash
# Clone and enter repo
cd Pinpoint

# Create venv (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install
pip install -r requirements.txt

# Optional env (see Configuration below)
set MODEL_NAME=yolov8n.pt
set MODEL_DEVICE=auto
set CONF_THRESHOLD=0.25

# Run (first run downloads YOLO weights)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** for the demo UI, or **http://localhost:8000/docs** for Swagger. With a CUDA-capable GPU, the app automatically uses the larger model (YOLOv8x) and full-resolution images for best quality and still low latency.

## GPU vs CPU: latency and accuracy

The same codebase runs in two modes depending on hardware, with different trade-offs:

| | **Local (GPU)** | **Deployed (CPU, e.g. Render)** |
|--|-----------------|----------------------------------|
| **Device** | CUDA | CPU only |
| **Model** | YOLOv8x (default) — larger, more accurate | YOLOv8n — small, fast on CPU |
| **Input** | No downscale (full resolution) | Capped (e.g. max dimension 1280 px) |
| **Typical inference** | **~100–200 ms** | **~20–60+ s** (e.g. ~30 s) |
| **Accuracy** | Higher (larger model + full res) | Lower (smaller model + resized input) |

**Why the gap?** GPU inference uses thousands of cores in parallel and optimized CUDA kernels; CPU runs the same math on a handful of cores, so each frame takes orders of magnitude longer. The deployed [live demo](https://pinpoint-21aa.onrender.com/) runs on Render’s free-tier CPU, so you can expect roughly **30 s or more** per image. Locally, with a GPU, the same request is typically **under 200 ms**. This reflects a deliberate systems trade-off: the hosted version prioritizes cost and compatibility (no GPU required), while local runs prioritize speed and quality.

You can reproduce these numbers: run the app locally with a GPU and call `POST /detect`; check the `timing.infer_ms` and `timing.total_ms` in the response. Then try the same image against the deployed API and compare.

## Run with Docker

CPU-only image:

```bash
# Build
docker build -t pinpoint .

# Run (port 8000)
docker run -p 8000:8000 pinpoint
```

Optional env:

```bash
docker run -p 8000:8000 -e CONF_THRESHOLD=0.3 pinpoint
```

## Example curl requests

**Detect all objects (no target):**

```bash
curl -X POST "http://localhost:8000/detect?conf=0.25" \
  -F "file=@/path/to/photo.jpg"
```

**Detect only “cell phone”:**

```bash
curl -X POST "http://localhost:8000/detect?target=cell%20phone&conf=0.3" \
  -F "file=@/path/to/photo.jpg"
```

**JSON only (faster, no annotated image):**

```bash
curl -X POST "http://localhost:8000/detect?annotate=0" \
  -F "file=@/path/to/photo.jpg"
```

## Configuration (environment)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DEVICE` | `auto` | `auto` (cuda if available, else cpu), `cpu`, or `cuda`. |
| `MODEL_NAME` | `yolov8n.pt` | YOLO weights used when running on **CPU**. |
| `GPU_MODEL_NAME` | `yolov8x.pt` | YOLO weights used when running on **GPU** (local: best model, no downscale). |
| `GPU_IMG_SIZE` | `1280` | Inference size on GPU. `GPU_MAX_DIM` = `0` (no resize) by default for full quality. |
| `CPU_IMG_SIZE` | `640` | Inference size on CPU. `CPU_MAX_DIM` = `1280` to keep inference fast. |
| `CONF_THRESHOLD` | `0.25` | Default confidence threshold (overridable by query `conf`). |
| `MAX_UPLOAD_MB` | `8` | Max upload size in MB. |

## Tests

```bash
pytest tests/ -v
```

Spatial logic is covered in `tests/test_spatial.py` (regions, offsets, direction strings, distance labels).

## Deployment

The live demo is hosted on **[Render](https://render.com)**. The app is containerized (Dockerfile, CPU-only image) and runs as a Render Web Service; Render sets `PORT` automatically. On the free tier the service may spin down when idle (cold start on first request), and inference is CPU-bound so responses are much slower than [local GPU](#gpu-vs-cpu-latency-and-accuracy)—see that section for the latency/accuracy trade-off.
