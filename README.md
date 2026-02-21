# Pinpoint

**Pinpoint** is a low-latency assistive object-localization web API that helps you find an item in an image or webcam frame by detecting objects and returning interpretable spatial guidance (e.g., *"keys are bottom-left"*).

- **Who it's for:** Everyday users who misplace small items (keys, wallet, phone, remote) and want quick, explainable location guidance from a photo or webcam. Also built to demonstrate production ML systems skills.
- **Stack:** Python 3.11, Ultralytics YOLOv8 (CPU), OpenCV, FastAPI, single-page Tailwind UI.

## Architecture overview

```
┌─────────────┐     POST /detect      ┌──────────────┐     ┌─────────────┐
│  Browser    │ ───────────────────► │  FastAPI     │ ──► │  YOLOv8n    │
│  (static)   │   multipart image    │  app/main    │     │  detector   │
└─────────────┘                      └──────┬───────┘     └──────┬──────┘
       ▲                                    │                    │
       │ JSON + base64 annotated image      │                    ▼
       │ inference_latency_ms               │             ┌─────────────┐
       └───────────────────────────────────┘             │  spatial     │
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

Open **http://localhost:8000** for the demo UI, or **http://localhost:8000/docs** for Swagger.

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

## Deploy on Render

The app is already containerized (Dockerfile). You can deploy it on [Render](https://render.com) as a Web Service.

### Option A: Deploy from dashboard (recommended)

1. Push this repo to GitHub (or GitLab).
2. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**.
3. Connect your repo and select the Pinpoint repository.
4. Configure:
   - **Name:** `pinpoint` (or any name).
   - **Region:** choose one.
   - **Runtime:** **Docker** (required).
   - **Branch:** `main` (or your default).
   - Leave **Dockerfile Path** blank (it’s in the repo root).
5. (Optional) Under **Environment**, add:
   - `MODEL_NAME` = `yolov8n.pt`
   - `CONF_THRESHOLD` = `0.25`
6. Click **Create Web Service**. Render will build the image from the Dockerfile and run it. The first deploy may take a few minutes (downloads YOLO weights).

Your service URL will be like `https://pinpoint-xxxx.onrender.com`. Open it to use the demo UI; use `POST /detect` for the API.

### Option B: Blueprint (render.yaml)

The repo includes a `render.yaml` blueprint. After connecting the repo, you can use **New** → **Blueprint** and point Render at this repo; it will create the web service from the blueprint.

### Notes

- **Free tier:** The service may spin down after inactivity; the first request after idle can be slow (cold start). **Inference on Render’s free CPU is slow (often 20–60+ seconds per image)**; the button stays disabled until the response returns. Paid plans have more CPU and are faster.
- **PORT:** The Dockerfile uses `PORT` from the environment (default 8000). Render sets `PORT` automatically; no need to set it yourself.
- **No GPU:** The Dockerfile is CPU-only; inference runs on CPU.

## License

Use and modify as needed for portfolio and interviews.
