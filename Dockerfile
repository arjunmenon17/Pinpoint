# Pinpoint - CPU-only deployment
FROM python:3.11-slim

WORKDIR /app

# Install runtime deps (OpenCV may need libgl1; libgl1-mesa-glx was removed in newer Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (avoids pulling ~2GB of CUDA deps on Render)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/

# Default env (override in run). Render sets PORT at runtime.
ENV MODEL_NAME=yolov8n.pt \
    CONF_THRESHOLD=0.25 \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
