# ── Churn Prediction Inference Server ────────────────────────────────────────
#
# Build:
#   docker build -t churn-prediction:latest .
#
# Run (mount the model artifact from the host):
#   docker run -p 8000:8000 \
#     -v $(pwd)/models_output:/app/models_output:ro \
#     churn-prediction:latest
#
# ⚠  HTTPS NOTE
# This container serves plain HTTP on port 8000.  In production it MUST sit
# behind an HTTPS-terminating reverse proxy or load balancer (e.g. AWS ALB,
# GCP Load Balancer, nginx, Traefik).  Customer data in the /predict payload
# must not travel over an unencrypted connection.
#
# Health check (used by orchestrators like ECS / Kubernetes):
#   GET http://localhost:8000/health
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Keeps Python from buffering stdout/stderr so logs appear immediately
ENV PYTHONUNBUFFERED=1
# Prevents Python from writing .pyc files into the image layer
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
# Copy only requirements first so Docker cache reuses this layer on code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Install the package ───────────────────────────────────────────────────────
# Copy source and package metadata, then install in editable mode so that
# all `src.*` imports resolve correctly without sys.path hacks.
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

# ── Model artifact ────────────────────────────────────────────────────────────
# The artifact (models_output/churn_model.joblib) is intentionally NOT baked
# into the image.  Baking it would mean rebuilding the image on every retrain.
# Instead, mount it as a read-only volume at runtime (see usage above).
VOLUME ["/app/models_output"]

# ── Expose & run ──────────────────────────────────────────────────────────────
EXPOSE 8000

# --workers 1 keeps memory predictable for a single-model server; increase if
# you need to handle high concurrency and have sufficient RAM.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
