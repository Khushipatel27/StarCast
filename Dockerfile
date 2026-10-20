FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACT_DIR=/app/artifacts

# xgboost needs OpenMP at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only xgboost build: drops ~1.4 GB of CUDA/NCCL libraries from the image
RUN sed 's/^xgboost/xgboost-cpu/' requirements.txt > /tmp/req.txt     && pip install --no-cache-dir -r /tmp/req.txt

COPY recsys/ recsys/
COPY app/ app/
COPY artifacts/ artifacts/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
