# V2 Audio Analysis Pipeline — Docker Image
# Target: ~100-120 MB (librosa + numpy mounted on volume)
#
# Strategy:
#   • python:3.11-slim base (~60 MB)
#   • ffmpeg from apt (~25 MB)
#   • Pure-Python FastAPI stack (~15 MB)
#   • librosa + numpy: install here OR mount from shared volume
#     (uncomment VOLUME line below and remove pip install librosa numpy
#      if you have a pre-built volume layer)

FROM python:3.11-slim

# System deps: ffmpeg for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (layer cache)
COPY v2/requirements.txt /app/v2/requirements.txt

# Install Python deps
# Remove librosa + numpy from this step if mounting from volume
RUN pip install --no-cache-dir -r /app/v2/requirements.txt

# Copy application code
COPY v2/ /app/v2/

# Mount point for heavy numerical libs (numpy, scipy, librosa) and Whisper cache
# To use: docker run -v heavy_modules:/opt/pymodules -v HF_cache:/root/.cache/huggingface ...
#         and set PYTHONPATH=/opt/pymodules
# VOLUME ["/opt/pymodules", "/root/.cache/huggingface"]

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "v2.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]
