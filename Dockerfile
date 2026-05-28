FROM python:3.13-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --user --no-cache-dir -r requirements.txt
ENV PATH=/root/.local/bin:$PATH
RUN opentelemetry-bootstrap -a install
COPY . .

FROM python:3.13-slim
WORKDIR /app

# Install ffmpeg for pydub and parselmouth audio/video processing
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8080
ENTRYPOINT ["opentelemetry-instrument", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
