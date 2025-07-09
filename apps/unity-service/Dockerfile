FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --allow-unauthenticated \
    curl \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install exact PyTorch versions from working environment
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# Web framework with exact versions
RUN pip install --no-cache-dir \
    fastapi==0.115.7 \
    uvicorn==0.34.0 \
    grpcio==1.70.0

COPY . .

EXPOSE 50052

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:50052/health || exit 1

CMD ["python", "unity_service.py"]
