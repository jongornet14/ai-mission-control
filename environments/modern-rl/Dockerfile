FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --allow-unauthenticated \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Core data science with exact versions
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.12.0 \
    scikit-learn==1.6.1

# ML frameworks with exact versions
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# Ray with exact version
RUN pip install --no-cache-dir ray==2.43.0

# Visualization with exact versions
RUN pip install --no-cache-dir \
    matplotlib==3.10.0 \
    tensorboard==2.18.0

# Web framework with exact versions
RUN pip install --no-cache-dir \
    fastapi==0.115.7 \
    uvicorn==0.34.0 \
    grpcio==1.70.0

COPY . .

EXPOSE 50054

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:50054/health || exit 1

CMD ["python", "modern_rl_service.py"]
