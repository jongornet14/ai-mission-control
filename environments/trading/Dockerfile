FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --allow-unauthenticated \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install exact versions from working Linux environment
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.12.0 \
    scikit-learn==1.6.1 \
    matplotlib==3.10.0

# Trading specific
RUN pip install --no-cache-dir yfinance

# Web framework with exact versions
RUN pip install --no-cache-dir \
    fastapi==0.115.7 \
    uvicorn==0.34.0 \
    grpcio==1.70.0

COPY . .

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:50051/health || exit 1

CMD ["python", "trading_service.py"]
