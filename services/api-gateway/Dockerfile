FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --allow-unauthenticated \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install API gateway dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.7 \
    uvicorn==0.34.0 \
    grpcio==1.70.0 \
    httpx==0.28.1 \
    aiohttp==3.11.11

# Copy the gateway service
COPY gateway_service.py .

COPY . .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "gateway_service:app", "--host", "0.0.0.0", "--port", "8080"]
