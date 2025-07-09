FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Box2D requirements
RUN apt-get update && apt-get install -y --allow-unauthenticated \
    curl \
    swig \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install basic packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy==1.26.4

# Install Box2D with proper system dependencies
RUN pip install --no-cache-dir box2d-py==2.3.5

# Install remaining packages with exact versions
RUN pip install --no-cache-dir \
    gym==0.26.2 \
    gym-notices==0.0.8 \
    pygame==2.6.1

# Web framework with exact versions
RUN pip install --no-cache-dir \
    fastapi==0.115.7 \
    uvicorn==0.34.0 \
    grpcio==1.70.0

COPY . .

EXPOSE 50053

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:50053/health || exit 1

CMD ["python", "gym_service.py"]
