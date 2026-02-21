# Use uv base image for building dependencies
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Create virtual environment
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
# 1. Copy only pyproject.toml to cache dependency installation
COPY pyproject.toml .

# 2. Compile requirements.txt from pyproject.toml
RUN uv pip compile pyproject.toml -o requirements.txt

# 3. Install dependencies from requirements.txt
RUN uv pip install -r requirements.txt

# Final stage
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install runtime dependencies for OpenCV and OpenVINO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    ocl-icd-libopencl1 \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
# We copy everything in '.' (root of build context) to '/app'.
# This includes 'reid_app' package directory, 'pyproject.toml', etc.
COPY . /app

# Set PYTHONPATH so python can find 'reid_app' package in /app
ENV PYTHONPATH=/app

# Create necessary directories for models (if not mounted)
RUN mkdir -p /models/gallery /models/unknown

# Expose port
EXPOSE 8002

# Run the application
CMD ["uvicorn", "reid_app.main:app", "--host", "0.0.0.0", "--port", "8002"]
