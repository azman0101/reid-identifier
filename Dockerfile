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
COPY pyproject.toml .
RUN uv pip compile pyproject.toml -o requirements.txt
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
    && apt-get install -y --no-install-recommends intel-opencl-icd || true \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ARG TAILWIND_VERSION=v4.2.0
ENV TAILWIND_VERSION=${TAILWIND_VERSION}

# Copy application code
COPY . /app

# Build static CSS
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        TAILWIND_BIN="tailwindcss-linux-arm64"; \
    elif [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then \
        TAILWIND_BIN="tailwindcss-linux-x64"; \
    else \
        echo "Unsupported architecture: $ARCH"; \
    fi && \
    if [ -n "$TAILWIND_BIN" ]; then \
      curl -f -sLO "https://github.com/tailwindlabs/tailwindcss/releases/download/${TAILWIND_VERSION}/$TAILWIND_BIN" && \
      chmod +x "$TAILWIND_BIN" && \
      ./"$TAILWIND_BIN" -i reid_app/static/input.css -o reid_app/static/output.css --minify && \
      rm "$TAILWIND_BIN"; \
    fi

# --- Add Version Metadata ---
ARG BUILD_DATE
ARG GIT_SHA

LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$GIT_SHA

# Inject build info into the app
RUN echo "{\"build_date\": \"$BUILD_DATE\", \"git_sha\": \"$GIT_SHA\"}" > /app/reid_app/version.json

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p /models/gallery /models/unknown

# Expose port
EXPOSE 8002

# Run the application
CMD ["uvicorn", "reid_app.main:app", "--host", "0.0.0.0", "--port", "8002"]
