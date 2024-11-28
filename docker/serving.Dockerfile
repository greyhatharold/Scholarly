FROM us-west1-docker.pkg.dev/cloud-marketplace/google/python:3.10-slim

WORKDIR /app

# Improved error handling and system prep
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add error handling and upgrade pip with verbose output
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Add ARG for PyTorch version
ARG PYTORCH_VERSION=2.0.1
ARG PYTHON_VERSION=3.10

# Install typing_extensions first with better error handling
RUN pip install --no-cache-dir typing_extensions sympy --verbose || \
    (echo "Failed to install base dependencies" && exit 1)

# Install PyTorch with better error handling
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu \
    --verbose || \
    (echo "Failed to install PyTorch packages" && exit 1)

# Modify version verification to handle CPU suffix
RUN python -c "import torch; assert torch.__version__.startswith('${PYTORCH_VERSION}'), f'Wrong torch version: {torch.__version__}'"

# Then install remaining requirements with verbose output
COPY requirements.txt .
RUN pip install -v --no-cache-dir -r requirements.txt || \
    (echo "Failed to install requirements.txt" && exit 1)

# Copy source files with verification
COPY src/ src/
COPY config/training_config.yaml config/training_config.yaml
RUN if [ ! -d "src" ] || [ ! -f "config/training_config.yaml" ]; then \
    echo "Required files missing"; \
    ls -la .; \
    exit 1; \
fi

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

ENTRYPOINT ["python", "-m", "src.serving.serve"]

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_CLOUD_PROJECT=${PROJECT_ID}
