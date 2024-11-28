FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Add build args
ARG PYTORCH_VERSION=2.0.1
ARG CUDA_VERSION=117
ARG PYTHON_VERSION=3.10

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV VIRTUAL_ENV=/opt/venv
RUN python${PYTHON_VERSION} -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install typing_extensions with better error handling
RUN pip install --no-cache-dir typing_extensions sympy --verbose || \
    (echo "Failed to install base dependencies" && exit 1)

# Install PyTorch with better error handling
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION} \
    --verbose \
    --no-deps || \
    (echo "Failed to install PyTorch packages" && exit 1)

# Add requirements.txt installation
COPY requirements.txt .
RUN pip install -v --no-cache-dir -r requirements.txt || \
    (echo "Failed to install requirements.txt" && exit 1)

# Add better file verification
COPY src/ src/
COPY config/training_config.yaml config/training_config.yaml
RUN if [ ! -d "src" ] || [ ! -f "config/training_config.yaml" ]; then \
    echo "Required files missing"; \
    ls -la .; \
    exit 1; \
fi

ENTRYPOINT ["python", "-m", "src.training.train"]
