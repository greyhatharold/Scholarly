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

# Install typing_extensions and sympy first
RUN pip install --no-cache-dir typing_extensions sympy

# Install PyTorch with explicit version
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION} \
    --no-deps

# Verify PyTorch installation
RUN python -c "import torch; assert torch.__version__ == '${PYTORCH_VERSION}', f'Wrong torch version: {torch.__version__}'"

# Copy source files with verification
COPY src/ src/
COPY config/training_config.yaml config/training_config.yaml
RUN test -d src && test -f config/training_config.yaml || (echo "Required files missing" && exit 1)

ENTRYPOINT ["python", "-m", "src.training.train"]
