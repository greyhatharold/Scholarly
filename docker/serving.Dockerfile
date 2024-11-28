FROM python:3.10-slim

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

# Install typing_extensions first
RUN pip install --no-cache-dir typing_extensions

# Modify PyTorch installation to use specific version
RUN pip install torch==${PYTORCH_VERSION} \
    torchvision>=0.15.2 \
    torchaudio>=2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

# Add version verification
RUN python -c "import torch; assert torch.__version__ == '${PYTORCH_VERSION}', f'Wrong torch version: {torch.__version__}'"

# Then install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files with verification
COPY src/ src/
COPY config/training_config.yaml config/training_config.yaml
RUN test -d src && test -f config/training_config.yaml || (echo "Required files missing" && exit 1)

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

ENTRYPOINT ["python", "-m", "src.serving.serve"]
