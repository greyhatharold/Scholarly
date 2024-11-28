FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.13 and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.13 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Add error handling and upgrade pip with verbose output
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch stable version
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Then install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --no-deps && \
    pip install --no-cache-dir -r requirements.txt

# Copy source files with verification
COPY src/ src/
COPY config/ config/
RUN test -d src && test -d config || (echo "Required directories missing" && exit 1)

ENTRYPOINT ["python", "-m", "src.training.train"]
