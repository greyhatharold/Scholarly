FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime-python3.9

WORKDIR /app

# Improved error handling and system prep
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add error handling and upgrade pip with verbose output
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch dependencies first to ensure correct versions
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Then install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --no-deps && \
    pip install --no-cache-dir -r requirements.txt

# Copy source files with verification
COPY src/ src/
COPY config/ config/
RUN test -d src && test -d config || (echo "Required directories missing" && exit 1)

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

ENTRYPOINT ["python", "-m", "src.serving.serve"]
