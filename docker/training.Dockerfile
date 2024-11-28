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

# Install requirements with better error handling and verbose output
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -v || \
    (pip install --no-cache-dir -r requirements.txt --no-deps && \
     pip install --no-cache-dir -r requirements.txt) || \
    (echo "Failed to install requirements" && \
     cat /root/.cache/pip/log/* && \
     exit 1)

# Copy source files with verification
COPY src/ src/
COPY config/ config/
RUN test -d src && test -d config || (echo "Required directories missing" && exit 1)

ENTRYPOINT ["python", "-m", "src.training.train"]
