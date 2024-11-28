FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Add error handling and upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install requirements with better error handling
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || (cat /root/.cache/pip/log/*; exit 1)

# Copy source files
COPY src/ src/
COPY config/ config/

# Verify files exist
RUN test -d src && test -d config

EXPOSE 8080

ENTRYPOINT ["python", "-m", "src.serving.serve"]
