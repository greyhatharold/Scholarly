FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || (cat /root/.cache/pip/log/*; exit 1)

COPY src/ src/
COPY config/ config/

RUN test -d src && test -d config

ENTRYPOINT ["python", "-m", "src.training.train"]
