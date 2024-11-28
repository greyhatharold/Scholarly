FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY config/ config/

EXPOSE 8080

ENTRYPOINT ["python", "-m", "src.serving.serve"]
