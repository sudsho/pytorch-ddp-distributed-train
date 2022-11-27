ARG PYTORCH_TAG=1.13.0-cuda11.6-cudnn8-runtime
FROM pytorch/pytorch:${PYTORCH_TAG}

WORKDIR /app

# system deps for opencv-style image work and curl for the dataset download
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch is already in the base image, so install the rest while keeping the pinned versions.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
# expose nothing; this image is for compute jobs, not serving.

CMD ["python", "-m", "src.train_single", "--config", "configs/default.yaml"]
