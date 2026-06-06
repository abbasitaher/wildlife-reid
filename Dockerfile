FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WILDLIFE_REID_CONFIG=configs/sea_turtle.gcp.yaml
ENV TORCH_HOME=/app/.torch
# Set WILDLIFE_REID_INDEX at deploy time, e.g.
#   gs://wildlifereidentification/sea_turtle/index/v1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
COPY app ./app

RUN pip install --no-cache-dir ".[gcp]"

# Bake ImageNet weights into the image so cold starts skip the ~208 MB download.
RUN python -c "\
from torchvision import models; \
models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1); \
print('EfficientNetV2-M weights cached under', __import__('os').environ['TORCH_HOME'])"

EXPOSE 8080

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
