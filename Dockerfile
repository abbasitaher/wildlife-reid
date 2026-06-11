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

# Cache MegaDescriptor weights in the image so cold starts skip the Hugging Face download.
RUN python -c "\
import timm; \
timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', pretrained=True, num_classes=0); \
print('MegaDescriptor-L-384 weights cached')"

EXPOSE 8080

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
