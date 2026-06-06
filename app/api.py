from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Query, UploadFile
from PIL import Image

from wildlife_reid.config import AppConfig, load_config
from wildlife_reid.index import VectorIndex
from wildlife_reid.logging_utils import get_logger
from wildlife_reid.manifest import load_manifest, validate_index_manifest
from wildlife_reid.retrieval import RetrievalService
from wildlife_reid.storage import exists as path_exists

CONFIG_PATH = os.environ.get("WILDLIFE_REID_CONFIG", "configs/sea_turtle.yaml")
INDEX_PATH = os.environ.get("WILDLIFE_REID_INDEX")

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config(CONFIG_PATH)
    service = RetrievalService(config)
    index_dir = INDEX_PATH or config.index_dir
    if path_exists(index_dir):
        logger.info("Loading gallery index from %s", index_dir)
        service.index = VectorIndex.load(index_dir, metric=config.index.metric)
        manifest = load_manifest(index_dir)
        validate_index_manifest(manifest, config)
        logger.info(
            "Index manifest OK (model_version=%s, gallery_size=%s)",
            manifest.get("model_version"),
            manifest.get("gallery_size"),
        )
    else:
        logger.warning("No index at %s; building from source images", index_dir)
        service.build_gallery_index()
        service.index.save(index_dir)
    logger.info("Service ready with %d gallery vectors", len(service.index.items))
    app.state.config = config
    app.state.service = service
    yield


app = FastAPI(title="Wildlife Re-Identification API", lifespan=lifespan)


@app.get("/health")
def health():
    service: RetrievalService = app.state.service
    return {
        "status": "ok",
        "gallery_size": len(service.index.items),
    }


@app.post("/search")
async def search(
    file: UploadFile = File(...),
    top_k: int | None = Query(default=None),
):
    service: RetrievalService = app.state.service
    config: AppConfig = app.state.config
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    embedding = service.embedder.embed_image(image)
    matches = service.index.search(embedding, top_k=top_k or config.index.top_k)
    return {"matches": matches}
