from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wildlife_reid import __version__
from wildlife_reid.config import extract_model_version, load_config
from wildlife_reid.models.backbone import get_backbone_spec
from wildlife_reid.models.encoder import resolve_embedding_dim
from wildlife_reid.retrieval import RetrievalService
from wildlife_reid.storage import join_uri, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save a FAISS gallery index.")
    parser.add_argument("--config", default="configs/sea_turtle.yaml")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for index artifacts (local path or gs:// URI).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    service = RetrievalService(config)
    index = service.build_gallery_index()
    output = args.output or config.index_dir
    index.save(output)

    manifest = {
        "dataset": config.dataset.name,
        "metadata_csv": config.dataset.metadata_csv,
        "gallery_split": config.dataset.gallery_split,
        "gallery_size": len(index.items),
        "backbone": config.model.backbone,
        "embedding_dim": resolve_embedding_dim(config.model.backbone, config.model.embedding_dim),
        "image_size": config.model.image_size or get_backbone_spec(config.model.backbone).image_size,
        "checkpoint": config.model.checkpoint,
        "model_version": extract_model_version(config.model.checkpoint),
        "index_metric": config.index.metric,
        "index_type": "IndexFlatIP" if config.index.metric == "cosine" else "IndexFlatL2",
        "package_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(join_uri(output, "manifest.json"), manifest)

    print(f"Saved index with {len(index.items)} embeddings to {output}")
    print(f"Wrote manifest to {join_uri(output, 'manifest.json')}")


if __name__ == "__main__":
    main()
