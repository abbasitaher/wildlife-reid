from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wildlife_reid.config import load_config
from wildlife_reid.evaluation import evaluate_retrieval
from wildlife_reid.retrieval import RetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate top-1/top-5 retrieval accuracy.")
    parser.add_argument("--config", default="configs/sea_turtle.yaml")
    parser.add_argument("--index", default=None, help="Optional path to a saved FAISS index directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    service = RetrievalService(config)
    if args.index:
        from wildlife_reid.index import VectorIndex

        service.index = VectorIndex.load(args.index, metric=config.index.metric)
    else:
        service.build_gallery_index()
        service.index.save(config.index_dir)

    result = evaluate_retrieval(config, service)
    print(f"Queries evaluated: {result.total}")
    print(f"Top-1 accuracy: {result.top1_accuracy:.4f}")
    print(f"Top-5 accuracy: {result.top5_accuracy:.4f}")


if __name__ == "__main__":
    main()
