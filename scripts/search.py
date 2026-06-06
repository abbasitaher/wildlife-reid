from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wildlife_reid.config import load_config
from wildlife_reid.index import VectorIndex
from wildlife_reid.retrieval import RetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the gallery for a query image.")
    parser.add_argument("--config", default="configs/sea_turtle.yaml")
    parser.add_argument("--query", required=True, help="Path to query image.")
    parser.add_argument("--index", default=None, help="Path to saved FAISS index directory.")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    service = RetrievalService(config)
    index_path = args.index or config.index_dir
    service.index = VectorIndex.load(index_path, metric=config.index.metric)

    result = service.search(args.query, top_k=args.top_k)
    print(json.dumps({"query": result.query_path, "matches": result.matches}, indent=2))


if __name__ == "__main__":
    main()
