from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wildlife_reid.config import AppConfig
from wildlife_reid.data.catalog import ImageRecord, filter_records, load_records
from wildlife_reid.embedding import EmbeddingService
from wildlife_reid.index import IndexedItem, VectorIndex


@dataclass
class SearchResult:
    query_path: str
    matches: list[dict]


class RetrievalService:
    def __init__(self, config: AppConfig, index: VectorIndex | None = None) -> None:
        self.config = config
        self.embedder = EmbeddingService(config)
        self.index = index or VectorIndex(metric=config.index.metric)

    def build_gallery_index(self, records: list[ImageRecord] | None = None) -> VectorIndex:
        records = records or load_records(self.config)
        gallery = filter_records(records, self.config.dataset.gallery_split)
        gallery = [record for record in gallery if record.exists()]
        if not gallery:
            raise FileNotFoundError("No gallery images found on disk. Ensure image files exist under the dataset root.")

        embeddings = self.embedder.embed_batch([record.path for record in gallery])
        items = [
            IndexedItem(identity=record.identity, image_path=str(record.path), embedding_index=idx)
            for idx, record in enumerate(gallery)
        ]

        index = VectorIndex(metric=self.config.index.metric)
        index.add(embeddings, items)
        self.index = index
        return index

    def search(self, query: str | Path, top_k: int | None = None) -> SearchResult:
        top_k = top_k or self.config.index.top_k
        query_path = Path(query)
        embedding = self.embedder.embed_image(query_path)
        matches = self.index.search(embedding, top_k=top_k)
        return SearchResult(query_path=str(query_path), matches=matches)
