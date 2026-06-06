from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import torch

from wildlife_reid.storage import download_dir, is_remote, upload_dir


@dataclass
class IndexedItem:
    identity: str
    image_path: str
    embedding_index: int


class VectorIndex:
    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        self.index: faiss.Index | None = None
        self.items: list[IndexedItem] = []

    def add(self, embeddings: torch.Tensor | np.ndarray, items: list[IndexedItem]) -> None:
        vectors = self._to_numpy(embeddings)
        if vectors.size == 0:
            raise ValueError("Cannot add empty embedding matrix to index.")

        if self.metric == "cosine":
            faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim) if self.metric == "cosine" else faiss.IndexFlatL2(dim)

        start = len(self.items)
        for offset, item in enumerate(items):
            item.embedding_index = start + offset
        self.index.add(vectors)
        self.items.extend(items)

    def search(self, query: torch.Tensor | np.ndarray, top_k: int = 5) -> list[dict]:
        if self.index is None or not self.items:
            return []

        vector = self._to_numpy(query)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        if self.metric == "cosine":
            faiss.normalize_L2(vector)

        scores, indices = self.index.search(vector.astype(np.float32), top_k)
        matches: list[dict] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            item = self.items[idx]
            matches.append(
                {
                    "identity": item.identity,
                    "image_path": item.image_path,
                    "score": float(score),
                }
            )
        return matches

    def save(self, directory: str | Path) -> None:
        if self.index is None:
            raise ValueError("Index is empty; nothing to save.")

        if is_remote(directory):
            with tempfile.TemporaryDirectory() as tmp:
                self._save_local(Path(tmp))
                upload_dir(tmp, str(directory))
        else:
            self._save_local(Path(directory))

    def _save_local(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in self.items], handle, indent=2)

    @classmethod
    def load(cls, directory: str | Path, metric: str = "cosine") -> VectorIndex:
        if is_remote(directory):
            with tempfile.TemporaryDirectory() as tmp:
                download_dir(str(directory), tmp)
                return cls._load_local(Path(tmp), metric)
        return cls._load_local(Path(directory), metric)

    @classmethod
    def _load_local(cls, directory: Path, metric: str = "cosine") -> VectorIndex:
        instance = cls(metric=metric)
        instance.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "metadata.json", encoding="utf-8") as handle:
            raw_items = json.load(handle)
        instance.items = [IndexedItem(**item) for item in raw_items]
        return instance

    @staticmethod
    def _to_numpy(embeddings: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(embeddings, torch.Tensor):
            return embeddings.detach().cpu().numpy().astype(np.float32)
        return np.asarray(embeddings, dtype=np.float32)
