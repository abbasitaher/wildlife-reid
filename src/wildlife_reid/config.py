from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from wildlife_reid.storage import join_uri


def extract_model_version(checkpoint: str | None) -> str | None:
    """Parse a version label such as ``v1`` from a checkpoint URI like ``.../models/v1/best.pt``."""
    if not checkpoint:
        return None
    match = re.search(r"/models/(v\d+)/", str(checkpoint).replace("\\", "/"))
    return match.group(1) if match else None


@dataclass
class DatasetConfig:
    name: str
    root: Path
    metadata_csv: str | None = None
    split_column: str = "split_open"
    gallery_split: str = "train"
    query_split: str = "test"
    path_column: str = "file_name"
    identity_column: str = "identity"
    datasets: list[str] = field(default_factory=list)
    layout: str = "csv_metadata"
    auto_split: bool = True


@dataclass
class ModelConfig:
    backbone: str = "megadescriptor_l_384"
    embedding_dim: int | None = None
    image_size: int | None = None
    checkpoint: str | None = None
    pretrained: bool = True


@dataclass
class IndexConfig:
    top_k: int = 5
    metric: str = "cosine"


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0
    max_per_category: int = 20
    seed: int = 71
    val_ratio: float = 0.1


@dataclass
class AppConfig:
    dataset: DatasetConfig
    model: ModelConfig
    index: IndexConfig
    paths: dict[str, str]
    training: TrainingConfig

    @property
    def artifacts_dir(self) -> str:
        return self.paths.get("artifacts", "artifacts")

    @property
    def index_dir(self) -> str:
        version = extract_model_version(self.model.checkpoint)
        if version:
            return join_uri(self.artifacts_dir, "index", version)
        return join_uri(self.artifacts_dir, "index")

    @property
    def checkpoints_dir(self) -> str:
        return join_uri(self.artifacts_dir, "checkpoints")


def _build_dataset(raw: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        name=raw["name"],
        root=Path(raw["root"]),
        metadata_csv=raw.get("metadata_csv"),
        split_column=raw.get("split_column", "split_open"),
        gallery_split=raw.get("gallery_split", "train"),
        query_split=raw.get("query_split", "test"),
        path_column=raw.get("path_column", "file_name"),
        identity_column=raw.get("identity_column", "identity"),
        datasets=raw.get("datasets", []),
        layout=raw.get("layout", "csv_metadata" if raw.get("metadata_csv") else "folder_per_identity"),
        auto_split=raw.get("auto_split", True),
    )


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return AppConfig(
        dataset=_build_dataset(raw["dataset"]),
        model=ModelConfig(**raw.get("model", {})),
        index=IndexConfig(**raw.get("index", {})),
        paths=raw.get("paths", {"artifacts": "artifacts"}),
        training=TrainingConfig(**raw.get("training", {})),
    )
