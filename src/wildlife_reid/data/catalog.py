from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wildlife_reid.config import AppConfig


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    identity: str
    split: str | None = None

    def exists(self) -> bool:
        return self.path.exists()


def _load_csv_records(config: AppConfig) -> list[ImageRecord]:
    dataset = config.dataset
    csv_path = dataset.root / dataset.metadata_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    records: list[ImageRecord] = []
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        rel_path = row_dict[dataset.path_column]
        identity = str(row_dict[dataset.identity_column])
        split_value = str(row_dict[dataset.split_column]) if dataset.split_column in row_dict else None
        records.append(
            ImageRecord(
                path=(dataset.root / rel_path).resolve(),
                identity=identity,
                split=split_value,
            )
        )
    return records


def _load_folder_records(config: AppConfig) -> list[ImageRecord]:
    dataset = config.dataset
    records: list[ImageRecord] = []
    image_ext = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

    for dataset_name in dataset.datasets:
        dataset_dir = dataset.root / dataset_name
        if not dataset_dir.exists():
            continue
        for identity_dir in sorted(dataset_dir.iterdir()):
            if not identity_dir.is_dir() or identity_dir.name.startswith("."):
                continue
            identity = f"{dataset_name}/{identity_dir.name}"
            for image_path in sorted(identity_dir.iterdir()):
                if image_path.suffix.lower() in image_ext:
                    records.append(ImageRecord(path=image_path.resolve(), identity=identity, split=None))
    return records


def load_records(config: AppConfig) -> list[ImageRecord]:
    if config.dataset.layout == "folder_per_identity":
        return _load_folder_records(config)
    return _load_csv_records(config)


def filter_records(records: list[ImageRecord], split: str | None = None) -> list[ImageRecord]:
    if split is None:
        return records
    return [record for record in records if record.split == split]


def group_by_identity(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.identity, []).append(record)
    return grouped
