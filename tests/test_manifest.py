import pytest

from wildlife_reid.config import AppConfig, DatasetConfig, IndexConfig, ModelConfig, TrainingConfig
from wildlife_reid.manifest import validate_index_manifest


def _config(checkpoint: str | None = "gs://bucket/sea_turtle/models/v1/best.pt") -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(name="sea_turtle", root="/data"),
        model=ModelConfig(
            backbone="efficientnet_v2_m",
            embedding_dim=256,
            image_size=384,
            checkpoint=checkpoint,
        ),
        index=IndexConfig(top_k=5, metric="cosine"),
        paths={"artifacts": "gs://bucket/sea_turtle"},
        training=TrainingConfig(),
    )


def _manifest(**overrides) -> dict:
    base = {
        "backbone": "efficientnet_v2_m",
        "embedding_dim": 256,
        "image_size": 384,
        "index_metric": "cosine",
        "checkpoint": "gs://bucket/sea_turtle/models/v1/best.pt",
        "model_version": "v1",
        "gallery_size": 100,
    }
    base.update(overrides)
    return base


def test_validate_index_manifest_ok():
    validate_index_manifest(_manifest(), _config())


def test_validate_index_manifest_rejects_backbone_mismatch():
    with pytest.raises(ValueError, match="backbone"):
        validate_index_manifest(_manifest(backbone="convnext_tiny"), _config())


def test_validate_index_manifest_rejects_version_mismatch():
    with pytest.raises(ValueError, match="model_version"):
        validate_index_manifest(_manifest(model_version="v2"), _config())


def test_validate_index_manifest_accepts_matching_version_different_checkpoint_path():
    manifest = _manifest(
        checkpoint="artifacts/sea_turtle/checkpoints/best.pt",
        model_version="v1",
    )
    validate_index_manifest(manifest, _config())
