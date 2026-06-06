"""Index manifest loading and validation against runtime config."""

from __future__ import annotations

import re
from typing import Any

from wildlife_reid.config import AppConfig, extract_model_version
from wildlife_reid.storage import exists, join_uri, read_json

_MANIFEST_NAME = "manifest.json"


def load_manifest(index_dir: str) -> dict[str, Any]:
    manifest_path = join_uri(index_dir, _MANIFEST_NAME)
    if not exists(manifest_path):
        raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
    return read_json(manifest_path)


def validate_index_manifest(manifest: dict[str, Any], config: AppConfig) -> None:
    """Raise ValueError if the index was built with a different model than configured."""
    errors: list[str] = []

    expected_version = extract_model_version(config.model.checkpoint)
    manifest_version = manifest.get("model_version")
    if expected_version and manifest_version != expected_version:
        errors.append(f"model_version: manifest={manifest_version!r} config={expected_version!r}")

    if manifest.get("backbone") != config.model.backbone:
        errors.append(f"backbone: manifest={manifest.get('backbone')!r} config={config.model.backbone!r}")

    if manifest.get("embedding_dim") != config.model.embedding_dim:
        errors.append(
            f"embedding_dim: manifest={manifest.get('embedding_dim')!r} config={config.model.embedding_dim!r}"
        )

    if manifest.get("image_size") != config.model.image_size:
        errors.append(f"image_size: manifest={manifest.get('image_size')!r} config={config.model.image_size!r}")

    if manifest.get("index_metric") != config.index.metric:
        errors.append(f"index_metric: manifest={manifest.get('index_metric')!r} config={config.index.metric!r}")

    manifest_checkpoint = manifest.get("checkpoint")
    config_checkpoint = config.model.checkpoint
    checkpoints_aligned = (
        not config_checkpoint
        or not manifest_checkpoint
        or (expected_version and manifest_version == expected_version)
        or _checkpoints_compatible(str(manifest_checkpoint), str(config_checkpoint))
    )
    if not checkpoints_aligned:
        errors.append(f"checkpoint: manifest={manifest_checkpoint!r} config={config_checkpoint!r}")

    if errors:
        raise ValueError(
            "Index manifest does not match runtime config — query and gallery embeddings "
            "may be incompatible. Rebuild the index with scripts/build_index.py.\n"
            + "\n".join(f"  - {item}" for item in errors)
        )


def _checkpoints_compatible(manifest_checkpoint: str, config_checkpoint: str) -> bool:
    if manifest_checkpoint == config_checkpoint:
        return True
    # Allow local vs gs:// paths when the versioned object key matches.
    manifest_version = extract_model_version(manifest_checkpoint)
    config_version = extract_model_version(config_checkpoint)
    if manifest_version and manifest_version == config_version:
        return True
    manifest_name = _checkpoint_basename(manifest_checkpoint)
    config_name = _checkpoint_basename(config_checkpoint)
    return bool(manifest_name and manifest_name == config_name)


def _checkpoint_basename(checkpoint: str) -> str | None:
    normalized = checkpoint.replace("\\", "/").rstrip("/")
    match = re.search(r"/models/(v\d+)/([^/]+)$", normalized)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    name = normalized.rsplit("/", 1)[-1]
    return name or None
