"""Filesystem abstraction supporting local paths and Google Cloud Storage URIs.

Any artifact path (index directory, model checkpoint, manifest) may be a local
path or a `gs://bucket/prefix` URI. GCS access uses `google-cloud-storage`,
imported lazily so the dependency is only required when a remote path is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GCS_PREFIX = "gs://"


def is_remote(path: str | Path) -> bool:
    return str(path).startswith(GCS_PREFIX)


def join_uri(base: str | Path, *parts: str) -> str:
    """Join path segments for both local paths and gs:// URIs."""
    if is_remote(base):
        result = str(base).rstrip("/")
        for part in parts:
            result = f"{result}/{part.strip('/')}"
        return result
    return str(Path(base, *parts))


def _split_gcs(uri: str) -> tuple[str, str]:
    rest = uri[len(GCS_PREFIX) :]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix


def _client():
    from google.cloud import storage  # lazy import

    return storage.Client()


def exists(path: str | Path) -> bool:
    if is_remote(path):
        bucket_name, prefix = _split_gcs(str(path))
        client = _client()
        bucket = client.bucket(bucket_name)
        blobs = client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/", max_results=1)
        return any(True for _ in blobs)
    return Path(path).exists()


def upload_dir(local_dir: str | Path, remote_uri: str) -> None:
    bucket_name, prefix = _split_gcs(str(remote_uri))
    bucket = _client().bucket(bucket_name)
    local_dir = Path(local_dir)
    for file in local_dir.rglob("*"):
        if not file.is_file():
            continue
        rel = file.relative_to(local_dir).as_posix()
        blob_name = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
        bucket.blob(blob_name).upload_from_filename(str(file))


def download_dir(remote_uri: str, local_dir: str | Path) -> Path:
    bucket_name, prefix = _split_gcs(str(remote_uri))
    client = _client()
    bucket = client.bucket(bucket_name)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    normalized = prefix.rstrip("/") + "/"
    for blob in client.list_blobs(bucket, prefix=normalized):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(normalized) :]
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
    return local_dir


def upload_file(local_file: str | Path, remote_uri: str) -> None:
    bucket_name, blob_name = _split_gcs(str(remote_uri))
    bucket = _client().bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(str(local_file))


def download_file(remote_uri: str, local_file: str | Path) -> Path:
    bucket_name, blob_name = _split_gcs(str(remote_uri))
    bucket = _client().bucket(bucket_name)
    local_file = Path(local_file)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    bucket.blob(blob_name).download_to_filename(str(local_file))
    return local_file


def read_json(path: str | Path) -> Any:
    if is_remote(path):
        bucket_name, blob_name = _split_gcs(str(path))
        bucket = _client().bucket(bucket_name)
        data = bucket.blob(blob_name).download_as_text(encoding="utf-8")
    else:
        data = Path(path).read_text(encoding="utf-8")
    return json.loads(data)


def write_json(path: str | Path, obj: Any) -> None:
    data = json.dumps(obj, indent=2)
    if is_remote(path):
        bucket_name, blob_name = _split_gcs(str(path))
        bucket = _client().bucket(bucket_name)
        bucket.blob(blob_name).upload_from_string(data, content_type="application/json")
    else:
        local_path = Path(path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(data, encoding="utf-8")
