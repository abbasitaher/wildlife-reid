import textwrap

from wildlife_reid.config import load_config


def _write_config(tmp_path, artifacts: str) -> str:
    content = textwrap.dedent(
        f"""
        dataset:
          name: sea_turtle
          root: /data/turtles
          metadata_csv: metadata_splits.csv
          split_column: split_open
          gallery_split: train
          query_split: test
        model:
          backbone: efficientnet_v2_m
          embedding_dim: 256
          image_size: 384
          checkpoint: null
        index:
          top_k: 5
          metric: cosine
        paths:
          artifacts: {artifacts}
        training:
          batch_size: 32
          epochs: 10
          learning_rate: 1.0e-5
          margin: 1.0
          max_per_category: 20
          seed: 71
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(content)
    return str(path)


def test_load_config_fields(tmp_path):
    config = load_config(_write_config(tmp_path, "artifacts/sea_turtle"))
    assert config.dataset.name == "sea_turtle"
    assert config.model.embedding_dim == 256
    assert config.index.top_k == 5
    assert config.training.seed == 71


def test_index_dir_local(tmp_path):
    config = load_config(_write_config(tmp_path, "artifacts/sea_turtle"))
    assert config.index_dir.replace("\\", "/") == "artifacts/sea_turtle/index"


def test_index_dir_remote(tmp_path):
    config = load_config(_write_config(tmp_path, "gs://bucket/sea_turtle"))
    assert config.index_dir == "gs://bucket/sea_turtle/index"
    assert config.checkpoints_dir == "gs://bucket/sea_turtle/checkpoints"


def test_index_dir_remote_versioned(tmp_path):
    content = textwrap.dedent(
        """
        dataset:
          name: sea_turtle
          root: /data/turtles
        model:
          backbone: efficientnet_v2_m
          checkpoint: gs://bucket/sea_turtle/models/v2/best.pt
        paths:
          artifacts: gs://bucket/sea_turtle
        """
    )
    path = tmp_path / "gcp.yaml"
    path.write_text(content)
    config = load_config(str(path))
    assert config.index_dir == "gs://bucket/sea_turtle/index/v2"
