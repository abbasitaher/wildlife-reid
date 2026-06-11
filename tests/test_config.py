import textwrap

from wildlife_reid.config import load_config
from wildlife_reid.models.backbone import get_backbone_spec


def _write_config(tmp_path, artifacts: str, backbone: str = "megadescriptor_l_384") -> str:
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
          backbone: {backbone}
          embedding_dim: null
          image_size: null
          checkpoint: null
          pretrained: false
        index:
          top_k: 5
          metric: cosine
        paths:
          artifacts: {artifacts}
        training:
          batch_size: 16
          epochs: 10
          learning_rate: 1.0e-4
          weight_decay: 1.0e-4
          arcface_margin: 0.5
          arcface_scale: 64.0
          max_per_category: 20
          seed: 71
          val_ratio: 0.1
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(content)
    return str(path)


def test_load_config_fields(tmp_path):
    config = load_config(_write_config(tmp_path, "artifacts/sea_turtle"))
    assert config.dataset.name == "sea_turtle"
    assert config.model.backbone == "megadescriptor_l_384"
    assert config.model.embedding_dim is None
    assert config.index.top_k == 5
    assert config.training.arcface_margin == 0.5
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
          backbone: megadescriptor_l_384
          checkpoint: gs://bucket/sea_turtle/models/v2/best.pt
        paths:
          artifacts: gs://bucket/sea_turtle
        """
    )
    path = tmp_path / "gcp.yaml"
    path.write_text(content)
    config = load_config(str(path))
    assert config.index_dir == "gs://bucket/sea_turtle/index/v2"


def test_backbone_spec_defaults():
    spec = get_backbone_spec("megadescriptor_l_384")
    assert spec.image_size == 384
    assert spec.feature_dim == 1536
