import textwrap
from pathlib import Path

from wildlife_reid.config import AppConfig, DatasetConfig, IndexConfig, ModelConfig, TrainingConfig, load_config
from wildlife_reid.data.catalog import ImageRecord, assign_default_splits, load_records


def _frog_config(tmp_path) -> AppConfig:
    content = textwrap.dedent(
        f"""
        dataset:
          name: frog
          root: {tmp_path.as_posix()}
          datasets:
            - source_a
          layout: folder_per_identity
          auto_split: true
          gallery_split: train
          query_split: test
        model:
          backbone: megadescriptor_l_384
        paths:
          artifacts: artifacts/frog
        training:
          seed: 7
        """
    )
    path = tmp_path / "frog.yaml"
    path.write_text(content)
    return load_config(str(path))


def test_assign_default_splits(tmp_path):
    records = [
        ImageRecord(path=Path("/a/1.jpg"), identity="frog_a"),
        ImageRecord(path=Path("/a/2.jpg"), identity="frog_a"),
        ImageRecord(path=Path("/b/1.jpg"), identity="frog_b"),
        ImageRecord(path=Path("/b/2.jpg"), identity="frog_b"),
    ]
    config = AppConfig(
        dataset=DatasetConfig(
            name="frog",
            root=tmp_path,
            layout="folder_per_identity",
            auto_split=True,
            gallery_split="train",
            query_split="test",
        ),
        model=ModelConfig(),
        index=IndexConfig(),
        paths={"artifacts": "artifacts/frog"},
        training=TrainingConfig(seed=7),
    )
    split_records = assign_default_splits(records, config)
    assert any(record.split == "test" for record in split_records)
    assert any(record.split == "train" for record in split_records)


def test_load_frog_folder_records(tmp_path):
    source = tmp_path / "source_a" / "frog001"
    source.mkdir(parents=True)
    (source / "img0.jpg").write_bytes(b"fake")
    (source / "img1.jpg").write_bytes(b"fake")

    config = _frog_config(tmp_path)
    records = load_records(config)
    assert len(records) == 2
    assert records[0].identity == "source_a/frog001"
    assert {record.split for record in records} <= {"train", "test"}
