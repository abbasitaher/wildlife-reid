import numpy as np

from wildlife_reid.index import IndexedItem, VectorIndex


def _items(n: int) -> list[IndexedItem]:
    return [IndexedItem(identity=f"t{i:03d}", image_path=f"/data/t{i:03d}/a.jpg", embedding_index=i) for i in range(n)]


def test_add_and_search_returns_self_first():
    rng = np.random.default_rng(0)
    vectors = rng.random((5, 8)).astype("float32")
    index = VectorIndex(metric="cosine")
    index.add(vectors, _items(5))

    # querying with an exact gallery vector should return that item first
    result = index.search(vectors[2], top_k=3)
    assert len(result) == 3
    assert result[0]["identity"] == "t002"
    assert result[0]["score"] >= result[1]["score"]


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    vectors = rng.random((4, 8)).astype("float32")
    index = VectorIndex(metric="cosine")
    index.add(vectors, _items(4))
    index.save(tmp_path / "index")

    loaded = VectorIndex.load(tmp_path / "index", metric="cosine")
    assert len(loaded.items) == 4
    assert loaded.search(vectors[0], top_k=1)[0]["identity"] == "t000"


def test_empty_index_search_returns_empty():
    assert VectorIndex(metric="cosine").search(np.zeros(8, dtype="float32"), top_k=5) == []
