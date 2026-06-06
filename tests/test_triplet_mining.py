from pathlib import Path

from wildlife_reid.data.catalog import ImageRecord, group_by_identity
from wildlife_reid.data.triplet_mining import create_triplets


def _records() -> list[ImageRecord]:
    records = []
    for identity in ("t001", "t002", "t003"):
        for i in range(3):
            records.append(ImageRecord(path=Path(f"/data/{identity}/img{i}.jpg"), identity=identity))
    return records


def test_group_by_identity():
    grouped = group_by_identity(_records())
    assert set(grouped) == {"t001", "t002", "t003"}
    assert all(len(v) == 3 for v in grouped.values())


def test_create_triplets_structure():
    triplets = create_triplets(_records(), max_per_identity=3, seed=0)
    assert triplets, "expected non-empty triplet list"

    identity_of = {}
    for record in _records():
        identity_of[record.path] = record.identity

    for anchor, positive, negative in triplets:
        # anchor and positive share an identity; negative differs
        assert identity_of[anchor] == identity_of[positive]
        assert identity_of[anchor] != identity_of[negative]
        assert anchor != positive


def test_create_triplets_skips_singletons():
    records = [ImageRecord(path=Path("/data/solo/img0.jpg"), identity="solo")]
    records += [ImageRecord(path=Path(f"/data/pair/img{i}.jpg"), identity="pair") for i in range(2)]
    triplets = create_triplets(records, max_per_identity=5, seed=1)
    # "solo" has a single image and cannot be an anchor identity with a positive
    anchors = {a for a, _, _ in triplets}
    assert all("solo" not in str(a) for a in anchors)
