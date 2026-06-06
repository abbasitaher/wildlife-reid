from __future__ import annotations

import random
from pathlib import Path

from tqdm import tqdm

from wildlife_reid.data.catalog import ImageRecord, group_by_identity


def create_triplets(
    records: list[ImageRecord],
    max_per_identity: int = 20,
    seed: int = 71,
) -> list[tuple[Path, Path, Path]]:
    """Build (anchor, positive, negative) triplets from identity-grouped images."""
    random.seed(seed)
    grouped = group_by_identity(records)
    identities = sorted(grouped.keys())
    triplets: list[tuple[Path, Path, Path]] = []

    for identity in tqdm(identities, desc="Mining triplets"):
        images = grouped[identity][:max_per_identity]
        if len(images) < 2:
            continue

        negative_identities = [item for item in identities if item != identity]
        if not negative_identities:
            continue

        for anchor in images:
            positives = [item for item in images if item.path != anchor.path]
            if not positives:
                continue
            positive = random.choice(positives)
            negative_identity = random.choice(negative_identities)
            negative = random.choice(grouped[negative_identity])
            triplets.append((anchor.path, positive.path, negative.path))

    random.shuffle(triplets)
    return triplets
