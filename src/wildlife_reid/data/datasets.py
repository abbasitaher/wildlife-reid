from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from wildlife_reid.transforms import apply_augmentation, apply_preprocess


class TripletDataset(Dataset):
    def __init__(
        self,
        triplets: list[tuple[Path, Path, Path]],
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.triplets = triplets
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.triplets)

    def _load(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        if self.augment:
            image = apply_augmentation(image)
        return apply_preprocess(image, self.image_size)

    def __getitem__(self, index: int):
        anchor_path, positive_path, negative_path = self.triplets[index]
        return self._load(anchor_path), self._load(positive_path), self._load(negative_path)
