from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from PIL import Image

from wildlife_reid.config import AppConfig
from wildlife_reid.models.encoder import EmbeddingEncoder, build_encoder
from wildlife_reid.storage import download_file, is_remote
from wildlife_reid.transforms import build_eval_transform


class EmbeddingService:
    def __init__(self, config: AppConfig, device: str | None = None) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model()
        self.transform = build_eval_transform(config.model.image_size)

    def _load_model(self) -> EmbeddingEncoder:
        model = build_encoder(
            backbone=self.config.model.backbone,
            embedding_dim=self.config.model.embedding_dim,
        )
        checkpoint = self.config.model.checkpoint
        if checkpoint:
            if is_remote(checkpoint):
                with tempfile.TemporaryDirectory() as tmp:
                    local_ckpt = download_file(checkpoint, Path(tmp) / "checkpoint.pt")
                    state = torch.load(local_ckpt, map_location=self.device, weights_only=False)
            else:
                state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            if isinstance(state, EmbeddingEncoder):
                model = state
            elif isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            else:
                model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def embed_image(self, image: Image.Image | Path | str) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.model(tensor).squeeze(0).cpu()

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image | Path | str], batch_size: int = 32) -> torch.Tensor:
        vectors: list[torch.Tensor] = []
        batch: list[torch.Tensor] = []

        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            batch.append(self.transform(image))
            if len(batch) == batch_size:
                tensor = torch.stack(batch).to(self.device)
                vectors.append(self.model(tensor).cpu())
                batch = []

        if batch:
            tensor = torch.stack(batch).to(self.device)
            vectors.append(self.model(tensor).cpu())

        return torch.cat(vectors, dim=0) if vectors else torch.empty((0, self.config.model.embedding_dim))
