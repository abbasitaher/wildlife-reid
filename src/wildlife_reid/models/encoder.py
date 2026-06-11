from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wildlife_reid.models.arcface import ArcFaceHead
from wildlife_reid.models.backbone import get_backbone_spec, load_feature_extractor


class EmbeddingEncoder(nn.Module):
    """Maps images to L2-normalized embeddings."""

    def __init__(
        self,
        backbone: str,
        embedding_dim: int | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.feature_extractor, self.spec = load_feature_extractor(backbone, pretrained=pretrained)
        self.output_dim = embedding_dim or self.spec.feature_dim
        self.projection: nn.Module | None = None

        if self.output_dim != self.spec.feature_dim:
            if self.spec.family == "megadescriptor":
                self.projection = nn.Sequential(
                    nn.Linear(self.spec.feature_dim, self.output_dim),
                    nn.BatchNorm1d(self.output_dim),
                )
            else:
                self.projection = nn.Sequential(
                    nn.Linear(self.spec.feature_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(512, self.output_dim),
                    nn.BatchNorm1d(self.output_dim),
                )

    @property
    def image_size(self) -> int:
        return self.spec.image_size

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.feature_extractor(x)
        if isinstance(features, tuple):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, 1)

        if self.projection is not None:
            if features.size(0) == 1 and self.training:
                self.projection.eval()
                features = self.projection(features)
                self.projection.train()
            else:
                features = self.projection(features)

        if normalize:
            features = F.normalize(features, p=2, dim=1)
        return features


class ArcFaceModel(nn.Module):
    """Encoder + ArcFace head for fine-tuning."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        embedding_dim: int | None = None,
        pretrained: bool = True,
        arcface_scale: float = 64.0,
        arcface_margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = EmbeddingEncoder(backbone=backbone, embedding_dim=embedding_dim, pretrained=pretrained)
        self.head = ArcFaceHead(
            embedding_dim=self.encoder.output_dim,
            num_classes=num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
        )

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        embeddings = self.encoder(images, normalize=False)
        if labels is None:
            return F.normalize(embeddings, p=2, dim=1) if normalize else embeddings
        return self.head(embeddings, labels)


def build_encoder(
    backbone: str,
    embedding_dim: int | None = None,
    pretrained: bool = True,
) -> EmbeddingEncoder:
    return EmbeddingEncoder(backbone=backbone, embedding_dim=embedding_dim, pretrained=pretrained)


def build_arcface_model(
    backbone: str,
    num_classes: int,
    embedding_dim: int | None = None,
    pretrained: bool = True,
    arcface_scale: float = 64.0,
    arcface_margin: float = 0.5,
) -> ArcFaceModel:
    return ArcFaceModel(
        backbone=backbone,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin,
    )


def resolve_embedding_dim(backbone: str, embedding_dim: int | None) -> int:
    spec = get_backbone_spec(backbone)
    return embedding_dim or spec.feature_dim
