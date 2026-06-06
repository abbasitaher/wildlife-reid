from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

BACKBONE_SPECS: dict[str, tuple[str, int]] = {
    "efficientnet_v2_s": ("efficientnet_v2_s", 1280),
    "efficientnet_v2_m": ("efficientnet_v2_m", 1280),
    "efficientnet_v2_l": ("efficientnet_v2_l", 1280),
    "convnext_tiny": ("convnext_tiny", 768),
    "convnext_small": ("convnext_small", 768),
    "convnext_base": ("convnext_base", 1024),
    "convnext_large": ("convnext_large", 1536),
}


def _load_backbone(name: str) -> tuple[nn.Module, int]:
    if name not in BACKBONE_SPECS:
        raise ValueError(f"Unsupported backbone '{name}'. Options: {', '.join(BACKBONE_SPECS)}")

    model_name, feature_dim = BACKBONE_SPECS[name]
    weights_enum = {
        "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        "efficientnet_v2_m": models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
        "efficientnet_v2_l": models.EfficientNet_V2_L_Weights.IMAGENET1K_V1,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "convnext_small": models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "convnext_base": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "convnext_large": models.ConvNeXt_Large_Weights.IMAGENET1K_V1,
    }[model_name]

    factory = getattr(models, model_name)
    backbone = factory(weights=weights_enum)

    if hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()
    if hasattr(backbone, "head"):
        backbone.head = nn.Identity()
    if hasattr(backbone, "avgpool"):
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    return backbone, feature_dim


class EmbeddingEncoder(nn.Module):
    """Shared-weight encoder that maps images to L2-normalized embeddings."""

    def __init__(self, backbone: str, embedding_dim: int = 256) -> None:
        super().__init__()
        self.backbone, feature_dim = _load_backbone(backbone)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(feature_dim, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, embedding_dim)
        self.dropout3 = nn.Dropout(0.1)
        self.bn3 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        x = self.flatten(features)
        x = self.bn1(self.dropout1(self.fc1(x))) if x.size(0) > 1 else self.dropout1(self.fc1(x))
        x = self.bn2(self.dropout2(self.fc2(x))) if x.size(0) > 1 else self.dropout2(self.fc2(x))
        x = self.bn3(self.dropout3(self.fc3(x))) if x.size(0) > 1 else self.dropout3(self.fc3(x))
        if normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


def build_encoder(backbone: str, embedding_dim: int = 256) -> EmbeddingEncoder:
    return EmbeddingEncoder(backbone=backbone, embedding_dim=embedding_dim)
