from __future__ import annotations

from dataclasses import dataclass

import timm
import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class BackboneSpec:
    family: str
    feature_dim: int
    image_size: int
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]
    pad_square: bool
    timm_name: str | None = None
    torchvision_name: str | None = None


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "megadescriptor_l_384": BackboneSpec(
        family="megadescriptor",
        timm_name="hf-hub:BVRA/MegaDescriptor-L-384",
        feature_dim=1536,
        image_size=384,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        pad_square=False,
    ),
    "megadescriptor_t_224": BackboneSpec(
        family="megadescriptor",
        timm_name="hf-hub:BVRA/MegaDescriptor-T-224",
        feature_dim=768,
        image_size=224,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        pad_square=False,
    ),
    # Legacy torchvision backbones (checkpoints from earlier versions).
    "efficientnet_v2_m": BackboneSpec(
        family="torchvision",
        torchvision_name="efficientnet_v2_m",
        feature_dim=1280,
        image_size=384,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        pad_square=True,
    ),
}


def get_backbone_spec(name: str) -> BackboneSpec:
    if name not in BACKBONE_SPECS:
        supported = ", ".join(sorted(BACKBONE_SPECS))
        raise ValueError(f"Unsupported backbone '{name}'. Options: {supported}")
    return BACKBONE_SPECS[name]


def _load_torchvision_backbone(name: str, pretrained: bool) -> nn.Module:
    weights_enum = {
        "efficientnet_v2_m": models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
    }[name]
    factory = getattr(models, name)
    backbone = factory(weights=weights_enum if pretrained else None)
    if hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()
    if hasattr(backbone, "head"):
        backbone.head = nn.Identity()
    if hasattr(backbone, "avgpool"):
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    return backbone


def load_feature_extractor(backbone: str, pretrained: bool = True) -> tuple[nn.Module, BackboneSpec]:
    spec = get_backbone_spec(backbone)
    if spec.family == "megadescriptor":
        assert spec.timm_name is not None
        model = timm.create_model(spec.timm_name, pretrained=pretrained, num_classes=0)
        return model, spec

    assert spec.torchvision_name is not None
    return _load_torchvision_backbone(spec.torchvision_name, pretrained=pretrained), spec
