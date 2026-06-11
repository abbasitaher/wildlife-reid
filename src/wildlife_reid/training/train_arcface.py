from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from wildlife_reid.config import AppConfig
from wildlife_reid.data.catalog import filter_records, load_records
from wildlife_reid.data.datasets import IdentityDataset, build_identity_samples, split_identity_samples
from wildlife_reid.models.encoder import build_arcface_model, resolve_embedding_dim
from wildlife_reid.storage import is_remote, join_uri, upload_file


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_retrieval_ranking(model, loader, device: torch.device) -> float:
    """Fraction of batches where same-identity pairs are closer than different-identity pairs."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images, normalize=True)
            if embeddings.size(0) < 2:
                continue
            distances = torch.cdist(embeddings, embeddings, p=2)
            for anchor_idx in range(embeddings.size(0)):
                anchor_label = labels[anchor_idx].item()
                positive_candidates = [idx for idx in range(embeddings.size(0)) if labels[idx].item() == anchor_label and idx != anchor_idx]
                negative_candidates = [idx for idx in range(embeddings.size(0)) if labels[idx].item() != anchor_label]
                if not positive_candidates or not negative_candidates:
                    continue
                positive_idx = positive_candidates[0]
                negative_idx = min(negative_candidates, key=lambda idx: distances[anchor_idx, idx].item())
                if distances[anchor_idx, positive_idx] < distances[anchor_idx, negative_idx]:
                    correct += 1
                total += 1

    return correct / total if total else 0.0


def train_arcface(config: AppConfig) -> str:
    set_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = [record for record in load_records(config) if record.exists()]
    train_records = filter_records(records, config.dataset.gallery_split)
    if not train_records:
        train_records = records
    if not train_records:
        raise FileNotFoundError("No training images found on disk.")

    samples, identity_to_idx = build_identity_samples(
        train_records,
        max_per_identity=config.training.max_per_category,
        seed=config.training.seed,
    )
    if len(identity_to_idx) < 2:
        raise ValueError("ArcFace training needs at least two identities.")

    train_samples, val_samples = split_identity_samples(
        samples,
        val_ratio=config.training.val_ratio,
        seed=config.training.seed,
    )

    image_size = config.model.image_size
    train_loader = DataLoader(
        IdentityDataset(train_samples, config.model.backbone, image_size, augment=True),
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        IdentityDataset(val_samples, config.model.backbone, image_size, augment=False),
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    embedding_dim = resolve_embedding_dim(config.model.backbone, config.model.embedding_dim)
    model = build_arcface_model(
        backbone=config.model.backbone,
        num_classes=len(identity_to_idx),
        embedding_dim=config.model.embedding_dim,
        pretrained=config.model.pretrained,
        arcface_scale=config.training.arcface_scale,
        arcface_margin=config.training.arcface_margin,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)

    checkpoints_uri = config.checkpoints_dir
    if is_remote(checkpoints_uri):
        local_checkpoint_dir = Path(tempfile.mkdtemp())
    else:
        local_checkpoint_dir = Path(checkpoints_uri)
        local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = local_checkpoint_dir / "best.pt"
    best_metric = -1.0

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images, labels=labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        val_ranking = evaluate_retrieval_ranking(model, val_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} val_ranking={val_ranking:.4f}")

        if val_ranking > best_metric:
            best_metric = val_ranking
            torch.save(
                {
                    "state_dict": model.encoder.state_dict(),
                    "backbone": config.model.backbone,
                    "embedding_dim": embedding_dim,
                    "image_size": image_size or model.encoder.image_size,
                    "num_classes": len(identity_to_idx),
                    "arcface_margin": config.training.arcface_margin,
                    "arcface_scale": config.training.arcface_scale,
                    "val_ranking": float(val_ranking),
                    "epoch": epoch,
                },
                best_path,
            )

    if is_remote(checkpoints_uri):
        remote_best = join_uri(checkpoints_uri, "best.pt")
        upload_file(best_path, remote_best)
        return remote_best
    return str(best_path)
