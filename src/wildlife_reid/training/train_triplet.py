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
from wildlife_reid.data.catalog import load_records
from wildlife_reid.data.datasets import TripletDataset
from wildlife_reid.data.triplet_mining import create_triplets
from wildlife_reid.models.encoder import build_encoder
from wildlife_reid.storage import is_remote, join_uri, upload_file


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_triplets(model, loader, device: torch.device, margin: float = 1.0) -> tuple[float, float]:
    model.eval()
    ap_distances: list[float] = []
    an_distances: list[float] = []

    with torch.no_grad():
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            anchor_emb = model(anchor, normalize=False)
            positive_emb = model(positive, normalize=False)
            negative_emb = model(negative, normalize=False)
            # Non-squared Euclidean distance to match nn.TripletMarginLoss, whose
            # margin is defined on ||a - p|| (not the squared distance).
            ap_distances.extend(torch.norm(anchor_emb - positive_emb, dim=1).cpu().tolist())
            an_distances.extend(torch.norm(anchor_emb - negative_emb, dim=1).cpu().tolist())

    ap = np.array(ap_distances)
    an = np.array(an_distances)
    # Fraction of triplets that already satisfy the training margin (zero loss).
    # Scale-aware but tied to the margin, unlike an arbitrary fixed threshold.
    margin_satisfied = float(np.mean((an - ap) > margin))
    # Scale-invariant ranking accuracy; used for model selection below.
    ranking = float(np.mean(ap < an))
    return margin_satisfied, ranking


def train_triplet(config: AppConfig) -> str:
    set_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = [record for record in load_records(config) if record.exists()]
    if not records:
        raise FileNotFoundError("No training images found on disk.")

    triplets = create_triplets(
        records,
        max_per_identity=config.training.max_per_category,
        seed=config.training.seed,
    )
    split_idx = int(len(triplets) * 0.9)
    train_triplets = triplets[:split_idx]
    val_triplets = triplets[split_idx:]

    train_loader = DataLoader(
        TripletDataset(train_triplets, config.model.image_size, augment=True),
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TripletDataset(val_triplets, config.model.image_size, augment=False),
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    model = build_encoder(config.model.backbone, config.model.embedding_dim).to(device)
    criterion = nn.TripletMarginLoss(margin=config.training.margin)
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
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
        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            optimizer.zero_grad()
            loss = criterion(
                model(anchor, normalize=False), model(positive, normalize=False), model(negative, normalize=False)
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        val_margin_satisfied, val_ranking = evaluate_triplets(model, val_loader, device, margin=config.training.margin)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} val_ranking={val_ranking:.4f} "
            f"val_margin_satisfied={val_margin_satisfied:.4f}"
        )

        if val_ranking > best_metric:
            best_metric = val_ranking
            # Save a portable, weights-only checkpoint: plain tensors plus the
            # primitive hyperparameters needed to rebuild the encoder. We avoid
            # pickling the AppConfig object so the checkpoint stays loadable across
            # OSes and code refactors (pickled dataclasses break on Path types and
            # renamed/removed classes).
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "backbone": config.model.backbone,
                    "embedding_dim": config.model.embedding_dim,
                    "image_size": config.model.image_size,
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
