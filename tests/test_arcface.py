import torch

from wildlife_reid.models.arcface import ArcFaceHead


def test_arcface_head_output_shape():
    head = ArcFaceHead(embedding_dim=32, num_classes=5, scale=64.0, margin=0.5)
    embeddings = torch.randn(4, 32)
    labels = torch.tensor([0, 1, 2, 3])
    logits = head(embeddings, labels)
    assert logits.shape == (4, 5)
