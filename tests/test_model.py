import torch

from src.model import AttentionCNN, SpatialAttention


def test_model_returns_logits_and_spatial_attention():
    model = AttentionCNN()
    logits, attention = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 2)
    assert attention.shape == (2, 1, 14, 14)
    assert torch.all(attention >= 0) and torch.all(attention <= 1)


def test_attention_accepts_non_square_feature_maps():
    layer = SpatialAttention(8)
    features = torch.randn(3, 8, 11, 7)
    attended, attention = layer(features)
    assert attended.shape == features.shape
    assert attention.shape == (3, 1, 11, 7)
