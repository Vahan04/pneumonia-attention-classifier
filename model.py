"""CNN classifier with an explicit spatial attention map."""

import torch
from torch import nn


class SpatialAttention(nn.Module):
    """Learn one importance value per spatial location."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        average = features.mean(dim=1, keepdim=True)
        maximum = features.amax(dim=1, keepdim=True)
        attention = torch.sigmoid(self.project(torch.cat((average, maximum), dim=1)))
        return features * attention, attention


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionCNN(nn.Module):
    """Compact binary classifier returning ``(logits, attention_map)``."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.attention = SpatialAttention(256)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        attended, attention_map = self.attention(features)
        return self.head(attended), attention_map
