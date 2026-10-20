"""Create an attention heatmap overlay for one chest X-ray."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .dataset import build_transform
from .model import AttentionCNN
from .utils import load_config


def explain(image_path: str | Path, checkpoint: str | Path, output_path: str | Path, image_size: int = 224) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = AttentionCNN().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = build_transform(image_size)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, attention = model(tensor)
    probability = torch.softmax(logits, dim=1)[0, 1].item()
    heatmap = torch.nn.functional.interpolate(attention, size=image.size[::-1], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (np.ptp(heatmap) + 1e-8)
    image_array = np.asarray(image).astype(np.float32) / 255.0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_array, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(heatmap, cmap="magma")
    axes[1].set_title("Spatial attention")
    axes[2].imshow(image_array, cmap="gray")
    axes[2].imshow(heatmap, cmap="magma", alpha=0.45)
    axes[2].set_title(f"P(pneumonia)={probability:.3f}")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="artifacts/best.pt")
    parser.add_argument("--output", default="artifacts/attention.png")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    explain(args.image, args.checkpoint, args.output, int(load_config(args.config)["image_size"]))


if __name__ == "__main__":
    main()
