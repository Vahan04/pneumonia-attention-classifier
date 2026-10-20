"""Evaluate a saved checkpoint on the held-out test split."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import ChestXrayDataset, CLASS_NAMES, build_transform
from .model import AttentionCNN
from .train import run_epoch
from .utils import load_config, save_json


def evaluate(config: dict, checkpoint: str | Path | None = None) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(checkpoint or config["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train first or pass --checkpoint.")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    size = int(config["image_size"])
    dataset = ChestXrayDataset(config["data_dir"], "test", size, build_transform(size))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]))
    model = AttentionCNN(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(payload["model"])
    criterion = torch.nn.CrossEntropyLoss()
    loss, metrics = run_epoch(model, loader, criterion, None, device, False)
    result = {"loss": loss, **metrics}
    save_json(result, Path(config["output_dir"]) / "test_metrics.json")
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    evaluate(load_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
