"""Train the attention classifier: ``python -m src.train --config config.yaml``."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ChestXrayDataset, CLASS_NAMES, build_transform
from .model import AttentionCNN
from .utils import binary_metrics, load_config, save_json, set_seed


def run_epoch(model, loader, criterion, optimizer, device, training: bool) -> tuple[float, dict[str, float]]:
    model.train(training)
    losses, targets, predictions, positive_scores = [], [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits, _ = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        positive_scores.extend(logits.softmax(dim=1)[:, 1].detach().cpu().tolist())
    return sum(losses) / max(len(losses), 1), binary_metrics(targets, predictions, positive_scores)


def train(config: dict) -> None:
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(config["data_dir"])
    size = int(config["image_size"])
    train_set = ChestXrayDataset(data_dir, "train", size, build_transform(size, train=True))
    val_set = ChestXrayDataset(data_dir, "val", size, build_transform(size))
    loader_args = {"batch_size": int(config["batch_size"]), "num_workers": int(config["num_workers"])}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    model = AttentionCNN(num_classes=len(CLASS_NAMES)).to(device)
    counts = torch.bincount(torch.tensor([label for _, label in train_set.samples]), minlength=2).float()
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / counts).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print(f"epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.3f}")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model": model.state_dict(), "config": config, "class_names": CLASS_NAMES}, output_dir / "best.pt")
    save_json({"history": history, "device": str(device)}, output_dir / "history.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
