"""Shared configuration, reproducibility, and metric helpers."""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML when available, with a clear dependency error."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to read config.yaml (PyYAML required).") from exc
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def binary_metrics(
    targets: list[int], predictions: list[int], positive_scores: list[float] | None = None
) -> dict[str, float]:
    """Compute clinically useful metrics, including sensitivity and specificity."""

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    result = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
    }
    negatives = sum(target == 0 for target in targets)
    true_negatives = sum(target == 0 and prediction == 0 for target, prediction in zip(targets, predictions))
    result["specificity"] = true_negatives / negatives if negatives else 0.0
    if len(set(targets)) == 2:
        result["roc_auc"] = float(roc_auc_score(targets, positive_scores or predictions))
    return result
