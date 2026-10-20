from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import ChestXrayDataset, CLASS_NAMES, build_transform


def test_dataset_reads_expected_shape(tmp_path: Path):
    for split in ("train",):
        for class_name in CLASS_NAMES:
            folder = tmp_path / split / class_name
            folder.mkdir(parents=True)
            Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(folder / "sample.png")
    dataset = ChestXrayDataset(tmp_path, "train", image_size=32, transform=build_transform(32))
    image, label = dataset[0]
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label in (0, 1)
