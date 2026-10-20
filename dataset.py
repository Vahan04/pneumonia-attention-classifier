"""Dataset and image preprocessing for the Kaggle chest X-ray dataset."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ChestXrayDataset(Dataset):
    """Read ``split/class/image`` files without requiring torchvision.

    The Kaggle dataset is expected to have the usual layout:
    ``chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing split directory: {split_dir}. "
                "Download the Kaggle dataset and pass its chest_xray directory."
            )

        self.samples: list[tuple[Path, int]] = []
        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(f"No images found in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
                image = (image - 0.5) / 0.5
        return image, label


def build_transform(image_size: int = 224, train: bool = False) -> Callable:
    """Return a small PIL-to-tensor transform with optional train augmentation."""

    def transform(image: Image.Image) -> torch.Tensor:
        if train:
            # Lightweight augmentation keeps the project usable without torchvision.
            if torch.rand(()) < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if torch.rand(()) < 0.2:
                image = image.rotate(float(torch.empty(1).uniform_(-7, 7)))
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.array(image, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        return (tensor - 0.5) / 0.5

    return transform
