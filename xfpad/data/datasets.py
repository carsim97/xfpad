"""PyTorch Datasets for grayscale fingerprint patches and pre-extracted features."""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FingerprintDataset(Dataset):
    """Loads 224x224 grayscale fingerprint patches from disk.

    Parameters
    ----------
    image_paths : list of file paths.
    labels      : list of integer labels (same length as image_paths).
    transform   : optional callable applied to the image tensor.

    Returns
    -------
    (image, label) where image is a [1, H, W] float tensor in [0, 1].
    """

    def __init__(self,
                 image_paths: Sequence[str],
                 labels: Sequence[int],
                 transform: Optional[callable] = None,
                 cache_in_ram: bool = False) -> None:
        if len(image_paths) != len(labels):
            raise ValueError(
                f"image_paths and labels length mismatch: "
                f"{len(image_paths)} vs {len(labels)}"
            )
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

        # Optional decode-once cache. The patches are 224x224 uint8 grayscale
        # (~50 KB each), so a full split costs a few hundred MB — negligible
        # against the machine's RAM, and it removes the per-epoch imread/decode
        # cost that otherwise serialises with training when num_workers=0.
        # The cached array holds the *raw decoded* image, so __getitem__
        # returns tensors bit-identical to the uncached path.
        self._cache = None
        if cache_in_ram:
            first = self._read(self.image_paths[0])
            buf = np.empty((len(self.image_paths), *first.shape), dtype=np.uint8)
            buf[0] = first
            for i in range(1, len(self.image_paths)):
                img = self._read(self.image_paths[i])
                if img.shape != first.shape:
                    raise ValueError(
                        f"cache_in_ram requires uniform image shapes; "
                        f"{self.image_paths[i]} has {img.shape}, expected {first.shape}"
                    )
                buf[i] = img
            self._cache = buf

    @staticmethod
    def _read(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            img = self._cache[idx]
        else:
            img = self._read(self.image_paths[idx])

        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


class FeatureDataset(Dataset):
    """Wraps pre-extracted (1280-D) embeddings + integer labels."""

    def __init__(self,
                 features: np.ndarray | torch.Tensor,
                 labels: Sequence[int] | torch.Tensor) -> None:
        if isinstance(features, np.ndarray):
            self.features = torch.from_numpy(features).float()
        else:
            self.features = features.float()

        if isinstance(labels, torch.Tensor):
            self.labels = labels.long()
        else:
            self.labels = torch.tensor(list(labels), dtype=torch.long)

        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"features and labels length mismatch: "
                f"{self.features.shape[0]} vs {self.labels.shape[0]}"
            )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]
