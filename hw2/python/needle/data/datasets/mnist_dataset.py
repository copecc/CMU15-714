import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            self.images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(num, rows * cols).astype(np.float32)
            self.images /= 255.0  # Normalize to [0.0, 1.0]
        with gzip.open(label_filename, "rb") as lbpath:
            magic, n = struct.unpack(">II", lbpath.read(8))
            self.labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
        self.rows = rows
        self.cols = cols
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        label = self.labels[index]
        batch = 1 if img.ndim == 1 else img.shape[0]
        # Apply transforms. This works for mnist, using batch as channel
        img = self.apply_transforms(img.reshape(self.rows, self.cols, -1))
        return img.reshape(batch, -1), label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION
