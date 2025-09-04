import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(self, base_folder: str, train: bool, p: Optional[int] = 0.5, transforms: Optional[List] = None):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.rows, self.cols, self.channels = 32, 32, 3

        train_files = [f"data_batch_{i}" for i in range(1, 6)]
        test_files = ["test_batch"]
        load_files = train_files if train else test_files

        batches = []
        import pickle

        for file in load_files:
            with open(os.path.join(base_folder, file), "rb") as fo:
                dict = pickle.load(fo, encoding="bytes")
                batches.append(dict)
        self.images = np.concatenate([b[b"data"] for b in batches]).astype(np.float32) / 255.0

        self.images = self.images.reshape((-1, self.channels, self.rows, self.cols))
        self.labels = np.concatenate([b[b"labels"] for b in batches])

    ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION

        img, label = self.images[index], self.labels[index]
        img = self.apply_transforms(img.reshape(self.rows, self.cols, -1))
        if isinstance(index, (int, np.integer)):  # For single sample, no batching dim
            return img.reshape(self.channels, self.rows, self.cols), label
        return img.reshape(-1, self.channels, self.rows, self.cols), label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION
