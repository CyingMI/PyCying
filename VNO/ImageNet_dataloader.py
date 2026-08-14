import os
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

class imagenetdataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()

        self.root = Path(root)
        self.transform = transform
        
        classes = []
        for folder in os.listdir(self.root):
            folder_path = self.root / folder

            if folder_path.is_dir():
                classes.append(folder)

        classes.sort()
        
        self.classes = classes
        self.classes_to_idx = self._classes_to_idx_(self.classes)
        self.samples = self._create_samples_(self.classes_to_idx)

    def _classes_to_idx_(self, classes):
        classes_to_idx = {}
        for idx, class_name in enumerate(classes):
            classes_to_idx[class_name] = idx

        return classes_to_idx

    def _create_samples_(self, classes_to_idx):
        samples = []
        for class_name, idx in classes_to_idx.items():
            class_path = self.root / class_name
            for image_name in os.listdir(class_path):
                image_path = class_path / image_name
                
                samples.append(
                    (image_path, idx)
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

class ImageNetTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):

        image = F.resize(
            image,
            self.size,
            interpolation=InterpolationMode.BILINEAR
        )

        image = F.to_tensor(image)

        label = torch.tensor(
            label,
            dtype = torch.float32
        )

        return image, label

def build_dataloader(
        root,
        transform,
        batch_size,
        shuffle,
        num_workers,
):

    dataset = imagenetdataset(
    root = root,
    transform = transform,
    )

    dataloader =  DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
    ) 

    return dataloader