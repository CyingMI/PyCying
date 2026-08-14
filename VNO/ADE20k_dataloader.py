from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


class ADE20KDataset(Dataset):
    def __init__(self, root, split, transform = None):
        super().__init__()

        self.root = Path(root)
        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "annotations" / split
        self.transform = transform

        self.images = sorted(self.image_dir.glob("*jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / (image_path.stem + ".png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

class ADEtransformer:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):

        image = F.resize(
            image,
            self.size,
            interpolation=InterpolationMode.BILINEAR
        )

        mask = F.resize(
            mask,
            self.size,
            interpolation=InterpolationMode.NEAREST
        )

        image = F.to_tensor(image)

        mask = torch.as_tensor(
            np.array(mask),
            dtype = torch.long
        )

        return image, mask
    


def build_dataloader(
        root,
        split,
        transform,
        batch_size,
        shuffle,
        num_workers,
        collate_fn,
):

    dataset = ADE20KDataset(
        root=root,
        split=split,
        transform = transform
    )
            

    dataloader =  DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=shuffle
    )      

    return dataloader