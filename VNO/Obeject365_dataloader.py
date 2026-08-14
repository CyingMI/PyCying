from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from collections import defaultdict  
import ijson
import torch
import random
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

class Objects365Dataset(Dataset):
    def __init__(self, root, annotation_file, transform):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.image_ids = []
        self.image_info = {}
        self.annotations_by_image = defaultdict(list)
        
        with open(annotation_file, "rb") as f:
            for image in ijson.items(f, "images.item"):
                self.image_ids.append(image["id"])
                image_id = image["id"]
                self.image_info[image_id] = image

        
        with open(annotation_file, "rb") as f:
            for anno in ijson.items(f, "annotations.item"):
                image_id = anno["image_id"]
                self.annotations_by_image[image_id].append(anno)
                
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = self._load_image_(image_id)
        annotations = self.annotations_by_image[image_id]
        target = self._parse_annotations_(annotations)
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.image_ids)
        

    def _load_image_(self, image_id):
        image_info = self.image_info[image_id]
        file_name = image_info["file_name"]
        image_path = self.root / file_name
        image = Image.open(image_path).convert("RGB")

        return image

    def _parse_annotations_(self, annotations):
        labels = []
        boxes = []
        for anno in annotations:
            labels.append(anno["category_id"])
            x, y, w, h = anno["bbox"]
            box = [x, y, x+w, y+h]
            boxes.append(box)
        target = {}
        target["labels"] = torch.tensor(labels, dtype=torch.float32) 
        target["boxes"] = torch.tensor(boxes, dtype = torch.float32).reshape(-1, 4)

        return target


class Obeject365Transform:
    def __init__(self, 
                 size,
                 horizontal_flip_prob=0.5,
                 normalize=True,
    ):
        self.size = size
        self.horizontal_flip_prob = horizontal_flip_prob
        self.normalize = normalize

    def __call__(self, image, target):

        old_w, old_h = image.size

        new_h, new_w = self.size

        image = F.resize(
            image,
            self.size,
            interpolation=InterpolationMode.BILINEAR
        )

        boxes = target["boxes"].clone()

        if boxes.numel() > 0:
            scale_x = new_w / old_w
            scale_y = new_h / old_h

            boxes[:, [0, 2]] *= scale_x

            boxes[:, [1, 3]] *= scale_y

        if random.random() < self.horizontal_flip_prob:

            image = F.hflip(image)

            if boxes.numel() > 0:
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()

                boxes[:, 0] = new_w - old_x2
                boxes[:, 2] = new_w - old_x1

        if boxes.numel() > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
                min=0,
                max=new_w
            )

            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
                min=0,
                max=new_h
            )

        target["boxes"] = boxes

        image = F.to_tensor(image)

        if self.normalize:
            image = F.normalize(
                image,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

        return image, target

def collate_fn(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets

def build_dataloader(
        root,
        annotation_file,
        transform,
        batch_size,
        shuffle,
        num_workers,
        collate_fn
):

    dataset = Objects365Dataset(
    root = root,
    annotation_file=annotation_file,
    transform = transform,
    )

    dataloader =  DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn = collate_fn
    ) 

    return dataloader