import torch
import numpy as np
import json
import cv2
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from cityscapesscripts.helpers.labels import labels
label_to_Id = {label.name: label.trainId for label in labels}

def fourier_inpaint(image: np.ndarray, new_h: int, new_w: int):
    H, W, _ = image.shape
    if new_h >= H and new_w >= W:
        return image.copy()

    y_idx = np.arange(H)[:, None] % new_h   # (H, 1)
    x_idx = np.arange(W)[None, :] % new_w   # (1, W)
    filled = image[y_idx, x_idx]            # 周期延拓后的完整图像 (H, W, C)

    return filled.astype(np.uint8)

class CityscapesDataset(Dataset):
    def __init__(self, root, split, target_size,num_query):
        self.root = Path(root)
        self.split = split
        self.size = target_size
        self.num_query = num_query
        
        gt_dir = self.root / 'gtFine' / split
        self.json_files = sorted(gt_dir.glob('*/*_gtFine_polygons.json'))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)

        img_path = Path(str(json_path).replace('_gtFine_polygons.json', '_leftImg8bit.png').replace('gtFine', 'leftImg8bit'))
        image = np.array(Image.open(img_path).convert('RGB'))  

        h, w = image.shape[:2]
        target_h, target_w = self.size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_h, pad_w = target_h - new_h, target_w - new_w
        img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)

        padding_mask = np.ones((target_h, target_w), dtype=bool)
        padding_mask[:new_h, :new_w] = False

        filled_img = fourier_inpaint(img_padded, new_h, new_w)
        img_tensor = self.transform(filled_img)

        masks_tensor = torch.zeros((self.num_query, target_h, target_w), dtype=torch.uint8)
        labels_tensor = torch.full((self.num_query,), -1, dtype=torch.long)
        bboxes_tensor = torch.zeros((self.num_query, 4), dtype=torch.float32)

        valid_idx = 0
        for obj in data['objects']:
            label_str = obj['label']
            label_id = label_to_Id.get(label_str, -1)   # -1 表示忽略
            if label_id == -1 or label_id == 255:          # 跳过无效类别
                continue

            polygon = np.array(obj['polygon'], dtype=np.int32)  # (N, 2)
            x_coords = polygon[:, 0]
            y_coords = polygon[:, 1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            polygon_scaled = (polygon * scale).astype(np.int32)
            bbox_scaled = [x_min * scale/target_w, y_min * scale/target_h, x_max * scale/target_w, y_max * scale/target_h]

            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_scaled], 1)   # 填充值为 1

            masks_tensor[valid_idx] = torch.as_tensor(mask, dtype=torch.uint8)
            labels_tensor[valid_idx] = torch.as_tensor(label_id, dtype=torch.long)
            bboxes_tensor[valid_idx] = torch.as_tensor(bbox_scaled, dtype=torch.float32)

            valid_idx += 1


        padding_mask_tensor = torch.as_tensor(padding_mask, dtype=torch.bool)

        return img_tensor, masks_tensor, labels_tensor, padding_mask_tensor, bboxes_tensor