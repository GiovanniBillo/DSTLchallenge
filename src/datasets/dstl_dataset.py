import os
import numpy as np
import torch
from torch.utils.data import Dataset
from src.config import DATA_DIR, N_CLS

class DSTLDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.ids = sorted([
            fname for fname in os.listdir(self.image_dir)
            if fname.endswith('.npy')
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fname = self.ids[idx]
        image_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        image = np.load(image_path)  # shape: (H, W, C)
        mask = np.load(mask_path)    # shape: (H, W, N_CLS)

        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))  # to (C, H, W)
        if mask.ndim == 3:
            mask = np.transpose(mask, (2, 0, 1))    # to (N_CLS, H, W)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

class DSTLDatasetFromArrays(Dataset):
    def __init__(self, images_path=None, masks_path=None, transform=None):
        """
        Args:
            images_path (str): Path to the x_trn_{N_CLS}.npy file.
            masks_path (str): Path to the y_trn_{N_CLS}.npy file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = np.load(images_path)  # shape: (N, H, W, C)
        self.masks = np.load(masks_path)    # shape: (N, H, W, N_CLS)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]  # (H, W, C)
        mask = self.masks[idx]    # (H, W, N_CLS)

        image = np.transpose(image, (2, 0, 1))  # to (C, H, W)
        mask = np.transpose(mask, (2, 0, 1))    # to (N_CLS, H, W)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    image_dir = os.path.join(DATA_DIR, "processed/train/images")
    mask_dir = os.path.join(DATA_DIR, "processed/train/masks")

    dataset = DSTLDataset(image_dir=image_dir, mask_dir=mask_dir)

    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for i, (images, masks) in enumerate(dataloader):
        print(f"Batch {i}")
        print(f"  Images shape: {images.shape}")  # [B, C, H, W]
        print(f"  Masks shape:  {masks.shape}")   # [B, N_CLS, H, W]
        # break


    print(f"{'-'*30} Check with Dataset array {'-'*30}")

    x_path = f"{DATA_DIR}/x_trn_{N_CLS}.npy"
    y_path = f"{DATA_DIR}/y_trn_{N_CLS}.npy"

    dataset = DSTLDatasetFromArrays(images_path=x_path, masks_path=y_path)

    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for i, (images, masks) in enumerate(dataloader):
        print(f"Batch {i}")
        print(f"  Images shape: {images.shape}")  # [B, C, H, W]
        print(f"  Masks shape:  {masks.shape}")   # [B, N_CLS, H, W]
        break
