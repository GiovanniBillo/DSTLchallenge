import os
import numpy as np
import torch
from torch.utils.data import Dataset
from src.config import DATA_DIR, N_CLS, PATCH_SIZE
import random

class DSTLPatchFromFolderDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=PATCH_SIZE, thresholds=None, augment=True, n_samples=10000):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.thresholds = thresholds
        self.augment = augment
        self.n_samples = n_samples

        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        assert len(self.image_files) == len(self.mask_files), "Image and mask count mismatch."

        self.n_classes = np.load(os.path.join(mask_dir, self.mask_files[0])).shape[-1]
        if self.thresholds is None:
            self.thresholds = [0.05] * self.n_classes

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Pick a random image/mask pair
        i = random.randint(0, len(self.image_files) - 1)
        img = np.load(os.path.join(self.image_dir, self.image_files[i]))  # (H, W, C)
        msk = np.load(os.path.join(self.mask_dir, self.mask_files[i]))    # (H, W, N_CLS)

        H, W = img.shape[:2]
        xm, ym = H - self.patch_size, W - self.patch_size

        # Try multiple times to get a patch with relevant class content
        # TODO: should perhaos divide this into a separate function get_patches
        for _ in range(10):
            xc, yc = random.randint(0, xm), random.randint(0, ym)
            im = img[xc:xc+self.patch_size, yc:yc+self.patch_size]
            ms = msk[xc:xc+self.patch_size, yc:yc+self.patch_size]

            for j in range(self.n_classes):
                if np.sum(ms[:, :, j]) / (self.patch_size ** 2) > self.thresholds[j]:
                    if self.augment:
                        if random.random() > 0.5:
                            im, ms = im[::-1], ms[::-1]
                        if random.random() > 0.5:
                            im, ms = im[:, ::-1], ms[:, ::-1]
                    x = (np.transpose(im, (2, 0, 1)) * 2 - 1).astype(np.float32)
                    y = np.transpose(ms, (2, 0, 1)).astype(np.float32)
                    return torch.tensor(x), torch.tensor(y)

        # fallback if no patch passed threshold
        return self.__getitem__(idx)


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
