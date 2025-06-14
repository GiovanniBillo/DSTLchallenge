import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from src.config import DATA_DIR, N_CLS
from src.models.unet import UNet
from src.datasets.dstl_dataset import DSTLDataset, DSTLDatasetFromArrays, DSTLPatchFromFolderDataset


def train_model(dataset, epochs=5, batch_size=8, lr=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = UNet(in_channels=8, out_channels=N_CLS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    return model

if __name__ == "__main__":
    # Choose dataset type here:
    use_array = False
    use_patches = True

    if use_array:
        x_path = os.path.join(DATA_DIR, f"x_trn_{N_CLS}.npy")
        y_path = os.path.join(DATA_DIR, f"y_trn_{N_CLS}.npy")
        dataset = DSTLDatasetFromArrays(x_path, y_path)
    if use_patches:
        image_dir = os.path.join(DATA_DIR, "processed/train/images")
        mask_dir = os.path.join(DATA_DIR, "processed/train/masks")
        dataset = DSTLPatchFromFolderDataset(image_dir, mask_dir)
    else:
        image_dir = os.path.join(DATA_DIR, "processed/train/images")
        mask_dir = os.path.join(DATA_DIR, "processed/train/masks")
        dataset = DSTLDataset(image_dir, mask_dir)

    print(f"Dataset size: {len(dataset)}")
    trained_model = train_model(dataset)

