import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle

from src.config import DATA_DIR, N_CLS, PRETRAINED_VIT_PATH, IN_CHANS, IMG_SIZE, PATCH_SIZE
from src.models.unet import UNet
from src.datasets.dstl_dataset import DSTLDataset, DSTLDatasetFromArrays, DSTLPatchFromFolderDataset
from src.utils.loss_utils import JaccardLoss, BCEJaccardLoss # 2 alternatives for loss, have to see which one is better
from src.models.CustomViT import CustomViT, CustomPretrainedViT
import argparse


parser=argparse.ArgumentParser(description="argument parser for train")
parser.add_argument("train_models")
args=parser.parse_args()

def train_unet(dataset, loss_fn, epochs=5, batch_size=8, lr=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = UNet(in_channels=IN_CHANS, out_channels=N_CLS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = loss_fn 
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

    # Save model weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/unet_latest.pt")
    print("Saved UNet weights to models/unet_latest.pt")

    return model

def train_ViT(dataset, loss_fn, epochs=5, batch_size=8, lr=1e-3, from_pretrained=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if from_pretrained:
        model = CustomPretrainedViT(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_chans=IN_CHANS,
            num_classes=N_CLS,
            model_name_or_path=PRETRAINED_VIT_PATH
        )

        print("Loaded TRAINED ViT model")
    else:
        model = CustomViT(
        img_size=160,
        patch_size=PATCH_SIZE,
        in_chans=8,
        embed_dim=768,     # Base size; use 1024 for ViT-L
        depth=12,          # ViT-B=12, ViT-L=24
        num_heads=12,
        num_classes=N_CLS     
        )
        print("Loaded UNTRAINED ViT model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = loss_fn 
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

    # Save model weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/vit_latest.pt")
    print("Saved ViT weights to models/vit_latest.pt")

    return model


if __name__ == "__main__":
    # Choose dataset type here:
    loss = JaccardLoss()
    # use_array = False
    use_patches = False 
    
    # if use_array:
        # x_path = os.path.join(DATA_DIR, f"x_trn_{N_CLS}.npy")
        # y_path = os.path.join(DATA_DIR, f"y_trn_{N_CLS}.npy")
        # dataset = DSTLDatasetFromArrays(x_path, y_path)
    if use_patches:
        image_dir = os.path.join(DATA_DIR, "processed/train/images")
        mask_dir = os.path.join(DATA_DIR, "processed/train/masks")
        dataset = DSTLPatchFromFolderDataset(image_dir, mask_dir)
    else:
        image_dir = os.path.join(DATA_DIR, "processed/train/images")
        mask_dir = os.path.join(DATA_DIR, "processed/train/masks")
        dataset = DSTLDataset(image_dir, mask_dir)
    
    print(f"Dataset size: {len(dataset)}")

    if args.train_models=="all" :
        trained_unet = train_unet(dataset, loss_fn=loss)
        trained_ViT = train_ViT(dataset, loss_fn=loss)
    elif args.train_models=="vit":
        trained_ViT = train_ViT(dataset, loss_fn=loss)
    elif args.train_models=="unet":
        trained_unet = train_unet(dataset, loss_fn=loss)
    else:
        print("unrecognized model.")
