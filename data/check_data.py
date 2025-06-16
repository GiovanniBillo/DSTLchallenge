import os
import numpy as np

def inspect_npy_dataset(base_dir="processed/train"):
    img_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

    assert len(img_files) == len(mask_files), "Mismatch between number of images and masks!"

    print(f"\n🗂 Total files: {len(img_files)} images + {len(mask_files)} masks")

    total_img_size = 0
    total_mask_size = 0
    img_shapes = set()
    mask_shapes = set()

    for f_img, f_mask in zip(img_files, mask_files):
        img_path = os.path.join(img_dir, f_img)
        mask_path = os.path.join(mask_dir, f_mask)

        img = np.load(img_path)
        mask = np.load(mask_path)

        total_img_size += os.path.getsize(img_path)
        total_mask_size += os.path.getsize(mask_path)

        img_shapes.add(img.shape)
        mask_shapes.add(mask.shape)

    print(f"\n📏 Unique image shapes: {img_shapes}")
    print(f"📏 Unique mask shapes:  {mask_shapes}")
    print(f"🧠 Image dtype:         {img.dtype}")
    print(f"🧠 Mask dtype:          {mask.dtype}")
    print(f"💾 Total image size:    {total_img_size / 1e6:.2f} MB")
    print(f"💾 Total mask size:     {total_mask_size / 1e6:.2f} MB")
    print(f"📦 Total dataset size:  {(total_img_size + total_mask_size) / 1e6:.2f} MB")

if __name__ == "__main__":
    inspect_npy_dataset()

