import torch
from timm.models.vision_transformer import VisionTransformer

class CustomViT(VisionTransformer):
    def __init__(self, img_size=160, patch_size=16, in_chans=10, **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            **kwargs
        )

def main():
    # Dummy input: batch of 8, 10-channel, 160x160 images
    x = torch.randn(8, 10, 160, 160)

    # Create the model
    model = CustomViT(
        img_size=160,
        patch_size=16,
        in_chans=10,
        embed_dim=768,     # Base size; use 1024 for ViT-L
        depth=12,          # ViT-B=12, ViT-L=24
        num_heads=12,
        num_classes=0      # No classification head yet
    )

    print(model)
    out = model.forward_features(x)
    print("Output shape:", out.shape)  # Expect (8, 768) from [CLS] token

if __name__ == "__main__":
    main()

