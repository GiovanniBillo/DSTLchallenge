import torch
import torch.nn.functional as F
from torch import nn
from transformers import ViTImageProcessor, ViTModel, ViTConfig 
from timm.models.vision_transformer import VisionTransformer
from src.config import PRETRAINED_VIT_PATH, N_CLS, IMG_SIZE, IN_CHANS, PATCH_SIZE, TRAIN_IMG_SIZE

class CustomViT(VisionTransformer):
    def __init__(self, img_size=160, patch_size=16, in_chans=10, **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            **kwargs
        )
class CustomPretrainedViT(nn.Module):
    def __init__(
        self,
        img_size=IMG_SIZE,
        patch_size=16,
        in_chans=IN_CHANS, # only to be able to load the model
        num_classes=10,
        model_name_or_path=PRETRAINED_VIT_PATH,
        **kwargs
    ):
        super().__init__()

        config = ViTConfig.from_pretrained(model_name_or_path)
        config.num_channels = in_chans
        config.image_size = TRAIN_IMG_SIZE 
        self.vit = ViTModel(config)
        state_dict = ViTModel.from_pretrained(model_name_or_path).state_dict()

        # Resize position embeddings before loading
        old_grid_size = 14  # because 224 / 16 = 14
        new_grid_size = 25  # because 400 / 16 = 25
        state_dict["embeddings.position_embeddings"] = self._resize_pos_embed(
            pos_embed=state_dict["embeddings.position_embeddings"],
            old_grid_size=old_grid_size,
            new_grid_size=new_grid_size
        )

        # Load all weights except the first layer
        del state_dict['embeddings.patch_embeddings.projection.weight']
        del state_dict['embeddings.patch_embeddings.projection.bias']
        self.vit.load_state_dict(state_dict, strict=False)

        # # Load pretrained model
        # self.vit = ViTModel.from_pretrained(model_name_or_path)

        # Patch the input layer if needed
        if IN_CHANS != 3:
            # print("entered the patch function")
            self._patch_input_layer(IN_CHANS)
            self.vit.config.num_channels = IN_CHANS 
            self.vit.embeddings.config.num_channels = IN_CHANS 


        # Resize positional embeddings if needed
        # print(config.image_size, img_size)
        # if img_size != config.image_size or patch_size != config.patch_size:
        #     print("fixed positional embeddings")
        #     self._resize_positional_embeddings()

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def _patch_input_layer(self, in_chans):
        old_conv = self.vit.embeddings.patch_embeddings.projection
        new_conv = nn.Conv2d(
            in_channels=in_chans,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )

        with torch.no_grad():
            if old_conv.weight.shape[1] == 3:
                # Copy pretrained weights into new conv
                new_conv.weight[:, :3] = old_conv.weight
                if in_chans > 3:
                    # Copy or average channel weights for remaining channels
                    new_conv.weight[:, 3:] = old_conv.weight[:, :1].repeat(1, in_chans - 3, 1, 1)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

            new_conv.bias = old_conv.bias

        self.vit.embeddings.patch_embeddings.projection = new_conv
    def _resize_pos_embed(self, pos_embed, old_grid_size, new_grid_size):
        """
        Resize the position embeddings from old grid size (e.g. 14x14) to new grid size (e.g. 25x25).
        
        Parameters:
            pos_embed (torch.Tensor): shape (1, 1+N, D), where N = H_old * W_old, D = hidden size
            old_grid_size (int or tuple): old patch grid size (e.g. 14 or (14,14))
            new_grid_size (int or tuple): new patch grid size (e.g. 25 or (25,25))
        
        Returns:
            new_pos_embed (torch.Tensor): resized positional embedding of shape (1, 1+H_new*W_new, D)
        """
        if isinstance(old_grid_size, int):
            old_grid_size = (old_grid_size, old_grid_size)
        if isinstance(new_grid_size, int):
            new_grid_size = (new_grid_size, new_grid_size)
            
        # Separate CLS token and patch tokens
        cls_token = pos_embed[:, :1, :]  # (1, 1, D)
        patch_pos_embed = pos_embed[:, 1:, :]  # (1, N, D)
        
        # Reshape to (1, D, H_old, W_old)
        D = patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.reshape(1, old_grid_size[0], old_grid_size[1], D).permute(0, 3, 1, 2)

        # Interpolate
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=new_grid_size, mode='bilinear', align_corners=False
        )

        # Reshape back to (1, H_new*W_new, D)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_grid_size[0]*new_grid_size[1], D)

        # Concatenate CLS token back
        new_pos_embed = torch.cat([cls_token, patch_pos_embed], dim=1)
        
        return new_pos_embed
    # def _resize_positional_embeddings(self):
    #     old_pos_embed = self.vit.embeddings.position_embeddings  # [1, 197, D]
    #     cls_token = old_pos_embed[:, :1]     # [1, 1, D]
    #     patch_pos_embed = old_pos_embed[:, 1:]  # [1, 196, D]

    #     old_grid_size = int(patch_pos_embed.shape[1] ** 0.5)
    #     patch_pos_embed = patch_pos_embed.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)

    #     new_grid_size = self.img_size // self.patch_size
    #     resized_embed = F.interpolate(patch_pos_embed, size=(new_grid_size, new_grid_size), mode='bilinear')
    #     resized_embed = resized_embed.permute(0, 2, 3, 1).reshape(1, new_grid_size ** 2, -1)

    #     new_pos_embed = torch.cat([cls_token, resized_embed], dim=1)
    #     self.vit.embeddings.position_embeddings = nn.Parameter(new_pos_embed)

    def forward(self, x):
        # print(x.size())
        # print(self.vit.config.num_channels)
        # print(self.vit.embeddings.config)
        # print(self.vit.embeddings.patch_embeddings.projection.in_channels)
        outputs = self.vit(pixel_values=x)
        pooled_output = outputs.pooler_output  # (B, hidden_dim)
        return self.classifier(pooled_output)

# class CustomPretrainedViT(torch.nn.Module):
#     def __init__(self, model_name_or_path=PRETRAINED_VIT_PATH, num_classes=N_CLS):
#         super().__init__()
#         self.vit = ViTModel.from_pretrained(model_name_or_path)
#         self.classifier = torch.nn.Linear(self.vit.config.hidden_size, num_classes)

#     def forward(self, x):
#         # x shape: (batch_size, channels, height, width) → convert to (batch_size, 3, 224, 224) if needed
#         # You should preprocess x first if using HuggingFace ViTModel
#         outputs = self.vit(pixel_values=x)
#         pooled_output = outputs.pooler_output
#         return self.classifier(pooled_output)


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

