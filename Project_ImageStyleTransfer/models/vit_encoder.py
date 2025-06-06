import torch
import torch.nn as nn
import timm

class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder for extracting patch features from images.
    Args:
        model_name (str): Name of the timm ViT model to use.
    Returns:
        torch.Tensor or list[torch.Tensor]: Feature map(s) from ViT.
    """
    def __init__(self, model_name='vit_base_patch16_224.augreg2_in21k_ft_in1k'):
        super(ViTEncoder, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        self.patch_size = self.vit.patch_embed.patch_size
        self.embedding_dim = self.vit.embed_dim
        self.vit.head = nn.Identity()

    def forward(self, x):
        """
        Extracts the final feature map from the input image using ViT.
        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W].
        Returns:
            torch.Tensor: Feature map of shape [B, C, H', W'].
        """
        B, C, H, W = x.shape
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        x = x[:, 1:]
        h = H // self.patch_size[0]
        w = W // self.patch_size[1]
        x = x.permute(0, 2, 1).contiguous().view(B, self.embedding_dim, h, w)
        print(f"[ViTEncoder] Output shape: {x.shape}")
        print(f"[ViTEncoder] min={x.min().item():.4f} max={x.max().item():.4f} mean={x.mean().item():.4f}")
        return x

    def forward_multi(self, x, layers=[3, 6, 9, 12]):
        """
        Extracts feature maps from multiple ViT layers.
        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W].
            layers (list[int]): List of block indices to extract features from.
        Returns:
            list[torch.Tensor]: List of feature maps from specified layers.
        """
        B, C, H, W = x.shape
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        feats = []
        for i, blk in enumerate(self.vit.blocks, 1):
            x = blk(x)
            if i in layers:
                x_feat = self.vit.norm(x)
                x_feat = x_feat[:, 1:]
                h = H // self.patch_size[0]
                w = W // self.patch_size[1]
                x_feat = x_feat.permute(0, 2, 1).contiguous().view(B, self.embedding_dim, h, w)
                feats.append(x_feat)
        return feats
