import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion(nn.Module):
    """
    Module for feature fusion using AdaIN and weighted blending for style transfer.
    Returns:
        torch.Tensor: Fused feature map.
    """
    def __init__(self, *args, **kwargs):
        super(Fusion, self).__init__()
    def calc_mean_std(self, feat, eps=1e-5):
        """
        Calculates mean and standard deviation for each channel in a feature map.
        Args:
            feat (torch.Tensor): Feature map of shape [B, C, H, W].
            eps (float): Small value to avoid division by zero.
        Returns:
            tuple: (mean, std) each of shape [B, C, 1, 1].
        """
        size = feat.size()
        b, c = size[:2]
        feat_var = feat.view(b, c, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(b, c, 1, 1)
        feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        return feat_mean, feat_std
    def adain(self, content_feat, style_feat):
        """
        Applies Adaptive Instance Normalization (AdaIN) to content and style features.
        Args:
            content_feat (torch.Tensor): Content feature map.
            style_feat (torch.Tensor): Style feature map.
        Returns:
            torch.Tensor: AdaIN-transformed feature map.
        """
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)
        normalized = (content_feat - content_mean) / (content_std + 1e-8)
        return normalized * style_std + style_mean
    def forward(self, content_feat, style_feat, style_weight=None):
        """
        Fuses content and style features using AdaIN and weighted blending.
        Args:
            content_feat (torch.Tensor): Content feature map.
            style_feat (torch.Tensor): Style feature map.
            style_weight (float, optional): Weight for style blending (default: random).
        Returns:
            torch.Tensor: Fused feature map.
        """
        if style_weight is None:
            style_weight = torch.rand(1).item()  # 0~1 랜덤
        adain_feat = self.adain(content_feat, style_feat)
        fused = (1-style_weight) * content_feat + style_weight * adain_feat
        return fused
