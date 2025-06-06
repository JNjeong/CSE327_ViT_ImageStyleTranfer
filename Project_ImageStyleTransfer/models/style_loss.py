import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleLoss(nn.Module):
    """
    Computes style and content loss for style transfer using Gram matrices and feature maps.
    Returns:
        tuple: (content_loss, style_loss)
    """
    def __init__(self):
        super(StyleLoss, self).__init__()
    def gram_matrix(self, x):
        """
        Computes the Gram matrix for a given feature map.
        Args:
            x (torch.Tensor): Feature map of shape [B, C, H, W].
        Returns:
            torch.Tensor: Gram matrix of shape [B, C, C].
        """
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    def forward(self, output_features, content_features, style_features):
        """
        Computes content and style loss between output, content, and style features.
        Args:
            output_features (list[torch.Tensor]): Output feature maps from ViT.
            content_features (list[torch.Tensor]): Content feature maps from ViT.
            style_features (list[torch.Tensor]): Style feature maps from ViT.
        Returns:
            tuple: (content_loss, style_loss)
        """
        # content loss: 여러 계층 평균
        content_loss = 0
        for out_f, cont_f in zip(output_features, content_features):
            content_loss += F.mse_loss(out_f, cont_f)
        content_loss /= len(output_features)
        # style loss: mid(9), late(12) layer 평균
        style_loss = 0
        for idx in [-2, -1]:  # 9, 12번째 계층
            out_f = output_features[idx]
            sty_f = style_features[idx]
            output_gram = self.gram_matrix(out_f)
            style_gram = self.gram_matrix(sty_f)
            style_loss += F.mse_loss(output_gram, style_gram)
        style_loss /= 2
        return content_loss, style_loss 