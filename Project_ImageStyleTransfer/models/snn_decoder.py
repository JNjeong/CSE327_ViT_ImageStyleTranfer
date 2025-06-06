import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    """
    U-Net style convolutional block.
    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
    Returns:
        torch.Tensor: Output feature map.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(True)
        )
    def forward(self, x):
        """
        Forward pass for UNetBlock.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)

class SNNDecoder(nn.Module):
    """
    Decoder that reconstructs an image from content features only.
    Input shape: [B, 3, 224, 224]
    Returns:
        torch.Tensor: Output image tensor of shape [B, 3, 224, 224].
    """
    def __init__(self):
        super(SNNDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        """
        Forward pass for SNNDecoder.
        Args:
            x (torch.Tensor): Input tensor [B, 3, 224, 224].
        Returns:
            torch.Tensor: Output tensor [B, 3, 224, 224].
        """
        out = self.model(x)
        out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        return out
