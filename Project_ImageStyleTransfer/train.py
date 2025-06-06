import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from models.vit_encoder import ViTEncoder
from models.snn_decoder import SNNDecoder
from models.fusion import Fusion
from models.style_loss import StyleLoss
from utils.dataset import StyleTransferDataset
from torchvision import models

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_dir = "data/content"
style_dir = "data/style"
output_dir = "output/hybrid_vitstyle"
os.makedirs(output_dir, exist_ok=True)

# 모델 초기화
vit_encoder = ViTEncoder().to(device)
decoder = SNNDecoder().to(device)
fusion = Fusion().to(device)
style_loss_fn = StyleLoss().to(device)

# VGG perceptual loss용 feature extractor
class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate VGG19 features for perceptual loss.
    Args:
        layer_idx (int): Index of the last VGG layer to use (default: 8 for relu2_2).
    Returns:
        torch.Tensor: Extracted feature map.
    """
    def __init__(self, layer_idx=8):  # relu2_2
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice = nn.Sequential(*[vgg[i] for i in range(layer_idx+1)])
        for param in self.slice.parameters():
            param.requires_grad = False
    def forward(self, x):
        return self.slice(x)

vgg_extractor = VGGFeatureExtractor().to(device)

# 데이터셋 및 로더
dataset = StyleTransferDataset(content_dir, style_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

content_weight = 1.0
perceptual_weight = 1.0

def train_vit_style(epochs=10):
    """
    Trains the ViT+SNN/Hybrid style transfer model.
    Args:
        epochs (int): Number of training epochs.
    Returns:
        None
    """
    optimizer = optim.Adam(list(decoder.parameters()) + list(fusion.parameters()), lr=1e-4)
    for epoch in range(epochs):
        # epoch별 style_weight 점진적 증가 (더 강하게)
        if epoch < 2:
            style_weight = 0.1
        elif epoch < 4:
            style_weight = 0.3
        elif epoch < 7:
            style_weight = 0.6
        else:
            style_weight = 1.0
        for i, (content, style, content_name, style_name) in enumerate(loader):
            content = content.to(device)
            style = style.to(device)
            # 1. content 이미지를 decoder에 직접 입력
            output = decoder(content)
            # 2. ViT feature 추출 (style 정보만)
            with torch.no_grad():
                style_feats = vit_encoder.forward_multi(style)
                content_feats = vit_encoder.forward_multi(content)
            output_feats = vit_encoder.forward_multi(output)
            # 3. loss 계산 (content: MSE, style: ViT feature 기반 style loss)
            c_loss = F.mse_loss(output, content)
            pixel_content_loss = F.l1_loss(output, content)
            # perceptual loss (VGG relu2_2)
            vgg_out = vgg_extractor(output)
            vgg_content = vgg_extractor(content)
            perceptual_loss = F.mse_loss(vgg_out, vgg_content)
            _, s_loss = style_loss_fn(output_feats, content_feats, style_feats)
            # content loss = MSE + L1 + perceptual
            total_content_loss = c_loss + pixel_content_loss + perceptual_weight * perceptual_loss
            loss = content_weight * total_content_loss + style_weight * s_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # content_name, style_name이 tuple일 경우 str로 변환
            c_name = content_name[0] if isinstance(content_name, (tuple, list)) else content_name
            s_name = style_name[0] if isinstance(style_name, (tuple, list)) else style_name
            save_path = os.path.join(
                output_dir,
                f"vitstyle_ep{epoch+1}_cw{content_weight}_sw{style_weight}_content{os.path.splitext(c_name)[0]}_style{os.path.splitext(s_name)[0]}.png"
            )
            save_image(output.clamp(0, 1), save_path)
        print(f"[ViT-style] Epoch {epoch+1} completed. Loss: {loss.item():.4f} (style_weight={style_weight})")
    print("[ViT-style] 학습 완료!")
    torch.save(decoder.state_dict(), "checkpoints/snn_decoder_vitstyle.pth")
    torch.save(fusion.state_dict(), "checkpoints/fusion_vitstyle.pth")

if __name__ == "__main__":
    train_vit_style(epochs=10)
