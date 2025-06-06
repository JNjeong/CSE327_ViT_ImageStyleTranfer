import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.utils import save_image
from PIL import Image

# 설정
content_dir = "data/content"
style_dir = "data/style"
output_dir = "output/cnn"
image_size = 224
num_steps = 100
style_weight = 1e4
content_weight = 1

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG19에 맞는 Normalize 값
normalization_mean = torch.tensor([0.485, 0.456, 0.406])
normalization_std = torch.tensor([0.229, 0.224, 0.225])

# 이미지 전처리
def image_loader(image_path, device):
    """
    Loads and preprocesses an image for VGG19 style transfer.
    Args:
        image_path (str): Path to the image file.
        device (torch.device): Device to load the image to.
    Returns:
        torch.Tensor: Preprocessed image tensor of shape [1, 3, H, W].
    """
    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_mean.tolist(),
                            std=normalization_std.tolist())
    ])
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

# 이미지 저장 (Unnormalize 포함)
def imsave(tensor, path):
    """
    Saves a tensor as an image after unnormalizing.
    Args:
        tensor (torch.Tensor): Image tensor to save.
        path (str): Output file path.
    Returns:
        None
    """
    unnormalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(normalization_mean, normalization_std)],
        std=[1/s for s in normalization_std]
    )
    image = tensor.clone().detach().cpu().squeeze(0)
    image = unnormalize(image)
    image = torch.clamp(image, 0, 1)
    save_image(image, path)

# 콘텐츠 손실
class ContentLoss(nn.Module):
    """
    Module to compute content loss (MSE) between input and target feature maps.
    Args:
        target (torch.Tensor): Target feature map.
    Returns:
        torch.Tensor: Input (unchanged).
    """
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# 스타일 손실
class StyleLoss(nn.Module):
    """
    Module to compute style loss (MSE between Gram matrices) for style transfer.
    Args:
        target_feature (torch.Tensor): Target style feature map.
    Returns:
        torch.Tensor: Input (unchanged).
    """
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

def gram_matrix(input):
    """
    Computes the Gram matrix for a given feature map.
    Args:
        input (torch.Tensor): Feature map of shape [B, C, H, W].
    Returns:
        torch.Tensor: Gram matrix of shape [B*C, B*C].
    """
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

# 스타일 전송 실행
def run_style_transfer(content_img, style_img, device, output_path):
    """
    Runs neural style transfer using VGG19.
    Args:
        content_img (torch.Tensor): Content image tensor.
        style_img (torch.Tensor): Style image tensor.
        device (torch.device): Device to run on.
        output_path (str): Path to save the output image.
    Returns:
        None
    """
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    normalization = nn.Sequential(
        transforms.Normalize(mean=normalization_mean.tolist(),
                             std=normalization_std.tolist())
    ).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential()
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_" + name, content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module("style_loss_" + name, style_loss)
            style_losses.append(style_loss)

    input_img = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([input_img], lr=0.01)

    for step in range(num_steps):
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_weight * style_score + content_weight * content_score
        loss.backward()
        optimizer.step()

    imsave(input_img, output_path)

# 메인 루프
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    content_images = sorted([f for f in os.listdir(content_dir) if f.endswith(('jpg', 'png'))])
    style_images = sorted([f for f in os.listdir(style_dir) if f.endswith(('jpg', 'png'))])

    for idx, content_name in enumerate(content_images):
        style_name = random.choice(style_images)  # style 개수가 적을 경우 랜덤 중복 허용

        content_path = os.path.join(content_dir, content_name)
        style_path = os.path.join(style_dir, style_name)
        output_path = os.path.join(output_dir, f"output_{idx}_{style_name.split('.')[0]}.png")

        content = image_loader(content_path, device)
        style = image_loader(style_path, device)

        print(f"▶ 스타일 전송 중: content={content_name}, style={style_name}")
        run_style_transfer(content, style, device, output_path)
        print(f"✅ 저장 완료: {output_path}") 