import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class StyleTransferDataset(Dataset):
    """
    Custom Dataset for style transfer.
    Args:
        content_dir (str): Directory containing content images.
        style_dir (str): Directory containing style images.
        image_size (int): Size to resize images to (default: 224).
    Returns:
        tuple: (content_tensor, style_tensor, content_filename, style_filename)
    """
    def __init__(self, content_dir, style_dir, image_size=224):
        self.content_paths = [os.path.join(content_dir, f) for f in sorted(os.listdir(content_dir)) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.style_paths = [os.path.join(style_dir, f) for f in sorted(os.listdir(style_dir)) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.style_idx_list = list(range(len(self.style_paths)))
        self.style_used_count = 0
        self.random_mode = False
        
    def __len__(self):
        """
        Returns the number of content images.
        Returns:
            int: Number of content images.
        """
        return len(self.content_paths)
        
    def __getitem__(self, idx):
        """
        Returns a tuple of (content image tensor, style image tensor, content filename, style filename).
        Args:
            idx (int): Index of the content image.
        Returns:
            tuple: (content_tensor, style_tensor, content_filename, style_filename)
        """
        # content 이미지는 1번부터 순서대로
        content_path = self.content_paths[idx]
        content_img = Image.open(content_path).convert('RGB')
        content_tensor = self.transform(content_img)
        
        # style 이미지는 1회씩 모두 사용 후 랜덤 배정
        if not self.random_mode:
            style_idx = self.style_idx_list[self.style_used_count]
            self.style_used_count += 1
            if self.style_used_count >= len(self.style_idx_list):
                self.random_mode = True
        else:
            style_idx = torch.randint(len(self.style_paths), (1,)).item()
        style_path = self.style_paths[style_idx]
        style_img = Image.open(style_path).convert('RGB')
        style_tensor = self.transform(style_img)
        
        return content_tensor, style_tensor, os.path.basename(content_path), os.path.basename(style_path) 