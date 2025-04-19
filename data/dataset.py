import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from .masking import apply_random_mask, get_mask_coverage_range

class InpaintingDataset(Dataset):
    def __init__(self, image_dir, transform=None, current_epoch=0, total_epochs=100):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        
    def __len__(self):
        return len(self.image_files)
    
    def update_epoch(self, epoch):
        """Update the current epoch to adjust masking difficulty"""
        self.current_epoch = epoch
    
    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        # Convert to numpy for masking
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).numpy() * 255
            image_np = image_np.astype(np.uint8)
        else:
            image_np = np.array(image)
        
        # Get appropriate coverage range based on training progress
        min_coverage, max_coverage = get_mask_coverage_range(
            self.current_epoch, self.total_epochs
        )
        
        # Apply random mask
        masked_image, mask, _ = apply_random_mask(image_np, min_coverage, max_coverage)
        
        # Convert back to tensors
        masked_image = torch.FloatTensor(masked_image.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask = torch.FloatTensor(mask.astype(np.float32) / 255.0).unsqueeze(0)
        
        # Original image as target
        if isinstance(image, torch.Tensor):
            target = image
        else:
            target = torch.FloatTensor(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1)
        
        # Create input tensor with mask channel
        input_tensor = torch.cat([masked_image, mask], dim=0)
        
        return {
            'input': input_tensor,  # Masked image + mask channel
            'target': target,       # Original image
            'mask': mask            # Mask only
        }