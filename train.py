import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from model import UNet
import torchvision.transforms.functional as F
# Custom Dataset class
class WaterSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir)])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Load and convert without resizing
        if self.transform:
            image = self.transform(image)
            # image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            mask = self.transform(mask)
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
            if random.random() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)
            # if random.random() > 0.5:
            #     if random.random() > 0.5:
            #         image = F.rotate(image, angle=30)
            #         mask = F.rotate(mask, angle=30)
            #     else:
            #         image = F.rotate(image, angle=-30)
            #         mask = F.rotate(mask, angle=-30)
        
        # Normalize mask to 0 and 1
        mask = (mask > 0.5).float()
        
        return image, mask
    
class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Soft Dice Loss for image segmentation.
    
    Args:
        weight (float): Weight for BCE loss (default: 1.0)
        dice_weight (float): Weight for Dice loss (default: 1.0)
        smooth (float): Small smoothing constant to avoid division by zero (default: 1e-7)
    """
    def __init__(self, weight=1.0, dice_weight=1.0, smooth=1e-8):
        super(BCEDiceLoss, self).__init__()
        self.weight = weight
        self.dice_weight = dice_weight
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Compute combined BCE and Soft Dice Loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (sigmoid output)
            targets (torch.Tensor): Ground truth binary masks
        
        Returns:
            torch.Tensor: Combined loss value
        """
        # Binary Cross-Entropy Loss
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets)
        
        # Soft Dice Loss
        # Flatten predictions and targets
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Compute intersection and union
        intersection = (inputs_flat * targets_flat).sum()
        
        # Soft Dice Loss calculation
        dice_loss = 1 - (2. * intersection + self.smooth) / \
                    (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        # Combine losses
        combined_loss = (self.weight * bce_loss) + (self.dice_weight * dice_loss)
        
        return combined_loss


class DiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Soft Dice Loss for image segmentation.
    
    Args:
        weight (float): Weight for BCE loss (default: 1.0)
        dice_weight (float): Weight for Dice loss (default: 1.0)
        smooth (float): Small smoothing constant to avoid division by zero (default: 1e-7)
    """
    def __init__(self, weight=1.0, dice_weight=1.0, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.dice_weight = dice_weight
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Compute combined BCE and Soft Dice Loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (sigmoid output)
            targets (torch.Tensor): Ground truth binary masks
        
        Returns:
            torch.Tensor: Combined loss value
        """
        
        # Soft Dice Loss
        # Flatten predictions and targets
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Compute intersection and union
        intersection = (inputs_flat * targets_flat).sum()
        
        # Soft Dice Loss calculation
        dice_loss = 1 - (2. * intersection + self.smooth) / \
                    (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        
        return dice_loss

# Training function
def train_model(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    running_loss = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(train_loader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate IoU
            pred = (outputs > 0.5).float()
            intersection = (pred * masks).sum((1, 2, 3))
            union = (pred + masks).gt(0).sum((1, 2, 3))
            iou = (intersection + 1e-8) / (union + 1e-8)
            
            running_loss += loss.item()
            running_iou += iou.mean().item()
    
    return running_loss / len(train_loader), running_iou / len(train_loader)

# Validation function with IoU calculation
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate IoU
            pred = (outputs > 0.5).float()
            intersection = (pred * masks).sum((1, 2, 3))
            union = (pred + masks).gt(0).sum((1, 2, 3))
            iou = (intersection + 1e-8) / (union + 1e-8)
            
            running_loss += loss.item()
            running_iou += iou.mean().item()
    
    return running_loss / len(val_loader), running_iou / len(val_loader)

# Main training pipeline
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)

    dir_img = 'training_dataset/preprocessed'
    dir_mask = 'training_dataset/mask'
    
    # Hyperparameters
    batch_size = 5
    num_epochs = 100
    learning_rate = 1e-4
    val_split = 0.2  # 20% for validation
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create full dataset
    full_dataset = WaterSegmentationDataset(dir_img, dir_mask, transform=transform)
    
    # Calculate lengths for split
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Total images: {dataset_size}")
    print(f"Training images: {train_size}")
    print(f"Validation images: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=4, drop_last=False)
    
    # Initialize model, criterion, and optimizer
    model = UNet().to(device)
    criterion = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5)

    # Create checkpoint directory if it doesn't exist
    os.makedirs('checkpoint', exist_ok=True)
    
    # Training loop
    best_val_iou = 0.0
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Training phase
        train_loss, train_iou = train_model(model, train_loader, criterion, optimizer, 
                               device, epoch)
        
        # Validation phase
        val_loss, val_iou = validate_model(model, val_loader, criterion, device)
        scheduler.step(train_loss)
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Training IoU: {train_iou:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation IoU: {val_iou:.4f}')
        print(f'Last learning rate: {scheduler._last_lr[0]}')
        
        # Save checkpoint if IoU improved
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, f'checkpoint/best_model.pth')
            print(f'New best model saved! IoU: {val_iou:.4f}')
        
        # Save periodic checkpoint
        if epoch > -1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, f'checkpoint/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()