import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from model import UNet, WaterSegmentationUNet

def predict_images(model, input_dir, output_dir, device):
    """
    Predict masks for all images in input_dir and save them to output_dir
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir)]
    
    # Process each image
    model.eval()
    with torch.no_grad():
        for img_name in tqdm(image_files, desc="Processing images"):
            # Load and preprocess image
            img_path = os.path.join(input_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            
            # Get original size for later resizing back
            original_size = image.size
            
            # Apply transforms
            image = transform(image).unsqueeze(0)  # Add batch dimension
            image = image.to(device)
            
            # Predict
            output = model(image)
            
            # Post-process prediction
            pred_mask = output.squeeze().cpu()
            pred_mask = (pred_mask > 0.5).float()
            
            # Convert to PIL Image and resize back to original size
            pred_mask = transforms.ToPILImage()(pred_mask)
    
            pred_mask = pred_mask.resize(original_size, Image.NEAREST)
            
            # Convert to binary image (0 or 255)
            pred_mask = np.array(pred_mask)
            # pred_mask = (pred_mask > 0.5) * 255
            # pred_mask = pred_mask.astype(np.uint8)
            
            # Save prediction
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.png")
            Image.fromarray(pred_mask).save(output_path)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    checkpoint_path = 'checkpoint/best_model.pth'  # Path to your checkpoint
    # checkpoint_path = 'checkpoint/model_epoch_21.pth'
    input_dir = 'testing_dataset/preprocessed'  # Directory containing images to predict
    output_dir = 'testing_dataset/output'  # Directory to save predictions
    
    # Initialize model
    model = WaterSegmentationUNet().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    if 'val_iou' in checkpoint:
        print(f"Validation IoU of loaded model: {checkpoint['val_iou']:.4f}")
    
    # Run prediction
    predict_images(model, input_dir, output_dir, device)
    print(f"Predictions saved to {output_dir}/")

if __name__ == '__main__':
    main()