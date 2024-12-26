import cv2
import numpy as np
import os
from torchvision import transforms

def enhance_water_features(image):
    # Step 1: Contrast and brightness enhancement (preserve natural colors)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def adjust_gamma(image, gamma=1.2):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    enhanced_image = adjust_gamma(enhanced_image)

    # Step 2: Extract water-relevant channels
    hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)

    # Water-related features
    saturation = hsv[:, :, 1]  # High saturation for water
    blue_channel = lab[:, :, 2]  # Blue-ish tones in 'b' channel of Lab space

    # Normalize blue channel and saturation
    blue_channel_normalized = cv2.normalize(blue_channel, None, 0, 255, cv2.NORM_MINMAX)
    saturation_normalized = cv2.normalize(saturation, None, 0, 255, cv2.NORM_MINMAX)

    # Step 3: Edge detection (Laplacian)
    laplacian_edges = cv2.Laplacian(saturation, cv2.CV_64F)
    laplacian_edges = cv2.convertScaleAbs(laplacian_edges)

    # Step 4: Blend features while preserving original colors
    # Combine blue channel and saturation as a mask
    water_mask = cv2.addWeighted(blue_channel_normalized, 0.5, saturation_normalized, 0.5, 0)

    # Enhance original image using the water mask
    water_mask_colored = cv2.cvtColor(water_mask, cv2.COLOR_GRAY2BGR)
    enhanced_with_mask = cv2.addWeighted(enhanced_image, 0.8, water_mask_colored, 0.2, 0)

    # Add edge enhancement
    laplacian_colored = cv2.cvtColor(laplacian_edges, cv2.COLOR_GRAY2BGR)
    final_combined = cv2.addWeighted(enhanced_with_mask, 0.9, laplacian_colored, 0.1, 0)

    return final_combined

def preprocess_for_water(image_path):
    """
    Complete preprocessing pipeline for water segmentation
    
    Args:
        image_path: Path to input image
    Returns:
        processed_image: Image ready for UNet input
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    print(f"processing {image_path}")
    # Apply water-focused enhancement
    processed = enhance_water_features(image)
    
    return processed


if __name__ == "__main__":
    input_dir = 'training_dataset/image'  # Directory containing images to predict
    output_dir = 'training_dataset/preprocessed'  # Directory to save predictions
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir)]
    for image_path in image_files:
      processed = preprocess_for_water(os.path.join(input_dir, image_path))
      cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(image_path)[0]}.png"), processed)
    input_dir = 'testing_dataset/image'  # Directory containing images to predict
    output_dir = 'testing_dataset/preprocessed'  # Directory to save predictions
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir)]
    for image_path in image_files:
      processed = preprocess_for_water(os.path.join(input_dir, image_path))
      cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(image_path)[0]}.png"), processed)