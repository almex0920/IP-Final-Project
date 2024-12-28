import cv2
import numpy as np
import os
from torchvision import transforms
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def enhance_water_features(image, debug=False):
    """
    Advanced color-preserving preprocessing pipeline for water surface segmentation.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image opened with cv2.imread()
    debug : bool, optional
        If True, will display intermediate processing steps
    
    Returns:
    --------
    tuple
        (enhanced_image, enhanced_mask)
        enhanced_image: Color image with enhanced features
        enhanced_mask: Grayscale segmentation-ready mask
    """
    # Ensure input is a valid image
    if image is None or image.size == 0:
        raise ValueError("Invalid input image")
    b,g,r = cv2.split(image)
    b_avg = cv2.mean(b)[0]
    g_avg = cv2.mean(g)[0]
    r_avg = cv2.mean(r)[0]
    avg = (b_avg + g_avg + r_avg ) / 3.0
    b_k = avg/b_avg
    g_k = avg/g_avg
    r_k = avg/r_avg
    b = (b*b_k).clip(0, 255)
    g = (g*g_k).clip(0, 255)
    r = (r*r_k).clip(0, 255)
    image = cv2.merge([b, g, r]).astype(np.uint8)
    # Convert BGR to RGB (cv2 uses BGR by default)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to different color spaces
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Separate channels
    h, s, v = cv2.split(img_hsv)
    l, a, b = cv2.split(img_lab)

    # Debug: original image
    if debug:
        plt.figure(figsize=(15,10))
        plt.subplot(3,3,1)
        plt.title('Original Image')
        plt.imshow(img_rgb)
        plt.axis('off')
    
    # 1. Color-Aware Contrast Enhancement
    # CLAHE on Value and Lightness channels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_enhanced = clahe.apply(v)
    l_enhanced = clahe.apply(l)
    
    # Debug: CLAHE enhanced channels
    if debug:
        plt.subplot(3,3,2)
        plt.title('CLAHE V Channel')
        plt.imshow(v_enhanced, cmap='gray')
        plt.axis('off')
        
        plt.subplot(3,3,3)
        plt.title('CLAHE L Channel')
        plt.imshow(l_enhanced, cmap='gray')
        plt.axis('off')
    
    # 2. Color-Based Water Surface Detection
    # Use color differences and saturation to highlight water
    water_mask = np.zeros_like(s)
    
    # Adaptive thresholding on saturation and color channels
    _, sat_thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b, _, _ = cv2.split(image)
    _, blue_thresh = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine thresholds
    water_mask = cv2.bitwise_and(sat_thresh, blue_thresh)
    
    # Debug: color-based masks
    if debug:
        plt.subplot(3,3,4)
        plt.title('Saturation Threshold')
        plt.imshow(sat_thresh, cmap='gray')
        plt.axis('off')
        
        plt.subplot(3,3,5)
        plt.title('Blue Channel Threshold')
        plt.imshow(blue_thresh, cmap='gray')
        plt.axis('off')
        
        plt.subplot(3,3,6)
        plt.title('Combined Water Mask')
        plt.imshow(water_mask, cmap='gray')
        plt.axis('off')
    
    # 3. Gradient and Edge Enhancement
    # Sobel operators on enhanced lightness channel
    sobelx = cv2.Sobel(l_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(l_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 4. Morphological Operations
    # Enhance water surface features
    kernel = np.ones((3,3), np.uint8)
    water_mask_processed = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
    water_mask_processed = cv2.morphologyEx(water_mask_processed, cv2.MORPH_OPEN, kernel)
    
    # 5. Combine Masks
    enhanced_mask = cv2.addWeighted(
        water_mask_processed, 0.6,
        gradient_magnitude, 0.4,
        0
    )
    
    # Normalize mask
    enhanced_mask = cv2.normalize(
        enhanced_mask, 
        None, 
        0, 255, 
        cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )
    
    # 6. Color-Preserved Enhancement
    # Reconstruct enhanced color image
    img_hsv_enhanced = cv2.merge([h, s, v_enhanced])
    img_rgb_enhanced = cv2.cvtColor(img_hsv_enhanced, cv2.COLOR_HSV2RGB)


    
    # Debug: final results
    if debug:
        plt.subplot(3,3,7)
        plt.title('Enhanced Mask')
        plt.imshow(enhanced_mask)
        plt.axis('off')
        
        plt.subplot(3,3,8)
        plt.title('Enhanced Color Image')
        plt.imshow(img_rgb_enhanced)
        plt.axis('off')
        
        plt.tight_layout()
       

    _, final_mask = cv2.threshold(enhanced_mask, 127, 255, cv2.THRESH_BINARY)
    
    # 2. Apply mask to the enhanced image
    # Option A: Direct masking for water region emphasis
    masked_enhancement = cv2.bitwise_and(img_rgb_enhanced, img_rgb_enhanced, mask=final_mask)
    
    # Option B: Blend the masked region with original for subtle enhancement
    alpha = 0.7  # Adjust this value to control enhancement strength
    mask_3ch = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB) / 255.0
    blended_result = (img_rgb_enhanced * alpha * mask_3ch + 
                     img_rgb * (1 - alpha * mask_3ch))
    blended_result = np.uint8(blended_result)
    
    # 3. Feature Enhancement in masked regions
    # Enhance blue channel in water regions
    b, g, r = cv2.split(img_rgb)
    b_enhanced = cv2.addWeighted(
        b, 1.2,  # Increase blue channel intensity
        np.zeros_like(b), 0,
        5  # Add slight brightness
    )
    
    # Only apply enhancement in masked regions
    b_final = np.where(final_mask == 255, b_enhanced, b)
    
    # Reconstruct the image with enhanced water features
    result = cv2.merge([b_final, g, r])
    
    # Debug visualization
    if debug:
        plt.subplot(3,3,7)
        plt.title('Enhanced Mask')
        plt.imshow(enhanced_mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(3,3,8)
        plt.title('Masked Enhancement')
        plt.imshow(cv2.cvtColor(masked_enhancement, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(3,3,9)
        plt.title('Final Result')
        plt.imshow(result)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

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
    predict_dir = 'testing_dataset/output'
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir)]
    for image_path in image_files:
      processed = preprocess_for_water(os.path.join(input_dir, image_path))
      cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(image_path)[0]}.png"), processed)