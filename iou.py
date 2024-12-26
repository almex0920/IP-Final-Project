import os
import cv2
import numpy as np
from glob import glob

def calculate_iou(true_mask, pred_mask):
    """
    Calculate Intersection over Union (IoU) for a pair of masks.

    :param true_mask: Ground truth mask (binary).
    :param pred_mask: Predicted mask (binary).
    :return: IoU score.
    """
    intersection = np.logical_and(true_mask, pred_mask)  # Intersection of true and predicted masks
    union = np.logical_or(true_mask, pred_mask)  # Union of true and predicted masks
    iou = np.sum(intersection) / np.sum(union)  # IoU calculation
    return iou

def calculate_mean_iou(true_mask_dir, pred_mask_dir):
    """
    Calculate the mean IoU across multiple mask images.

    :param true_mask_dir: Directory containing ground truth masks.
    :param pred_mask_dir: Directory containing predicted masks.
    :return: Mean IoU.
    """
    true_masks = sorted(glob(os.path.join(true_mask_dir, '*.png')))  # Load true masks
    pred_masks = sorted(glob(os.path.join(pred_mask_dir, '*.png')))  # Load predicted masks
    
    assert len(true_masks) == len(pred_masks), "The number of true and predicted masks should be the same."

    iou_scores = []

    for true_mask_path, pred_mask_path in zip(true_masks, pred_masks):
        # Read mask images
        true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale image
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale image

        # Binarize masks (ensure binary masks, 0 is background, 1 is foreground)
        true_mask = (true_mask > 127).astype(np.uint8)
        pred_mask = (pred_mask > 127).astype(np.uint8)

        # Calculate IoU for each mask pair
        iou = calculate_iou(true_mask, pred_mask)
        iou_scores.append(iou)
        print(f"Image pair: {true_mask_path} vs {pred_mask_path}, IoU: {iou:.4f}")

    # Calculate the mean IoU
    mean_iou = np.mean(iou_scores)
    return mean_iou

if __name__ == "__main__":
    # Define paths to the ground truth and predicted mask directories
    true_mask_dir = 'testing_dataset/mask/'  # Replace with your ground truth mask directory
    pred_mask_dir = 'testing_dataset/output/'  # Replace with your predicted mask directory

    # Calculate the mean IoU across all mask images
    mean_iou = calculate_mean_iou(true_mask_dir, pred_mask_dir)
    print(f"Mean IoU across all images: {mean_iou:.4f}")