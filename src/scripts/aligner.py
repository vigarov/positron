import os
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
import argparse
from typing import List, Dict, Tuple, Optional

# Import SAM2 components
from sam2.build_sam2 import build_sam2_model
from sam2.predictor import SAM2Predictor


class ImageAligner:
    def __init__(self, sam2_checkpoint: str, device: str = "cuda"):
        """
        Initialize the ImageAligner with the SAM2 model.
        
        Args:
            sam2_checkpoint: Path to the SAM2 model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Using device: {self.device}")
        
        # Load SAM2 model
        self.model = build_sam2_model(sam2_checkpoint)
        self.model.to(self.device)
        self.predictor = SAM2Predictor(self.model)
    
    def detect_object(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect the most prominent object in an image using SAM2.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (mask, center_point)
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image in the predictor
        self.predictor.set_image(image)
        
        # Get automatic masks (without prompt)
        masks, scores, _ = self.predictor.predict()
        
        if len(masks) == 0:
            # If no masks found automatically, try with a point in the center
            h, w = image.shape[:2]
            center_point = np.array([[w//2, h//2]])
            masks, _, _ = self.predictor.predict(
                point_coords=center_point,
                point_labels=np.array([1]),  # 1 for foreground
            )
        
        # Get the mask with the highest score or the first one
        if len(masks) == 0:
            raise ValueError(f"No objects detected in {image_path}")
        
        best_mask = masks[0]
        
        # Calculate center of mass of the mask
        y_indices, x_indices = np.where(best_mask)
        if len(y_indices) == 0:
            # If mask is empty, use center of image
            h, w = image.shape[:2]
            center = np.array([w//2, h//2])
        else:
            center = np.array([int(np.mean(x_indices)), int(np.mean(y_indices))])
        
        return best_mask, center
    
    def calculate_shift(self, 
                       source_center: np.ndarray, 
                       target_center: np.ndarray) -> np.ndarray:
        """
        Calculate the shift needed to align the source center with the target center.
        
        Args:
            source_center: Center point of the object in the source image
            target_center: Center point of the object in the target image
            
        Returns:
            Shift as (dx, dy) to apply to the source image
        """
        return target_center - source_center
    
    def apply_shift(self, 
                   image_path: str, 
                   shift: np.ndarray, 
                   output_path: str):
        """
        Apply a shift to an image and save the result.
        
        Args:
            image_path: Path to the image to shift
            shift: (dx, dy) shift to apply
            output_path: Path to save the shifted image
        """
        # Load image
        image = cv2.imread(image_path)
        
        # Create translation matrix
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        
        # Apply shift
        h, w = image.shape[:2]
        shifted_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Save result
        cv2.imwrite(output_path, shifted_image)
    
    def align_images(self, 
                    ground_truth_path: str, 
                    source_paths: List[str], 
                    output_dir: str) -> Dict[str, np.ndarray]:
        """
        Align multiple source images to a ground truth image.
        
        Args:
            ground_truth_path: Path to ground truth image
            source_paths: List of paths to source images to align
            output_dir: Directory to save aligned images
            
        Returns:
            Dictionary mapping source path to the calculated shift
        """
        # Detect object in ground truth image
        gt_mask, gt_center = self.detect_object(ground_truth_path)
        
        # Process each source image
        shifts = {}
        for source_path in source_paths:
            try:
                # Detect object in source image
                source_mask, source_center = self.detect_object(source_path)
                
                # Calculate shift
                shift = self.calculate_shift(source_center, gt_center)
                shifts[source_path] = shift
                
                # Create output path
                source_name = os.path.basename(source_path)
                output_path = os.path.join(output_dir, f"aligned_{source_name}")
                
                # Apply shift and save
                self.apply_shift(source_path, shift, output_path)
                print(f"Aligned {source_path} with shift {shift}")
                
            except Exception as e:
                print(f"Error processing {source_path}: {e}")
        
        return shifts


def process_image_mapping(mapping_file: str, 
                         sam2_checkpoint: str,
                         output_dir: str,
                         device: str = "cuda"):
    """
    Process image mapping file and align all image sets.
    
    Args:
        mapping_file: Path to JSON file with image mappings
        sam2_checkpoint: Path to SAM2 model checkpoint
        output_dir: Directory to save aligned images
        device: Device to run on ('cuda' or 'cpu')
    """
    # Load mapping file
    with open(mapping_file, 'r') as f:
        mappings = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize aligner
    aligner = ImageAligner(sam2_checkpoint, device)
    
    # Process each mapping
    for map_id, image_paths in mappings.items():
        if len(image_paths) < 3:
            print(f"Skipping mapping {map_id}: not enough images")
            continue
        
        # Get ground truth and source paths
        ground_truth = image_paths[0]
        source_paths = image_paths[1:]
        
        # Create output subdirectory for this mapping
        map_output_dir = os.path.join(output_dir, f"set_{map_id}")
        os.makedirs(map_output_dir, exist_ok=True)
        
        print(f"\nProcessing mapping {map_id}:")
        print(f"  Ground truth: {ground_truth}")
        print(f"  Sources: {source_paths}")
        
        # Align images
        shifts = aligner.align_images(ground_truth, source_paths, map_output_dir)
        
        # Save shifts to file
        shifts_file = os.path.join(map_output_dir, "shifts.json")
        with open(shifts_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_shifts = {k: v.tolist() for k, v in shifts.items()}
            json.dump(serializable_shifts, f, indent=2)
        
        # Copy ground truth to output directory
        gt_output = os.path.join(map_output_dir, os.path.basename(ground_truth))
        if not os.path.exists(gt_output):
            import shutil
            shutil.copy(ground_truth, gt_output)


def main():
    parser = argparse.ArgumentParser(description="Align images using SAM2 object detection")
    parser.add_argument("--mapping", type=str, required=True, 
                        help="Path to JSON file with image mappings")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SAM2 model checkpoint")
    parser.add_argument("--output", type=str, default="aligned_images",
                        help="Directory to save aligned images")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    process_image_mapping(args.mapping, args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
