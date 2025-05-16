#!/usr/bin/env python3
import os
import argparse
import tifffile
import numpy as np
from pathlib import Path


def crop_tiff(input_path, output_path=None, width_reduction_factor=0.07):
    img = tifffile.imread(input_path)
    
    original_height, original_width = img.shape[:2]
    new_width = int(original_width * (1 - width_reduction_factor))
    new_height = int(original_height * (new_width / original_width))
    
    width_crop_per_side = (original_width - new_width) // 2
    height_crop_per_side = (original_height - new_height) // 2
    
    cropped_img = img[
        height_crop_per_side:height_crop_per_side + new_height,
        width_crop_per_side:width_crop_per_side + new_width
    ]
    
    # If dimensions are odd, we might need to adjust by 1 pixel
    if original_width - new_width != 2 * width_crop_per_side:
        cropped_img = img[
            height_crop_per_side:height_crop_per_side + new_height,
            width_crop_per_side:width_crop_per_side + new_width + 1
        ]
    
    if original_height - new_height != 2 * height_crop_per_side:
        cropped_img = img[
            height_crop_per_side:height_crop_per_side + new_height + 1,
            width_crop_per_side:width_crop_per_side + new_width
        ]
    
    if output_path is None:
        output_path = input_path
    
    tifffile.imwrite(output_path, cropped_img)
    return output_path


def process_directory(input_dir, output_dir=None, width_reduction_factor=0.07):
    input_dir = Path(input_dir)
    
    if output_dir:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all TIFF files in the directory
    tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    
    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files to process")
    
    for tiff_file in tiff_files:
        if output_dir:
            output_path = output_dir / tiff_file.name
        else:
            output_path = tiff_file
        
        try:
            saved_path = crop_tiff(tiff_file, output_path, width_reduction_factor)
            print(f"Processed {tiff_file} -> {saved_path}")
        except Exception as e:
            print(f"Error processing {tiff_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Crop TIFF images by reducing width while maintaining aspect ratio')
    parser.add_argument('input_path', help='Path to TIFF file or directory containing TIFF files')
    parser.add_argument('--output', '-o', help='Output directory for cropped images (optional)')
    parser.add_argument('--reduction', '-r', type=float, default=0.07, 
                       help='Width reduction factor (default: 0.07 = 7%)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    width_reduction = args.reduction
    
    if width_reduction <= 0 or width_reduction >= 1:
        print(f"Invalid reduction factor: {width_reduction}. Please provide a value between 0 and 1.")
        return
    
    if input_path.is_file() and input_path.suffix.lower() in ['.tif', '.tiff']:
        output_path = Path(args.output) / input_path.name if args.output else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        saved_path = crop_tiff(input_path, output_path, width_reduction)
        print(f"Processed {input_path} -> {saved_path}")
    
    elif input_path.is_dir():
        process_directory(input_path, args.output, width_reduction)
    
    else:
        print(f"Invalid input: {input_path}. Please provide a TIFF file or directory containing TIFF files.")


if __name__ == "__main__":
    main()
