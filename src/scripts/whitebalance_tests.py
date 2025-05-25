#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, color, img_as_float
import os
import argparse
from matplotlib import gridspec
import warnings
import logging

# Import modules using proper package paths
from src.processing.white_balance import White_Balance
from src.utils import apply_channel_scaling

# Configure logging
logger = logging.getLogger("whitebalance_tests")
logger.setLevel(logging.INFO)
# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Create formatter
formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
# Add handler to logger
logger.addHandler(console_handler)

# Custom warning handler to track warning origins
def warning_handler(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"Warning in {filename}:{lineno} - {category.__name__}: {message}")
    return

# Set our custom warning handler
warnings.showwarning = warning_handler


def create_subplot(fig, position, image, title, axis_off=True):
    ax = plt.subplot(position)
    
    # Check image range before passing to imshow
    if np.issubdtype(image.dtype, np.floating):
        min_val = np.min(image)
        max_val = np.max(image)
        if min_val < 0 or max_val > 1:
            logger.warning(f"Image for '{title}' has out-of-range values: min={min_val}, max={max_val}")
    
    ax.imshow(image)
    ax.set_title(title)
    if axis_off:
        ax.axis('off')
    return ax

def save_image(output_dir, base_name, suffix, image):
    path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
    io.imsave(path, image)
    logger.info(f"Saved image to {path}")
    return path

def apply_and_compare_algorithms(img_path, output_dir=None, save_results=False):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:,:,:3]
    
    gray_world_wb = White_Balance(method='gray_world')
    white_patch_wb = White_Balance(method='white_patch')
    shades_of_gray_wb = White_Balance(method='shades_of_gray')
    retinex_wb = White_Balance(method='retinex')
    
    gray_world_result = gray_world_wb.apply(img)
    white_patch_result = white_patch_wb.apply(img)
    shades_of_gray_result = shades_of_gray_wb.apply(img)
    retinex_result = retinex_wb.apply(img)
    
    def check_result(name, result):
        if np.issubdtype(result.dtype, np.floating):
            min_val = np.min(result)
            max_val = np.max(result)
            if min_val < 0 or max_val > 1:
                logger.warning(f"Method '{name}' produced out-of-range values: min={min_val}, max={max_val}")
    
    check_result('gray_world', gray_world_result)
    check_result('white_patch', white_patch_result)
    check_result('shades_of_gray', shades_of_gray_result)
    check_result('retinex', retinex_result)
    
    has_opencv_extra = hasattr(cv2, 'xphoto')
    try:
        if has_opencv_extra:
            opencv_wb = White_Balance(method='opencv')
            opencv_result = opencv_wb.apply(img)
            check_result('opencv', opencv_result)
        else:
            logger.info("OpenCV xphoto module not available. Skipping SimpleWB method.")
            opencv_result = img.copy()
    except (AttributeError, cv2.error, RuntimeError) as e:
        logger.error(f"Error applying white balance: {e}")
        opencv_result = img.copy()
        has_opencv_extra = False
    
    results = {
        'original': img,
        'gray_world': gray_world_result,
        'white_patch': white_patch_result,
        'shades_of_gray': shades_of_gray_result,
        'retinex': retinex_result,
        'opencv': opencv_result
    }
    
    file_name = os.path.basename(img_path)
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3)
    
    create_subplot(fig, gs[0, 0], img, 'Original')
    create_subplot(fig, gs[0, 1], gray_world_result, 'Gray World')
    create_subplot(fig, gs[0, 2], white_patch_result, 'White Patch')
    create_subplot(fig, gs[1, 0], shades_of_gray_result, 'Shades of Gray (p=6)')
    
    retinex_title = 'Retinex (CLAHE)'
    create_subplot(fig, gs[1, 1], retinex_result, retinex_title)
    
    opencv_title = 'OpenCV SimpleWB' if has_opencv_extra else 'OpenCV SimpleWB (Not Available)'
    create_subplot(fig, gs[1, 2], opencv_result, opencv_title)
    
    plt.tight_layout()
    plt.suptitle(f"White Balance Comparisons - {file_name}", fontsize=16)
    plt.subplots_adjust(top=0.93)
    
    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(file_name)[0]
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        plt.savefig(comparison_path)
        logger.info(f"Saved comparison figure to {comparison_path}")
        
        for name, image in results.items():
            if name != 'opencv' or has_opencv_extra:
                save_image(output_dir, base_name, name, image)

    plt.show()
    
    return results

# Calculate metrics to quantitatively compare the different white balance algorithms
def calculate_color_metrics(results):
    logger.debug("Calculating color metrics")
    metrics = {}
    
    for name, img in results.items():
        avg_rgb = np.mean(img_as_float(img), axis=(0, 1))
        
        std_rgb = np.std(img_as_float(img), axis=(0, 1))
        
        img_hsv = color.rgb2hsv(img_as_float(img))
        avg_saturation = np.mean(img_hsv[:,:,1])
        
        avg_brightness = np.mean(img_as_float(img))
        
        metrics[name] = {
            'avg_rgb': avg_rgb,
            'std_rgb': std_rgb,
            'avg_saturation': avg_saturation,
            'avg_brightness': avg_brightness,
            'rgb_ratio': avg_rgb / np.mean(avg_rgb)
        }
        
        logger.debug(f"Metrics for {name}: avg_rgb={avg_rgb}, rgb_ratio={metrics[name]['rgb_ratio']}")
    
    return metrics

# Plot RGB ratios to visualize color balance
def plot_rgb_ratios(metrics, title, output_path=None):
    logger.debug(f"Plotting RGB ratios: {title}")
    methods = list(metrics.keys())
    rgb_ratios = [metrics[method]['rgb_ratio'] for method in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.2

    ax.bar(x - width, [ratio[0] for ratio in rgb_ratios], width, label='R')
    ax.bar(x, [ratio[1] for ratio in rgb_ratios], width, label='G')
    ax.bar(x + width, [ratio[2] for ratio in rgb_ratios], width, label='B')

    ax.set_ylabel('RGB Channel Ratio')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved RGB ratio plot to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare different white balance algorithms on images')
    parser.add_argument('images', nargs='+', help='Paths to input images')
    parser.add_argument('--output', '-o', default=None, help='Output directory for processed images')
    parser.add_argument('--save', '-s', action='store_true', help='Save processed images')
    parser.add_argument('--metrics', '-m', action='store_true', help='Display metrics charts')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info(f"OpenCV xphoto module available: {hasattr(cv2, 'xphoto')}")
    
    for img_path in args.images:
        if not os.path.exists(img_path):
            logger.warning(f"Image path {img_path} does not exist. Skipping.")
            continue
            
        logger.info(f"Processing image: {img_path}")
        results = apply_and_compare_algorithms(img_path, args.output, args.save)
        
        if args.metrics:
            metrics = calculate_color_metrics(results)
            plot_title = f'RGB Balance Comparison - {os.path.basename(img_path)}'
            metrics_output = None
            if args.save and args.output:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                metrics_output = os.path.join(args.output, f"{base_name}_metrics.png")
            plot_rgb_ratios(metrics, plot_title, metrics_output)

if __name__ == "__main__":
    main()