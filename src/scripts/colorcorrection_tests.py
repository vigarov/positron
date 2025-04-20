#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, img_as_float
import os
import argparse
from matplotlib import gridspec
import warnings
import logging
from src.processing.color_correct import Color_Correction

logger = logging.getLogger("colorcorrection_tests")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def warning_handler(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"Warning in {filename}:{lineno} - {category.__name__}: {message}")
    return
warnings.showwarning = warning_handler

def create_subplot(fig, position, image, title, axis_off=True):
    ax = plt.subplot(position)
    
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

def save_image(output_dir, base_name, method_name, img):
    output_path = os.path.join(output_dir, f"{base_name}_{method_name}.png")
    if np.issubdtype(img.dtype, np.floating):
        # Ensure float images are in 0-1 range before saving
        img = np.clip(img, 0, 1)
    io.imsave(output_path, img)
    logger.info(f"Saved {method_name} result to {output_path}")

def apply_and_compare_algorithms(img_path, output_dir=None, save_results=False):
    """
    Applies all color correction algorithms to an image and displays the results.
    
    Parameters:
    - img_path: Path to the input image
    - output_dir: Directory to save processed images (if save_results is True)
    - save_results: Whether to save the processed images
    """
    # Read the image (in RGB format)
    img = io.imread(img_path)
    if img.shape[2] == 4:  # If image has alpha channel, remove it
        img = img[:,:,:3]
    
    # Apply color correction algorithms using the Color_Correction class
    hist_eq = Color_Correction(method='histogram_equalization')
    gamma_corr = Color_Correction(method='gamma_correction').set_params(gamma=0.8)
    
    hist_eq_result = hist_eq.apply(img)
    gamma_corr_result = gamma_corr.apply(img)
    
    # Check the results for out-of-range values
    def check_result(name, result):
        if np.issubdtype(result.dtype, np.floating):
            min_val = np.min(result)
            max_val = np.max(result)
            if min_val < 0 or max_val > 1:
                logger.warning(f"Method '{name}' produced out-of-range values: min={min_val}, max={max_val}")
    
    check_result('histogram_equalization', hist_eq_result)
    check_result('gamma_correction', gamma_corr_result)
    
    # Try to use OpenCV's tonemap if available
    has_tonemap = hasattr(cv2, 'createTonemapDrago')
    try:
        if has_tonemap:
            tonemap_cc = Color_Correction(method='opencv_tonemap')
            tonemap_result = tonemap_cc.apply(img)
            check_result('opencv_tonemap', tonemap_result)
        else:
            logger.info("OpenCV tonemap module not available. Skipping tonemap method.")
            tonemap_result = img.copy()
    except (AttributeError, cv2.error, RuntimeError) as e:
        logger.error(f"Error applying tonemap: {e}")
        tonemap_result = img.copy()
        has_tonemap = False
    
    # Create results dictionary for display and return
    results = {
        'original': img,
        'histogram_eq': hist_eq_result,
        'gamma_corr': gamma_corr_result,
        'tonemap': tonemap_result
    }
    
    # Display the results
    file_name = os.path.basename(img_path)
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3)
    
    # Create subplots with helper function
    create_subplot(fig, gs[0, 0], img, 'Original')
    create_subplot(fig, gs[0, 1], hist_eq_result, 'Histogram Equalization')
    create_subplot(fig, gs[1, 0], gamma_corr_result, f'Gamma Correction (γ={gamma_corr.params["gamma_correction"]["gamma"]})')
    
    tonemap_title = 'OpenCV Tonemap' if has_tonemap else 'OpenCV Tonemap (Not Available)'
    create_subplot(fig, gs[2, 0], tonemap_result, tonemap_title)
    
    plt.tight_layout()
    plt.suptitle(f"Color Correction Comparisons - {file_name}", fontsize=16)
    plt.subplots_adjust(top=0.93)
    
    # Save the comparison figure if requested
    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(file_name)[0]
        comparison_path = os.path.join(output_dir, f"{base_name}_cc_comparison.png")
        plt.savefig(comparison_path)
        logger.info(f"Saved comparison figure to {comparison_path}")
        
        # Save individual results
        for name, image in results.items():
            if name != 'tonemap' or has_tonemap:
                save_image(output_dir, base_name, name, image)

    plt.show()
    
    return results

def calculate_image_metrics(results):
    """
    Calculate metrics to quantitatively compare the different color correction algorithms.
    """
    logger.debug("Calculating image metrics")
    metrics = {}
    
    for name, img in results.items():
        # Convert to float for calculations
        img_float = img_as_float(img)
        
        # Calculate contrast (standard deviation of luminance)
        luminance = 0.299 * img_float[:,:,0] + 0.587 * img_float[:,:,1] + 0.114 * img_float[:,:,2]
        contrast = np.std(luminance)
        
        # Calculate average RGB values and saturation
        avg_rgb = np.mean(img_float, axis=(0, 1))
        
        # Calculate color saturation (max - min across channels)
        max_val = np.max(img_float, axis=2)
        min_val = np.min(img_float, axis=2)
        saturation = np.mean(max_val - min_val)
        
        # Store metrics
        metrics[name] = {
            'contrast': contrast,
            'avg_rgb': avg_rgb,
            'saturation': saturation,
            'brightness': np.mean(luminance)
        }
        
        logger.debug(f"Metrics for {name}: contrast={contrast:.4f}, saturation={saturation:.4f}, brightness={metrics[name]['brightness']:.4f}")
    
    return metrics

def plot_metrics(metrics, title, output_path=None):
    """
    Plot metrics to visualize color correction effects.
    """
    logger.debug(f"Plotting metrics: {title}")
    methods = list(metrics.keys())
    
    # Extract the metrics to plot
    contrasts = [metrics[method]['contrast'] for method in methods]
    saturations = [metrics[method]['saturation'] for method in methods]
    brightnesses = [metrics[method]['brightness'] for method in methods]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot contrast
    ax1.bar(methods, contrasts, color='skyblue')
    ax1.set_ylabel('Contrast')
    ax1.set_title('Contrast Comparison')
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    
    # Plot saturation
    ax2.bar(methods, saturations, color='salmon')
    ax2.set_ylabel('Saturation')
    ax2.set_title('Saturation Comparison')
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    
    # Plot brightness
    ax3.bar(methods, brightnesses, color='lightgreen')
    ax3.set_ylabel('Brightness')
    ax3.set_title('Brightness Comparison')
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.93)
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved metrics plot to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare different color correction algorithms on images')
    parser.add_argument('images', nargs='+', help='Paths to input images')
    parser.add_argument('--output', '-o', default=None, help='Output directory for processed images')
    parser.add_argument('--save', '-s', action='store_true', help='Save processed images')
    parser.add_argument('--metrics', '-m', action='store_true', help='Display metrics charts')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging level based on args
    if args.debug:
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info(f"OpenCV tonemap available: {hasattr(cv2, 'createTonemapDrago')}")
    
    for img_path in args.images:
        if not os.path.exists(img_path):
            logger.warning(f"Image path {img_path} does not exist. Skipping.")
            continue
            
        logger.info(f"Processing image: {img_path}")
        results = apply_and_compare_algorithms(img_path, args.output, args.save)
        
        if args.metrics:
            metrics = calculate_image_metrics(results)
            plot_title = f'Color Correction Metrics - {os.path.basename(img_path)}'
            metrics_output = None
            if args.save and args.output:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                metrics_output = os.path.join(args.output, f"{base_name}_cc_metrics.png")
            plot_metrics(metrics, plot_title, metrics_output)

if __name__ == "__main__":
    main() 