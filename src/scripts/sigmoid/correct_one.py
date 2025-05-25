import argparse
import json
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import sys
# Assuming the script is in positron/scripts/sigmoid/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.processing.sigmoid_correct import SigmoidCorrect

def load_and_normalize_tiff(image_path):
    """Loads a TIFF image and normalizes it to a float32 array in [0, 1]."""
    try:
        img = tifffile.imread(image_path)
    except FileNotFoundError:
        # Running the script from a different dir might cause this, trying to load from project root
        print(f"Error: Image file not found at {image_path}")
        if not os.path.isabs(image_path) and image_path.startswith("data"):
             alt_path = os.path.join(project_root, image_path.replace('\\', os.sep)) # Handle windows paths in json
             print(f"Attempting to load from: {alt_path}")
             try:
                 img = tifffile.imread(alt_path)
             except FileNotFoundError:
                 print(f"Error: Image file still not found at {alt_path}")
                 raise
        else:
            raise    
    if np.min(img) < -1e-5 or np.max(img) > 1.0 + 1e-5:
        print(f"Warning: Input image values are outside the expected [0,1] range. Min: {np.min(img)}, Max: {np.max(img)} ; 99th percentile: {np.percentile(img,99)}")
        img = (img - np.min(img))/(np.max(img) - np.min(img))
    if img.ndim == 2 or img.shape[-1] >= 4:
        print(f"Error: Input image is not a 3-channel RGB image")
        exit(-1)
    return img


def main():
    parser = argparse.ArgumentParser(description="Apply Sigmoid Correction to an image and plot results with several parameters")
    parser.add_argument("--mappings", type=str, required=True, help="Path to the image mapping JSON file.")
    parser.add_argument("--index", type=int, default=0, help="Index of the image set in the mapping file (default: 0).")
    args = parser.parse_args()

    # Load image paths from mapping file
    try:
        with open(args.mappings, 'r') as f:
            image_mapping = json.load(f)
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {args.mappings}")
        return
    
    image_set_key = str(args.index)
    if image_set_key not in image_mapping:
        print(f"Error: Image index {args.index} (key: '{image_set_key}') not found in mapping file.")
        print(f"Available keys: {list(image_mapping.keys())}")
        return

    paths = image_mapping[image_set_key]
    da_tiff_path = paths[0]
    source1_path = paths[1]
    
    print(f"Loading DA TIFF (Ground Truth) from: {da_tiff_path}")
    print(f"Loading Source 1 (Base Image) from: {source1_path}")

    base_image = load_and_normalize_tiff(source1_path)
    gt_image = load_and_normalize_tiff(da_tiff_path)

    # Define hyperparameter combinations
    wb_methods_to_test = ['gray_world', 'scale_to_green_mean', 'percentile_reference_balance']
    correction_modes_to_test = ['center_mean', 'stretch_percentiles']

    results_for_plotting = [] 

    default_params = {
        'map_to_z': 4.0,
        'lumi_factor': 0.0, 
        'center_loc': 0.0, 
        'center_upper_percentile': 95.0,
        'stretch_lower_percentile': 1.0,
        'stretch_upper_percentile': 95.0,
        'percentile_ref_target_point': 'white',
        'percentile_ref_lower': 1.0,
        'percentile_ref_upper': 95.0,
        'rotate_image': True
    }

    for wb_method in wb_methods_to_test:
        for corr_mode in correction_modes_to_test:
            config = {'wb_method': wb_method}
            title_params = {} # For constructing the plot title

            # Set general SigmoidCorrect params from our defaults
            config['map_to_z'] = default_params['map_to_z']
            config['rotate_image'] = default_params['rotate_image']
            title_params['map_z'] = config['map_to_z'] # Shorten for title

            if corr_mode == 'center_mean':
                config['center_loc'] = default_params['center_loc']
                config['center_upper_percentile'] = default_params['center_upper_percentile']
                title_params['ctr_loc'] = config['center_loc']
                title_params['ctr_P%'] = config['center_upper_percentile']
            elif corr_mode == 'stretch_percentiles':
                config['lumi_factor'] = default_params['lumi_factor']
                config['stretch_lower_percentile'] = default_params['stretch_lower_percentile']
                config['stretch_upper_percentile'] = default_params['stretch_upper_percentile']
                title_params['lumi'] = config['lumi_factor']
                title_params['str_L%'] = config['stretch_lower_percentile']
                title_params['str_U%'] = config['stretch_upper_percentile']
            
            # White balance specific params for percentile_reference_balance
            if wb_method == 'percentile_reference_balance':
                config['percentile_ref_target_point'] = default_params['percentile_ref_target_point']
                config['percentile_ref_lower'] = default_params['percentile_ref_lower']
                config['percentile_ref_upper'] = default_params['percentile_ref_upper']
                # Add to title if desired, e.g.
                title_params['WBtgt'] = config['percentile_ref_target_point'][0].upper() # W or B

            print(f"Processing: WB='{wb_method}', Mode='{corr_mode}', Config={config}")
            config['correction_mode'] = corr_mode
            sigmoid_processor = SigmoidCorrect(config)
            
            processed_image = sigmoid_processor.apply(base_image.copy()) 
            results_for_plotting.append({
                'image': processed_image, 
                'corr_mode': corr_mode, 
                'wb_method': wb_method, 
                'params': title_params
            })

    # Plotting: 2 (base, GT) + 6 (results) = 8 plots. Arrange in 2x4 grid.
    num_plots = 2 + len(results_for_plotting)
    rows, cols = 2, 4 
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5)) # Adjusted figsize
    axes = axes.ravel() 

    axes[0].imshow(base_image)
    axes[0].set_title("Base Image (Source 1)")
    axes[0].axis('off')

    axes[1].imshow(gt_image)
    axes[1].set_title("Ground Truth (DA TIFF)")
    axes[1].axis('off')

    for i, result_data in enumerate(results_for_plotting):
        ax_idx = i + 2
        if ax_idx >= len(axes):
            print(f"Warning: Not enough axes ({len(axes)}) for plot {ax_idx}. Skipping remaining plots.")
            break
        
        ax = axes[ax_idx]
        ax.imshow(result_data['image'])
        
        params_str_parts = []
        for k,v in result_data['params'].items():
             # Format floats to 1 decimal place if they are floats, else use original value
            val_str = f"{v:.1f}" if isinstance(v, float) and not v.is_integer() else str(v)
            params_str_parts.append(f"{k}:{val_str}")
        params_summary = ", ".join(params_str_parts)
        
        title = (f"Mode: {result_data['corr_mode'].replace('_', ' ').title()}\n"
                 f"WB: {result_data['wb_method'].replace('_', ' ').title()}\n"
                 f"{params_summary}")
        ax.set_title(title, fontsize=7)
        ax.axis('off')
    
    for j in range(num_plots, rows * cols):
        if j < len(axes):
            fig.delaxes(axes[j])

    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)
    plt.show()

if __name__ == '__main__':
    main() 