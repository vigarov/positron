import matplotlib.pyplot as plt
import skimage.color
import skimage.transform
import numpy as np
import torch
import tifffile
import skimage.io as io
from skimage.transform import resize

def show_image_transform(model, source_img_path, gt_img_path, display_width=800, color_space='LAB'):
    """
    Loads a source image, transforms it using the model, and displays it alongside the ground truth.
    
    Parameters:
        source_img_path: Source image path
        gt_img_path: Ground truth image path
        display_width: Width for displaying images
        color_space: 'LAB' or 'RGB', the color space the model operates in.
    Returns:
        matplotlib.figure.Figure: The figure object containing the plots.
    """
    model.eval()
    model_device = next(model.parameters()).device

    src_rgb_raw = io.imread(source_img_path)
    gt_rgb_raw = io.imread(gt_img_path)

    if src_rgb_raw.shape[-1] == 4:
        src_rgb_raw = src_rgb_raw[..., :3]
    if gt_rgb_raw.shape[-1] == 4:
        gt_rgb_raw = gt_rgb_raw[..., :3]

    def _resize_img(img, width):
        aspect_ratio = img.shape[0] / img.shape[1]
        new_height = int(width * aspect_ratio)
        return resize(img, (new_height, width), anti_aliasing=True)

    src_rgb_resized = _resize_img(src_rgb_raw, display_width)
    gt_rgb_resized = _resize_img(gt_rgb_raw, display_width)
    
    src_rgb_norm = np.clip(src_rgb_resized, 0, 1) # Already 0-1 from resize, but good practice

    if color_space == 'LAB':
        input_data_for_model = skimage.color.rgb2lab(src_rgb_norm)
    elif color_space == 'RGB':
        input_data_for_model = src_rgb_norm # Model expects RGB 0-1
    else:
        raise ValueError(f"Unsupported color_space: {color_space}")

    input_flat = input_data_for_model.reshape(-1, 3)
    input_tensor = torch.tensor(input_flat, dtype=torch.float32).to(model_device)

    with torch.no_grad():
        pred_flat = model(input_tensor).cpu().numpy()
    
    pred_img_transformed = pred_flat.reshape(input_data_for_model.shape)

    if color_space == 'LAB':
        pred_rgb_img = skimage.color.lab2rgb(pred_img_transformed)
    elif color_space == 'RGB':
        # pred_img_transformed is already RGB (0-1 range from model)
        pred_rgb_img = pred_img_transformed
    
    # Clip final RGB images to [0, 1] range
    pred_rgb_img = np.clip(pred_rgb_img, 0, 1)
    gt_rgb_clipped = np.clip(gt_rgb_resized,0,1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(src_rgb_norm)
    axes[0].set_title("Original Source Image (Input to Model Pre-processing)")
    axes[0].axis('off')

    axes[1].imshow(pred_rgb_img)
    axes[1].set_title(f"Model Transformed Output (from {color_space})")
    axes[1].axis('off')

    axes[2].imshow(gt_rgb_clipped)
    axes[2].set_title("Ground Truth Image")
    axes[2].axis('off')

    fig.tight_layout()
    return fig
