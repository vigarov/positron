import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.color
import numpy as np
import torch
from skimage import color

def _visualize_one_sample(gt, x, pred):
    fig, ax = plt.subplots(figsize=(8, 6))

    gt_rgb = skimage.color.lab2rgb(gt)
    x_rgb = skimage.color.lab2rgb(x) 
    pred_rgb = skimage.color.lab2rgb(pred)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    gt_square = patches.Rectangle((0, 0), 1, 1, linewidth=1, 
                                edgecolor='black', facecolor=gt_rgb)
    ax.add_patch(gt_square)
    
    x_rect = patches.Rectangle((1, 0.5), 1, 0.5, linewidth=1,
                             edgecolor='black', facecolor=x_rgb)
    ax.add_patch(x_rect)
    
    pred_rect = patches.Rectangle((1, 0), 1, 0.5, linewidth=1,
                                edgecolor='black', facecolor=pred_rgb)
    ax.add_patch(pred_rect)

    ax.text(0.5, -0.1, "Ground Truth", ha='center')
    ax.text(1.5, 1.05, "Input", ha='center')
    ax.text(1.5, -0.1, "Prediction", ha='center')
    delta_e = np.sqrt(np.sum((pred - gt)**2))
    
    box_color = 'green' if delta_e < 2.3 else 'red'
    text = f'$\Delta E_{{ab}} = {delta_e:.2f}$'
    
    text_box = dict(boxstyle='round,pad=0.5', fc='white', 
                   ec=box_color, lw=2)
    ax.text(1, -0.2, text, ha='center', va='center',
            bbox=text_box)
    return fig, ax

def viz_sample(model, dataset, idx, color_space='LAB'):
    model.eval()
    sample = dataset[idx]
    x_val = sample['x']
    y_true_raw = sample['y']

    with torch.no_grad():
        y_pred_raw = model(x_val.unsqueeze(0).to(next(model.parameters()).device)).squeeze().cpu().numpy()

    if color_space == 'LAB':
        # y_true_raw and y_pred_raw are LAB values
        rgb_true = color.lab2rgb(y_true_raw.numpy().reshape(1, 1, 3)).squeeze()
        rgb_pred = color.lab2rgb(y_pred_raw.reshape(1, 1, 3)).squeeze()
    elif color_space == 'RGB':
        # y_true_raw and y_pred_raw are RGB values (0-1 range)
        rgb_true = y_true_raw.numpy()
        rgb_pred = y_pred_raw
        # Clip predictions to [0,1] just in case model outputs slightly outside
        rgb_pred = np.clip(rgb_pred, 0, 1)
    else:
        raise ValueError(f"Unsupported color_space: {color_space}")

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    axes[0].imshow([[rgb_true]])
    axes[0].set_title(f"True Color\n{color_space}: {np.round(y_true_raw.numpy(), 2)}")
    axes[0].axis('off')
    
    axes[1].imshow([[rgb_pred]])
    axes[1].set_title(f"Predicted Color\n{color_space}: {np.round(y_pred_raw, 2)}")
    axes[1].axis('off')
    
    fig.tight_layout()
    return fig, (y_true_raw, y_pred_raw)
