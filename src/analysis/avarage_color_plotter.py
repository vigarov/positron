import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

csv_path = "../outs/new_color_checkers.csv"
df = pd.read_csv(csv_path)
df_clean = df.dropna()

unique_pixels = df_clean[['image_id', 'pixel_num']].drop_duplicates().values
unique_pixels = sorted(unique_pixels, key=lambda x: (x[0], x[1]))

output_dir = "../outs/color_plots"
os.makedirs(output_dir, exist_ok=True)

def normalize_rgb(r, g, b):
    return r/255.0, g/255.0, b/255.0

unique_image_ids = sorted(df_clean['image_id'].unique())

total_plots = 0

for image_id in unique_image_ids:
    image_pixels = [(id, pixel) for id, pixel in unique_pixels if id == image_id]
    n_pixels = len(image_pixels)
    
    n_cols = 6
    n_rows = (n_pixels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15*2.5))
    
    if n_rows == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    for idx, (_, pixel_num) in enumerate(image_pixels):
        pixel_data = df_clean[(df_clean['image_id'] == image_id) & 
                             (df_clean['pixel_num'] == pixel_num)]
        
        da_avg = [
            pixel_data['DA_R'].mean(),
            pixel_data['DA_G'].mean(),
            pixel_data['DA_B'].mean()
        ]
        
        n1_avg = [
            pixel_data['N1_R'].mean(),
            pixel_data['N1_G'].mean(),
            pixel_data['N1_B'].mean()
        ]
        
        n2_avg = [
            pixel_data['N2_R'].mean(),
            pixel_data['N2_G'].mean(),
            pixel_data['N2_B'].mean()
        ]
        
        da_norm = normalize_rgb(*da_avg)
        n1_norm = normalize_rgb(*n1_avg)
        n2_norm = normalize_rgb(*n2_avg)
        
        ax = axes[idx]
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add color squares/rectangles
        # Left square: DA
        da_square = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor=da_norm)
        ax.add_patch(da_square)
        ax.text(0.5, -0.1, f"DA: R:{da_avg[0]:.1f}, G:{da_avg[1]:.1f}, B:{da_avg[2]:.1f}", ha='center', fontsize=6)
        
        # Right top rectangle: N1
        n1_rect = patches.Rectangle((1, 0.5), 1, 0.5, linewidth=1, edgecolor='black', facecolor=n1_norm)
        ax.add_patch(n1_rect)
        ax.text(1.5, 1.05, f"N1: R:{n1_avg[0]:.1f}, G:{n1_avg[1]:.1f}, B:{n1_avg[2]:.1f}", ha='center', fontsize=6)
        
        # Right bottom rectangle: N2
        n2_rect = patches.Rectangle((1, 0), 1, 0.5, linewidth=1, edgecolor='black', facecolor=n2_norm)
        ax.add_patch(n2_rect)
        ax.text(1.5, -0.1, f"N2: R:{n2_avg[0]:.1f}, G:{n2_avg[1]:.1f}, B:{n2_avg[2]:.1f}", ha='center', fontsize=6)
        
        ax.set_title(f"Pixel: {int(pixel_num)}", fontsize=9)
    
    for idx in range(n_pixels, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.suptitle(f"Image ID: {int(image_id)}", fontsize=16)
    
    output_path = os.path.join(output_dir, f"image_{int(image_id)}_colors.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    total_plots += n_pixels

print(f"Generated {len(unique_image_ids)} images with a total of {total_plots} color plots in {output_dir}")
