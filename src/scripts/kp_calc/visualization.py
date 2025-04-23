import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def display_pixel_stats(app, pixel_data):
    """Display histograms for each image source"""
    # Clear previous plots
    app.fig.clear()
    
    # Get the number of sources
    num_sources = len(pixel_data)
    
    # Create subplots for each source
    axs = app.fig.subplots(num_sources, 1, sharex=True)
    if num_sources == 1:  # Handle case with single subplot
        axs = [axs]
    
    # Increase figure size
    app.fig.set_size_inches(15, 8)  # More height for vertical layout
    
    # Define the sources
    sources = list(pixel_data.keys())
    
    # Define colors for RGB channels with improved visibility
    channel_colors = ['r', 'g', 'b']
    channel_names = ['Red', 'Green', 'Blue']
    
    # Plot histograms for each source
    for i, source in enumerate(sources):
        ax = axs[i]
        ax.set_title(f"{source} Histogram", fontsize=12, fontweight='bold')
        
        # Get image data
        img_data = pixel_data[source]
        
        # Use more bins for higher precision
        bins = 256  # One bin for each possible pixel value (0-255)
        
        # Plot histogram for each RGB channel
        for channel_idx, (color, name) in enumerate(zip(channel_colors, channel_names)):
            channel_data = img_data[:, :, channel_idx].flatten()
            
            # Create histogram with 256 bins for full precision
            hist, bin_edges = np.histogram(channel_data, bins=bins, range=(0, 255))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Use step plot for more precise visualization
            ax.step(bin_centers, hist, color=color, alpha=0.7, 
                   label=f"{name} Channel", linewidth=1.5, where='mid')
            
            # Add a light fill under the curve for better visibility
            ax.fill_between(bin_centers, hist, alpha=0.1, color=color, step='mid')
        
        # Add grid and legend with improved styling
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=10, loc='upper right', framealpha=0.7)
        
        # Only add x-label to the bottom subplot
        if i == num_sources - 1:
            ax.set_xlabel("Pixel Value", fontsize=10)
        
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_xlim(0, 255)
        
        # Add minor grid lines for more precise reading
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.1, linestyle=':')
        
        # Format y-axis to use scientific notation for large values
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        
    # Add better spacing between subplots
    app.fig.tight_layout(pad=2.0)
    app.canvas_fig.draw()

def create_waveform(image_data, ax, title):
    """Generate a waveform visualization similar to Darktable's"""
    if image_data is None:
        return
    
    # Get the image dimensions
    height, width = image_data.shape[:2]
    
    # Create empty waveform images for each channel (higher resolution for smoother gradients)
    waveform_height = 512  # Doubled from 256 for higher vertical resolution
    waveform_r = np.zeros((waveform_height, width), dtype=np.float32)
    waveform_g = np.zeros((waveform_height, width), dtype=np.float32)
    waveform_b = np.zeros((waveform_height, width), dtype=np.float32)
    
    # For each column in the image
    for x in range(width):
        # Get the column of pixels
        column = image_data[:, x]
        
        # For each pixel in the column, increment the waveform count at the corresponding intensity
        for y in range(height):
            r, g, b = column[y]
            # Scale to waveform height 
            r_idx = min(int(r / 255 * (waveform_height-1)), waveform_height-1)
            g_idx = min(int(g / 255 * (waveform_height-1)), waveform_height-1) 
            b_idx = min(int(b / 255 * (waveform_height-1)), waveform_height-1)
            
            waveform_r[waveform_height-1-r_idx, x] += 1
            waveform_g[waveform_height-1-g_idx, x] += 1
            waveform_b[waveform_height-1-b_idx, x] += 1
    
    # Apply Gaussian blur to smooth the waveforms slightly (reduces jagged edges)
    sigma = 1.0  # Adjust this value for more/less smoothing
    waveform_r = gaussian_filter(waveform_r, sigma=sigma)
    waveform_g = gaussian_filter(waveform_g, sigma=sigma)
    waveform_b = gaussian_filter(waveform_b, sigma=sigma)
    
    # Find the range of values present in the data
    # First identify which rows in each channel have any data
    r_active = np.any(waveform_r > 0.01, axis=1)  # Slightly increased threshold to ignore noise
    g_active = np.any(waveform_g > 0.01, axis=1)
    b_active = np.any(waveform_b > 0.01, axis=1)
    
    # Combine to find the overall active range
    active_rows = r_active | g_active | b_active
    
    if np.any(active_rows):
        # Find the first and last active row to determine the range
        active_indices = np.where(active_rows)[0]
        min_row = max(0, active_indices[0] - 20)  # Add a larger margin
        max_row = min(waveform_height-1, active_indices[-1] + 20)
    else:
        # Default full range if no data is found
        min_row = 0
        max_row = waveform_height-1
    
    # Apply logarithmic scaling to enhance visibility of lower values (Darktable-style)
    # This makes even sparse data visible while keeping the display clean
    log_factor = 2.0  # Increased from 1.0 for better visibility of low values
    
    if np.max(waveform_r) > 0:
        waveform_r = np.log1p(waveform_r * log_factor) / np.log1p(np.max(waveform_r) * log_factor)
    if np.max(waveform_g) > 0:
        waveform_g = np.log1p(waveform_g * log_factor) / np.log1p(np.max(waveform_g) * log_factor)
    if np.max(waveform_b) > 0:
        waveform_b = np.log1p(waveform_b * log_factor) / np.log1p(np.max(waveform_b) * log_factor)
    
    # Create a combined RGB waveform with gradient filling (Darktable-style)
    waveform_rgb = np.ones((waveform_height, width, 4), dtype=np.float32)  # RGBA with alpha
    
    # Apply a gradient effect to each channel
    for i in range(waveform_height):
        # Apply gradient effect based on position (more opacity at the bottom)
        gradient_factor = 0.8 + 0.2 * (1.0 - i / waveform_height)
        
        # Create masks with anti-aliased edges for smoother appearance
        mask_r = waveform_r[i, :] > 0.001
        mask_g = waveform_g[i, :] > 0.001
        mask_b = waveform_b[i, :] > 0.001
        
        # Value-based opacity for better visualization
        alpha_r = waveform_r[i, :] * gradient_factor * 0.9  # Slightly reduced transparency
        alpha_g = waveform_g[i, :] * gradient_factor * 0.9
        alpha_b = waveform_b[i, :] * gradient_factor * 0.9
        
        # Define more saturated colors for better visibility
        r_color = np.array([1.0, 0.2, 0.2])  # Brighter red
        g_color = np.array([0.2, 1.0, 0.2])  # Brighter green
        b_color = np.array([0.2, 0.2, 1.0])  # Brighter blue
        
        # Apply colors with variable opacity
        if np.any(mask_r):
            waveform_rgb[i, mask_r, 0] = r_color[0]
            waveform_rgb[i, mask_r, 1] = r_color[1]
            waveform_rgb[i, mask_r, 2] = r_color[2]
            waveform_rgb[i, mask_r, 3] = alpha_r[mask_r]
        
        if np.any(mask_g):
            waveform_rgb[i, mask_g, 0] = g_color[0]
            waveform_rgb[i, mask_g, 1] = g_color[1]
            waveform_rgb[i, mask_g, 2] = g_color[2]
            waveform_rgb[i, mask_g, 3] = alpha_g[mask_g]
        
        if np.any(mask_b):
            waveform_rgb[i, mask_b, 0] = b_color[0]
            waveform_rgb[i, mask_b, 1] = b_color[1]
            waveform_rgb[i, mask_b, 2] = b_color[2]
            waveform_rgb[i, mask_b, 3] = alpha_b[mask_b]
        
        # Add combined/overlay colors for overlapping areas
        mask_rg = mask_r & mask_g
        mask_rb = mask_r & mask_b
        mask_gb = mask_g & mask_b
        mask_rgb = mask_r & mask_g & mask_b
        
        yellow = np.array([1.0, 1.0, 0.2])  # Brighter yellow
        magenta = np.array([1.0, 0.2, 1.0])  # Brighter magenta
        cyan = np.array([0.2, 1.0, 1.0])  # Brighter cyan
        white = np.array([1.0, 1.0, 1.0])  # White
        
        if np.any(mask_rg):
            waveform_rgb[i, mask_rg, 0] = yellow[0]
            waveform_rgb[i, mask_rg, 1] = yellow[1]
            waveform_rgb[i, mask_rg, 2] = yellow[2]
            waveform_rgb[i, mask_rg, 3] = np.maximum(alpha_r[mask_rg], alpha_g[mask_rg])
        
        if np.any(mask_rb):
            waveform_rgb[i, mask_rb, 0] = magenta[0]
            waveform_rgb[i, mask_rb, 1] = magenta[1]
            waveform_rgb[i, mask_rb, 2] = magenta[2]
            waveform_rgb[i, mask_rb, 3] = np.maximum(alpha_r[mask_rb], alpha_b[mask_rb])
        
        if np.any(mask_gb):
            waveform_rgb[i, mask_gb, 0] = cyan[0]
            waveform_rgb[i, mask_gb, 1] = cyan[1]
            waveform_rgb[i, mask_gb, 2] = cyan[2]
            waveform_rgb[i, mask_gb, 3] = np.maximum(alpha_g[mask_gb], alpha_b[mask_gb])
        
        if np.any(mask_rgb):
            waveform_rgb[i, mask_rgb, 0] = white[0]
            waveform_rgb[i, mask_rgb, 1] = white[1]
            waveform_rgb[i, mask_rgb, 2] = white[2]
            waveform_rgb[i, mask_rgb, 3] = np.maximum(np.maximum(alpha_r[mask_rgb], alpha_g[mask_rgb]), alpha_b[mask_rgb])
    
    # Create a clean white background
    ax.set_facecolor('#ffffff')
    
    # Display the waveform with proper alpha blending - higher quality interpolation
    ax.imshow(waveform_rgb, aspect='auto', origin='lower', interpolation='bicubic')
    
    # Set labels and title with a more professional font
    ax.set_title(title, fontsize=14, fontweight='bold', fontname='Arial')
    ax.set_xlabel('Position (left to right)', fontsize=12, fontname='Arial')
    ax.set_ylabel('Brightness (dark to bright)', fontsize=12, fontname='Arial')
    
    # Set y-axis limits to focus on the active range
    ax.set_ylim(min_row, max_row)
    
    # Calculate appropriate tick positions based on the visible range
    range_size = max_row - min_row
    if range_size > 400:
        step = 128
    elif range_size > 200:
        step = 64
    elif range_size > 100:
        step = 32
    else:
        step = 16
        
    tick_start = (min_row // step) * step
    tick_positions = np.arange(tick_start, max_row + step, step)
    tick_positions = tick_positions[tick_positions <= waveform_height-1]
    
    # Set ticks with nice formatting
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(int(255 * (1 - pos/waveform_height))) for pos in tick_positions], 
                     fontsize=10, fontname='Arial')
    
    # Remove x ticks as they don't have meaningful absolute values
    ax.set_xticks([])
    
    # Add subtle grid lines for better readability
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#888888')
    
    # Add thin gray lines at the boundaries of active data
    ax.axhline(y=min_row, color='#aaaaaa', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axhline(y=max_row, color='#aaaaaa', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Draw vertical markers at every quarter of the width for reference
    for frac in [0.25, 0.5, 0.75]:
        ax.axvline(x=width*frac, color='#888888', linestyle='--', linewidth=0.5, alpha=0.3)

def display_waveforms(app, pixel_data):
    """Display waveforms for each image region"""
    # Clear previous plots
    app.waveform_fig.clear()
    app.waveform_fig.patch.set_facecolor('#f0f0f0')  # Set a light gray background for the figure
    
    # Increase figure size for better resolution and clarity
    app.waveform_fig.set_size_inches(16, 10)
    
    # Add a bit more space between subplots
    app.waveform_fig.subplots_adjust(hspace=0.3)
    
    # Create subplots for each image with increased height
    axs = app.waveform_fig.subplots(3, 1, sharex=True)
    
    # Define the sources
    sources = ["Digital (DA)", "Negative 1", "Negative 2"]
    
    # Create a waveform for each source
    for i, source in enumerate(sources):
        if source in pixel_data:
            create_waveform(pixel_data[source], axs[i], f"Waveform - {source}")
            # Set background color for each subplot
            axs[i].set_facecolor('#ffffff')  # White background
        else:
            axs[i].text(0.5, 0.5, f"No data for {source}", 
                     ha='center', va='center', fontsize=14, fontname='Arial', fontweight='bold')
            axs[i].set_facecolor('#f8f8f8')  # Light gray background for empty plots
    
    # Add a general information text about waveforms with nicer formatting
    app.waveform_fig.text(0.01, 0.01, 
                        "Waveforms show brightness distribution across the image width.\n"
                        "X-axis: Position from left to right.  Y-axis: Brightness from dark (bottom) to bright (top).\n"
                        "RGB channels are overlaid with their respective colors. Vertical lines mark 25%, 50%, and 75% positions.", 
                        fontsize=10, fontname='Arial', color='#444444')
    
    # Add better spacing between subplots
    app.waveform_fig.tight_layout(pad=3.0)
    app.waveform_canvas.draw() 