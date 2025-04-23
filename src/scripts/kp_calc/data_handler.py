import numpy as np
from visualization import display_pixel_stats, display_waveforms
from image_processor import load_tiff_image

def extract_pixels(app):
    """Extract pixel values from selected regions in all images"""
    # Check if rectangles are defined for all images
    missing = [i for i in range(3) if app.rectangles[i] is None]
    if missing:
        missing_sources = [["Digital (DA)", "Negative 1", "Negative 2"][i] for i in missing]
        app.status_var.set(f"Missing rectangles for: {', '.join(missing_sources)}")
        return
    
    # Extract pixel values from each image
    pixel_data = {}
    
    # First check all ROI dimensions to see if there are large size discrepancies
    roi_dimensions = []
    for idx, rect in app.rectangles.items():
        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        roi_dimensions.append((width, height))
        source_name = ["Digital (DA)", "Negative 1", "Negative 2"][idx]
        print(f"ROI for {source_name}: {width}x{height} pixels")
    
    # Check for size discrepancies
    if len(roi_dimensions) == 3:
        max_width_diff = max(roi_dimensions, key=lambda x: x[0])[0] - min(roi_dimensions, key=lambda x: x[0])[0]
        max_height_diff = max(roi_dimensions, key=lambda x: x[1])[1] - min(roi_dimensions, key=lambda x: x[1])[1]
        
        if max_width_diff > 5 or max_height_diff > 5:
            app.status_var.set(f"Warning: ROI size differences detected. Width diff: {max_width_diff}, Height diff: {max_height_diff}")
            print(f"Warning: ROI size differences may cause issue with exact pixel coordinates.")
            print(f"To ensure consistent results, try to draw rectangles of the same size.")
    
    # Now extract the ROIs
    for idx, rect in app.rectangles.items():
        try:
            # Load the original image
            img_path = app.image_paths[idx]
            img_rgb = load_tiff_image(img_path)
            
            # Extract rectangle region
            x1, y1, x2, y2 = rect
            
            # Ensure boundaries are within image
            img_h, img_w = img_rgb.shape[:2]
            x1 = max(0, min(x1, img_w-1))
            y1 = max(0, min(y1, img_h-1))
            x2 = max(0, min(x2, img_w-1))
            y2 = max(0, min(y2, img_h-1))
            
            # Check if ROI is valid
            if x1 >= x2 or y1 >= y2:
                raise ValueError(f"Invalid ROI: ({x1},{y1}) to ({x2},{y2})")
            
            # Extract ROI
            roi = img_rgb[y1:y2, x1:x2]
            
            # Store data
            source_name = ["Digital (DA)", "Negative 1", "Negative 2"][idx]
            pixel_data[source_name] = roi
            print(f"Extracted ROI for {source_name}: shape {roi.shape}")
        except Exception as e:
            app.status_var.set(f"Error extracting pixels from {img_path}: {str(e)}")
            print(f"Error extracting pixels from {img_path}: {str(e)}")
            return
    
    # Store the current pixel data for saving later
    app.current_pixel_data = pixel_data
    
    # Find smallest ROI dimensions to ensure coordinates work across all images
    min_height = min(roi.shape[0] for roi in pixel_data.values())
    min_width = min(roi.shape[1] for roi in pixel_data.values())
    print(f"Using minimum ROI dimensions: {min_width}x{min_height} for consistent coordinates")
    
    # First, find unique pixels in Digital (DA) image
    da_roi = pixel_data.get("Digital (DA)")
    if da_roi is None:
        app.status_var.set("Error: Digital (DA) ROI not found")
        return
        
    # We'll only consider the pixels within the minimum dimensions
    da_constrained = da_roi[:min_height, :min_width]
    
    # Analyze the Digital (DA) ROI to count unique RGB values
    da_flattened = da_constrained.reshape(-1, da_constrained.shape[2])
    
    # Build a map of RGB values to coordinates
    rgb_to_coord = {}
    for y in range(min_height):
        for x in range(min_width):
            rgb = tuple(da_constrained[y, x])
            if rgb not in rgb_to_coord:
                rgb_to_coord[rgb] = (y, x)
    
    print(f"Digital (DA): Found {len(rgb_to_coord)} unique RGB values in constrained ROI")
    
    # Determine K = minimum of 10 and number of unique RGB values
    k = min(10, len(rgb_to_coord))
    print(f"Will select {k} unique pixels based on Digital (DA) image")
    
    # Convert unique RGB values to a list and randomly select K of them
    unique_rgb_list = list(rgb_to_coord.keys())
    if len(unique_rgb_list) > k:
        np.random.shuffle(unique_rgb_list)
        selected_rgb = unique_rgb_list[:k]
    else:
        selected_rgb = unique_rgb_list
    
    # Get the coordinates for these RGB values
    selected_positions = [rgb_to_coord[rgb] for rgb in selected_rgb]
    print(f"Selected {len(selected_positions)} unique pixel positions in Digital (DA)")
    
    # Store the exact same coordinates for each image source
    app.unique_pixels_by_source = {}
    for source in pixel_data.keys():
        # Use the EXACT same coordinates for all sources
        app.unique_pixels_by_source[source] = selected_positions
        roi = pixel_data[source]
        
        # Validate that all positions are valid for this ROI
        valid_count = sum(1 for y, x in selected_positions if y < roi.shape[0] and x < roi.shape[1])
        if valid_count < len(selected_positions):
            print(f"Warning: {len(selected_positions) - valid_count} coordinates out of bounds for {source}")
    
    # Store the selected positions as reference
    app.selected_pixels = selected_positions
        
    # Display pixel statistics
    display_pixel_stats(app, pixel_data)
    
    # Display waveforms
    display_waveforms(app, pixel_data)
    
    # Update raw data display
    app.update_raw_data_display()
    
    # Format the message with details
    status_msg = f"Pixel data extracted. Using {len(selected_positions)} pixels with identical coordinates across all images."
    app.status_var.set(status_msg)
    print(f"Selected pixel coordinates: {selected_positions}")

def generate_combined_csv_data(app):
    """Generate a combined CSV with data from all three images side by side"""
    # Get all sources
    all_sources = ["Digital (DA)", "Negative 1", "Negative 2"]
    available_sources = [s for s in all_sources if s in app.current_pixel_data]
    
    if not available_sources:
        return "No data available from any source"
    
    # Map source names to their picture IDs
    pic_ids = {"Digital (DA)": "DA", "Negative 1": "N1", "Negative 2": "N2"}
    
    # Start with CSV header
    header = ["pixel_num", "y_rect", "x_rect"]
    
    # Add columns for each source and channel
    for source in available_sources:
        for channel in ['R', 'G', 'B']:
            header.append(f"{pic_ids[source]}_{channel}")
    
    csv_lines = [",".join(header)]
    
    # If we have pixel positions, use those
    if hasattr(app, 'selected_pixels') and app.selected_pixels:
        pixel_positions = app.selected_pixels
        
        # Generate data for each pixel position
        for i, (y, x) in enumerate(pixel_positions):
            pixel_num = i + 1  # 1-based pixel numbering
            
            # Start with the pixel coordinates
            line_data = [str(pixel_num), str(y), str(x)]
            
            # Add pixel values for each source and channel
            for source in available_sources:
                img_data = app.current_pixel_data[source]
                height, width = img_data.shape[:2]
                
                # Check if this position is valid for this image
                if y < height and x < width:
                    pixel = img_data[y, x]
                    # Add each channel value
                    for c_idx, _ in enumerate(['R', 'G', 'B']):
                        pixel_value = int(pixel[c_idx])
                        line_data.append(str(pixel_value))
                else:
                    # If position is out of bounds, add empty values
                    line_data.extend(["N/A", "N/A", "N/A"])
            
            # Add this line to the CSV
            csv_lines.append(",".join(line_data))
            
        # Add some statistics at the end
        csv_lines.append("")  # Empty line
        
        # Add min, max, average values for each source/channel
        for stat_type in ["Min", "Max", "Average"]:
            stat_line = [stat_type, "", ""]  # Empty for y_rect, x_rect
            
            for source in available_sources:
                img_data = app.current_pixel_data[source]
                height, width = img_data.shape[:2]
                
                # Collect valid pixel values for each channel
                channel_values = [[], [], []]  # R, G, B
                
                for y, x in pixel_positions:
                    if y < height and x < width:
                        pixel = img_data[y, x]
                        for c_idx in range(3):
                            channel_values[c_idx].append(int(pixel[c_idx]))
                
                # Add statistics for each channel
                for c_idx, values in enumerate(channel_values):
                    if values:
                        if stat_type == "Min":
                            stat_line.append(str(min(values)))
                        elif stat_type == "Max":
                            stat_line.append(str(max(values)))
                        elif stat_type == "Average":
                            avg = sum(values) / len(values)
                            stat_line.append(f"{avg:.1f}")
                    else:
                        stat_line.append("N/A")
                        
            csv_lines.append(",".join(stat_line))
        
    else:
        csv_lines.append("No pixel positions selected")
    
    # Join into a single string
    return "\n".join(csv_lines) 