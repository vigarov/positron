import numpy as np
from visualization import display_pixel_stats, display_waveforms
from image_processor import load_tiff_image
import os

def _extract_pixel_data(app, show_visualizations=True):
    """
    Extract pixel data from images and return it. 
    Common functionality used by both extract_pixels and process_and_save_data.
    
    Args:
        app: The application instance
        show_visualizations: Whether to show histograms and waveforms
        
    Returns:
        True if pixel extraction was successful, False otherwise
    """
    # Check if rectangles are defined for all images
    if app.multiple_rectangles:
        missing = [i for i in range(3) if not app.rectangles[i] or len(app.rectangles[i]) == 0]
    else:
        missing = [i for i in range(3) if app.rectangles[i] is None]
        
    if missing:
        missing_sources = [["Digital (DA)", "Negative 1", "Negative 2"][i] for i in missing]
        app.status_var.set(f"Missing rectangles for: {', '.join(missing_sources)}")
        return False
    
    # Extract pixel values from each image
    # In multiple rectangles mode, we'll create a separate pixel_data dictionary for each rectangle
    if app.multiple_rectangles:
        pixel_data_list = []
        num_rectangles = len(app.rectangles[0])
        
        for rect_idx in range(num_rectangles):
            pixel_data = {}
            
            # First check ROI dimensions for this rectangle
            roi_dimensions = []
            for src_idx in range(3):
                rect = app.rectangles[src_idx][rect_idx]
                x1, y1, x2, y2 = rect
                width = x2 - x1
                height = y2 - y1
                roi_dimensions.append((width, height))
                source_name = ["Digital (DA)", "Negative 1", "Negative 2"][src_idx]
                print(f"Rectangle {rect_idx+1}, ROI for {source_name}: {width}x{height} pixels")
            
            # Check for size discrepancies
            max_width_diff = max(roi_dimensions, key=lambda x: x[0])[0] - min(roi_dimensions, key=lambda x: x[0])[0]
            max_height_diff = max(roi_dimensions, key=lambda x: x[1])[1] - min(roi_dimensions, key=lambda x: x[1])[1]
            
            if max_width_diff > 5 or max_height_diff > 5:
                print(f"Warning for rectangle {rect_idx+1}: ROI size differences detected. Width diff: {max_width_diff}, Height diff: {max_height_diff}")
            
            # Extract ROIs for this rectangle
            for src_idx in range(3):
                try:
                    # Load the original image
                    img_path = app.image_paths[src_idx]
                    img_rgb = load_tiff_image(img_path)
                    
                    # Extract rectangle region
                    rect = app.rectangles[src_idx][rect_idx]
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
                    source_name = ["Digital (DA)", "Negative 1", "Negative 2"][src_idx]
                    pixel_data[source_name] = roi
                    print(f"Rectangle {rect_idx+1}, Extracted ROI for {source_name}: shape {roi.shape}")
                except Exception as e:
                    app.status_var.set(f"Error extracting pixels from {img_path}: {str(e)}")
                    print(f"Error extracting pixels from {img_path}: {str(e)}")
                    return False
            
            # Add this rectangle's pixel data to the list
            pixel_data_list.append(pixel_data)
        
        # Process each rectangle's pixel data
        for rect_idx, pixel_data in enumerate(pixel_data_list):
            # Find smallest ROI dimensions for this rectangle
            min_height = min(roi.shape[0] for roi in pixel_data.values())
            min_width = min(roi.shape[1] for roi in pixel_data.values())
            print(f"Rectangle {rect_idx+1}: Using minimum ROI dimensions: {min_width}x{min_height} for consistent coordinates")
            
            # Find unique pixels in Digital (DA) image for this rectangle
            da_roi = pixel_data.get("Digital (DA)")
            if da_roi is None:
                app.status_var.set(f"Error: Digital (DA) ROI not found for rectangle {rect_idx+1}")
                return False
                
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
            
            print(f"Rectangle {rect_idx+1}, Digital (DA): Found {len(rgb_to_coord)} unique RGB values in constrained ROI")
            
            # Determine K = minimum of 10 and number of unique RGB values
            k = min(10, len(rgb_to_coord))
            print(f"Rectangle {rect_idx+1}: Will select {k} unique pixels based on Digital (DA) image")
            
            # Convert unique RGB values to a list and randomly select K of them
            unique_rgb_list = list(rgb_to_coord.keys())
            if len(unique_rgb_list) > k:
                np.random.shuffle(unique_rgb_list)
                selected_rgb = unique_rgb_list[:k]
            else:
                selected_rgb = unique_rgb_list
            
            # Get the coordinates for these RGB values
            selected_positions = [rgb_to_coord[rgb] for rgb in selected_rgb]
            print(f"Rectangle {rect_idx+1}: Selected {len(selected_positions)} unique pixel positions in Digital (DA)")
            
            # Store for each rectangle
            if not hasattr(app, 'unique_pixels_by_source_multi'):
                app.unique_pixels_by_source_multi = []
            if not hasattr(app, 'selected_pixels_multi'):
                app.selected_pixels_multi = []
                
            # Use same coordinates for all sources in this rectangle
            rect_unique_pixels = {}
            for source in pixel_data.keys():
                rect_unique_pixels[source] = selected_positions
                roi = pixel_data[source]
                
                valid_count = sum(1 for y, x in selected_positions if y < roi.shape[0] and x < roi.shape[1])
                if valid_count < len(selected_positions):
                    print(f"Rectangle {rect_idx+1}, Warning: {len(selected_positions) - valid_count} coordinates out of bounds for {source}")
            
            app.unique_pixels_by_source_multi.append(rect_unique_pixels)
            app.selected_pixels_multi.append(selected_positions)
        
        # For visualization, use the first rectangle's data as current
        app.current_pixel_data = pixel_data_list[0]
        app.unique_pixels_by_source = app.unique_pixels_by_source_multi[0]
        app.selected_pixels = app.selected_pixels_multi[0]
        
        # Store all rectangles' data for processing
        app.all_rectangles_pixel_data = pixel_data_list
    else:
        # Original single rectangle mode
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
                return False
        
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
            return False
            
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
    
    # Update raw data display
    app.update_raw_data_display()
    
    # Show visualizations if requested
    if show_visualizations:
        # Display pixel statistics
        display_pixel_stats(app, app.current_pixel_data)
        
        # Display waveforms
        display_waveforms(app, app.current_pixel_data)
    
    return True

def extract_pixels(app):
    """Extract pixel values from selected regions in all images and display visualizations"""
    success = _extract_pixel_data(app, show_visualizations=True)
    
    if success:
        # Format the message with details
        status_msg = f"Pixel data extracted. Using {len(app.selected_pixels)} pixels with identical coordinates across all images."
        app.status_var.set(status_msg)
        print(f"Selected pixel coordinates: {app.selected_pixels}")
    
    return success

def process_and_save_data(app):
    """Extract pixels and automatically append data to the specified file without displaying visualizations"""
    # Extract pixel data without showing visualizations
    success = _extract_pixel_data(app, show_visualizations=False)
    
    if not success:
        return False
    
    # If data processing path is not set, stop here
    if not hasattr(app, 'data_processing_path') or not app.data_processing_path:
        app.status_var.set("No data processing path set")
        return False
    
    try:
        # Check if the file exists and has content
        file_exists = os.path.isfile(app.data_processing_path)
        file_has_content = file_exists and os.path.getsize(app.data_processing_path) > 0
        
        with open(app.data_processing_path, 'a') as f:
            # If file is new or empty, include the header
            if not file_has_content:
                # Generate header once
                header_line = _generate_csv_header(include_image_id=True)
                f.write(header_line + '\n')
            
            # For multiple rectangles mode, process each rectangle
            if app.multiple_rectangles and hasattr(app, 'all_rectangles_pixel_data'):
                for rect_idx, pixel_data in enumerate(app.all_rectangles_pixel_data):
                    # Set current data for this rectangle
                    app.current_pixel_data = pixel_data
                    app.selected_pixels = app.selected_pixels_multi[rect_idx]
                    
                    # Generate CSV data for this rectangle (without header)
                    csv_data = generate_combined_csv_data(app, include_stats=False, include_image_id=True, include_rectangle_id=True, rectangle_id=rect_idx+1)
                    csv_lines = csv_data.split('\n')
                    data_lines = csv_lines[1:]  # Skip header
                    
                    # Add a newline before the data if needed
                    if file_has_content or rect_idx > 0:
                        f.write('\n')
                    
                    # Add the data lines for this rectangle
                    f.write('\n'.join(data_lines))
            else:
                # Single rectangle mode
                csv_data = generate_combined_csv_data(app, include_stats=False, include_image_id=True)
                csv_lines = csv_data.split('\n')
                data_lines = csv_lines[1:]  # Skip header
                
                # Add a newline before the data if file already has content
                if file_has_content:
                    f.write('\n')
                
                # Add data lines
                f.write('\n'.join(data_lines))
            
            # Final newline
            f.write('\n')
        
        if app.multiple_rectangles:
            num_rectangles = len(app.rectangles[0])
            app.status_var.set(f"Data for {num_rectangles} rectangles successfully appended to {app.data_processing_path}")
            print(f"Data for Image Set {app.current_set_id} with {num_rectangles} rectangles appended to {app.data_processing_path}")
        else:
            app.status_var.set(f"Data successfully appended to {app.data_processing_path}")
            print(f"Data for Image Set {app.current_set_id} appended to {app.data_processing_path}")
        
        return True
        
    except Exception as e:
        app.status_var.set(f"Error saving data: {str(e)}")
        print(f"Error saving data to {app.data_processing_path}: {str(e)}")
        return False

def _generate_csv_header(include_image_id=False, include_rectangle_id=False):
    """Generate the CSV header"""
    header = []
    
    # Add image_id as first column if requested
    if include_image_id:
        header.append("image_id")
        
    # Add rectangle_id if in multi-rectangle mode
    if include_rectangle_id:
        header.append("rectangle_id")
    else:
        # Only include pixel_num when not adding rectangle_id
        header.append("pixel_num")
        
    # Add coordinate columns
    header.extend(["y_rect", "x_rect"])
    
    # Add raw coordinate columns when in data processing mode
    if include_image_id:
        header.extend(["y_raw", "x_raw"])
    
    # Add columns for each source and channel
    for source in ["Digital (DA)", "Negative 1", "Negative 2"]:
        pic_id = {"Digital (DA)": "DA", "Negative 1": "N1", "Negative 2": "N2"}[source]
        for channel in ['R', 'G', 'B']:
            header.append(f"{pic_id}_{channel}")
    
    return ",".join(header)

def generate_combined_csv_data(app, include_stats=True, include_image_id=False, include_rectangle_id=False, rectangle_id=None):
    """Generate a combined CSV with data from all three images side by side
    
    Args:
        app: The application instance
        include_stats: Whether to include statistics (min, max, avg) at the end
        include_image_id: Whether to include the image set ID as a column
        include_rectangle_id: Whether to include the rectangle ID as a column
        rectangle_id: The ID of the current rectangle (only used if include_rectangle_id is True)
    """
    # Get all sources
    all_sources = ["Digital (DA)", "Negative 1", "Negative 2"]
    available_sources = [s for s in all_sources if s in app.current_pixel_data]
    
    if not available_sources:
        return "No data available from any source"
    
    # Map source names to their picture IDs
    pic_ids = {"Digital (DA)": "DA", "Negative 1": "N1", "Negative 2": "N2"}
    
    # Generate header
    header = _generate_csv_header(include_image_id, include_rectangle_id)
    csv_lines = [header]
    
    # If we have pixel positions, use those
    if hasattr(app, 'selected_pixels') and app.selected_pixels:
        pixel_positions = app.selected_pixels
        
        # Generate data for each pixel position
        for i, (y, x) in enumerate(pixel_positions):
            pixel_num = i + 1  # 1-based pixel numbering
            
            # Start building the data row
            line_data = []
            
            # Add image_id if requested
            if include_image_id:
                line_data.append(str(app.current_set_id))
            
            # Add rectangle_id or pixel_num
            if include_rectangle_id:
                line_data.append(str(rectangle_id))
            else:
                line_data.append(str(pixel_num))
                
            # Add pixel coordinates relative to rectangle
            line_data.extend([str(y), str(x)])
            
            # Add raw coordinates when in data processing mode
            if include_image_id:
                # Get the rectangle coordinates for the first available source (Digital DA)
                first_source = available_sources[0]
                rect_idx = ["Digital (DA)", "Negative 1", "Negative 2"].index(first_source)
                
                if app.multiple_rectangles:
                    # For multiple rectangles mode, get the rectangle by index
                    if rectangle_id is not None and rect_idx in app.rectangles:
                        rect = app.rectangles[rect_idx][rectangle_id-1]  # rectangle_id is 1-based
                        x1, y1, x2, y2 = rect
                        # Calculate raw coordinates by adding rectangle origin
                        raw_y = y + y1
                        raw_x = x + x1
                        line_data.extend([str(raw_y), str(raw_x)])
                    else:
                        line_data.extend(["NaN", "NaN"])
                else:
                    # For single rectangle mode
                    if rect_idx in app.rectangles:
                        rect = app.rectangles[rect_idx]
                        x1, y1, x2, y2 = rect
                        # Calculate raw coordinates by adding rectangle origin
                        raw_y = y + y1
                        raw_x = x + x1
                        line_data.extend([str(raw_y), str(raw_x)])
                    else:
                        # Fallback if rectangle info is not available
                        line_data.extend(["NaN", "NaN"])
            
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
        
        # Add statistics at the end if requested
        if include_stats:
            csv_lines.append("")  # Empty line
            
            # Add min, max, average values for each source/channel
            for stat_type in ["Min", "Max", "Average"]:
                stat_line = []
                
                # Add empty cell for first column (image_id, rectangle_id, or pixel_num)
                stat_line.append("")
                
                # Add empty cell for rectangle_id if included
                if include_rectangle_id:
                    stat_line.append("")
                
                # Add stat type and empty cell for coordinates
                stat_line.append(stat_type)
                stat_line.append("")
                
                # Add empty cells for raw coordinates if they're included
                if include_image_id:
                    stat_line.extend(["", ""])
                
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