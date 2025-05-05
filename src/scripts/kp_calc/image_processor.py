import numpy as np
import tifffile
from PIL import Image, ImageTk
import os

def load_tiff_image(path):
    """Load a TIFF image using tifffile library"""
    try:
        # Use tifffile to read the TIFs ; many of the DarkTable converted-files do not work with PIL/rawpy
        img_array = tifffile.imread(path)
        
        if img_array.dtype != np.uint8:
            # Convert to 8-bit for display
            if img_array.max() > 0:
                img_normalized = (img_array / img_array.max() * 255).astype(np.uint8)
            else:
                img_normalized = img_array.astype(np.uint8)
            img_array = img_normalized
        
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=2)
        elif len(img_array.shape) == 3 and img_array.shape[2] > 3:
            raise ValueError("TIFF image has more than 3 channels")
        
        return img_array
        
    except Exception as e:
        print(f"Error loading TIFF with tifffile: {str(e)}")
        return np.ones((50, 50, 3), dtype=np.uint8) * np.array([255, 0, 0], dtype=np.uint8)

def load_current_set(app):
    try:
        relative_paths = app.image_mappings[app.current_set_id]
        
        app.image_paths = [os.path.join(app.root_directory, path) for path in relative_paths]
        
        app.set_var.set(str(app.current_set_id))
        
        if not hasattr(app, 'all_rectangles_drawn') or not app.all_rectangles_drawn:
            if hasattr(app, 'multiple_rectangles') and app.multiple_rectangles:
                app.rectangles = {0: [], 1: [], 2: []}
                app.all_rects_drawn_on_da = False
                app.multi_rect_placement_mode = False
                app.rect_offsets = []
            else:
                app.rectangles = {0: None, 1: None, 2: None}
            
            if hasattr(app, 'all_rectangles_drawn'):
                app.all_rectangles_drawn = False
            if hasattr(app, 'pixels_extracted'):
                app.pixels_extracted = False
        
        load_current_image(app)
        
        if hasattr(app, 'multiple_rectangles') and app.multiple_rectangles:
            app.status_var.set(f"Loaded image set {app.current_set_id}. Draw multiple rectangles on the Digital (DA) image.")
        else:
            app.status_var.set(f"Loaded image set {app.current_set_id}. Draw rectangle on the Digital (DA) image.")
        
        if hasattr(app, 'update_ui_state'):
            app.update_ui_state()
        
    except Exception as e:
        app.status_var.set(f"Error loading image set: {str(e)}")
        print(f"Error: {str(e)}")

def load_current_image(app):
    try:
        img_path = app.image_paths[app.current_img_idx]
        
        sources = ["Digital (DA)", "Negative 1", "Negative 2"]
        app.source_var.set(sources[app.current_img_idx])
        
        app.current_img_rgb = load_tiff_image(img_path)
        
        img_h, img_w = app.current_img_rgb.shape[:2]
        
        # Calculate scaling to fit in the fixed display size while preserving aspect ratio
        scale_w = app.display_width / img_w
        scale_h = app.display_height / img_h
        app.scale_factor = min(scale_w, scale_h)
        
        new_width = int(img_w * app.scale_factor)
        new_height = int(img_h * app.scale_factor)
        
        pil_img = Image.fromarray(app.current_img_rgb)
        display_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        app.tk_img = ImageTk.PhotoImage(image=display_img)
        
        app.canvas.delete("all")
        
        canvas_width = app.canvas.winfo_width()
        canvas_height = app.canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = app.display_width
        if canvas_height <= 1:
            canvas_height = app.display_height
            
        # Center the image in the canvas
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        
        # Light gray rectangle to represent the image area
        app.canvas.create_rectangle(
            x_offset-1, y_offset-1, 
            x_offset+new_width+1, y_offset+new_height+1, 
            outline="#999999", width=1, fill="#dddddd"
        )
        
        app.canvas.create_image(x_offset, y_offset, anchor='nw', image=app.tk_img)
        
        # Store the offsets for later use in coordinate calculations
        app.img_x_offset = x_offset
        app.img_y_offset = y_offset
        
        # Redraw rectangle(s) if they exist for this image
        if hasattr(app, 'multiple_rectangles') and app.multiple_rectangles:
            # For multiple rectangles mode, redraw all rectangles
            if app.rectangles[app.current_img_idx] and len(app.rectangles[app.current_img_idx]) > 0:
                for i, rect in enumerate(app.rectangles[app.current_img_idx]):
                    x1, y1, x2, y2 = rect
                    x1 = int(x1 * app.scale_factor) + app.img_x_offset
                    y1 = int(y1 * app.scale_factor) + app.img_y_offset
                    x2 = int(x2 * app.scale_factor) + app.img_x_offset
                    y2 = int(y2 * app.scale_factor) + app.img_y_offset
                    
                    # Determine color based on whether it's been adjusted
                    if (hasattr(app, 'adjusted_rectangles') and
                        app.current_img_idx in app.adjusted_rectangles and
                        i < len(app.adjusted_rectangles[app.current_img_idx])):
                        # This rectangle has been adjusted by the user - draw in red
                        color = "red"
                    elif (hasattr(app, 'current_rect_idx') and
                          i == app.current_rect_idx and
                          not app.is_rectangle_placed):
                        # This is the current rectangle being positioned - draw in blue
                        color = "blue"
                        
                        # Draw a circle in the center to indicate it's the next one to position
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        circle_radius = 5
                        app.canvas.create_oval(
                            center_x - circle_radius, center_y - circle_radius,
                            center_x + circle_radius, center_y + circle_radius,
                            fill="blue", outline="white", tags="rect"
                        )
                    else:
                        # This rectangle is calculated based on transformation - draw in blue
                        color = "blue"
                    
                    app.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="rect")
        else:
            # For single rectangle mode
            if app.rectangles[app.current_img_idx] is not None:
                x1, y1, x2, y2 = app.rectangles[app.current_img_idx]
                x1 = int(x1 * app.scale_factor) + app.img_x_offset
                y1 = int(y1 * app.scale_factor) + app.img_y_offset
                x2 = int(x2 * app.scale_factor) + app.img_x_offset
                y2 = int(y2 * app.scale_factor) + app.img_y_offset
                app.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="rect")
        
    except Exception as e:
        app.status_var.set(f"Error loading image: {str(e)}")
        print(f"Error: {str(e)}") 