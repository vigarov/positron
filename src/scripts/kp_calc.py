import os
import json
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import tifffile
from matplotlib.colors import LinearSegmentedColormap
import platform  # For OS detection
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

class RectangleDrawingApp:
    def __init__(self, root, image_mappings, num_sets=6):
        self.root = root
        self.root.title("Rectangle Drawing Tool")
        self.root.geometry("1600x1000")  # Increase initial window height
        
        # Maximize the window by default based on OS
        self.maximize_window()
        
        # Load first N sets from mappings
        self.image_mappings = {int(k): v for k, v in image_mappings.items() if int(k) < num_sets}
        self.current_set_id = 0
        self.current_img_idx = 0  # 0=da_tiff, 1=source1, 2=source2
        
        # Rectangle coordinates for each image in the current set
        self.rectangles = {0: None, 1: None, 2: None}  # (x1, y1, x2, y2)
        self.drawing = False
        
        # For rectangle size synchronization
        self.rect_width = None
        self.rect_height = None
        self.rect_placement_mode = False  # True when placing a fixed-size rectangle
        
        # Initialize scale factor
        self.scale_factor = 1.0
        
        # Current pixel data
        self.current_pixel_data = None
        
        # Fixed display size for all images
        self.display_width = 800
        self.display_height = 600
        
        # Setup UI
        self.setup_ui()
        
        # Load images for the first set
        self.load_current_set()
        
    def maximize_window(self):
        """Maximize the window based on the detected operating system"""
        os_name = platform.system()
        
        try:
            if os_name == "Windows":
                self.root.state('zoomed')
            elif os_name == "Linux":
                self.root.attributes('-zoomed', True)
            elif os_name == "Darwin":  # macOS
                # On macOS, we'll use a combination of geometry and state
                # Full screen can be disruptive, so we'll make it almost full screen
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                self.root.geometry(f"{screen_width}x{screen_height-80}+0+0")  # Leave space for menu bar
                # Alternative for true fullscreen: self.root.attributes('-fullscreen', True)
            else:
                # For other OS, try a generic approach
                self.root.attributes('-fullscreen', True)
                # Add escape key binding to exit fullscreen if needed
                self.root.bind("<Escape>", lambda event: self.root.attributes("-fullscreen", False))
            
            self.status_var.set(f"Window maximized for {os_name} OS")
        except Exception as e:
            print(f"Could not maximize window: {str(e)}")
            # If maximizing fails, at least make it a good size
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.root.geometry(f"{int(screen_width*0.8)}x{int(screen_height*0.8)}+50+50")
        
    def setup_ui(self):
        # Main frame with scrollbar
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Add canvas and scrollbar for scrolling
        self.canvas_main = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient=tk.VERTICAL, command=self.canvas_main.yview)
        self.canvas_main.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Main frame inside canvas for scrolling
        self.main_frame = ttk.Frame(self.canvas_main)
        self.canvas_window = self.canvas_main.create_window((0, 0), window=self.main_frame, anchor=tk.NW)
        
        # Configure scroll region when frame size changes
        self.main_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas_main.bind("<Configure>", self.on_canvas_configure)
        
        # Top control panel
        self.control_panel = ttk.Frame(self.main_frame)
        self.control_panel.pack(fill=tk.X, pady=5)
        
        # Set selector
        ttk.Label(self.control_panel, text="Image Set:").pack(side=tk.LEFT, padx=5)
        self.set_var = tk.StringVar(value=str(self.current_set_id))
        set_values = list(self.image_mappings.keys())
        self.set_combo = ttk.Combobox(self.control_panel, textvariable=self.set_var, 
                                     values=set_values, width=5, state="readonly")
        self.set_combo.pack(side=tk.LEFT, padx=5)
        self.set_combo.bind("<<ComboboxSelected>>", self.on_set_changed)
        
        # Source selector
        ttk.Label(self.control_panel, text="Source:").pack(side=tk.LEFT, padx=5)
        self.source_var = tk.StringVar(value="Digital (DA)")
        sources = ["Digital (DA)", "Negative 1", "Negative 2"]
        self.source_combo = ttk.Combobox(self.control_panel, textvariable=self.source_var, 
                                        values=sources, width=12, state="readonly")
        self.source_combo.pack(side=tk.LEFT, padx=5)
        self.source_combo.bind("<<ComboboxSelected>>", self.on_source_changed)
        
        # Buttons
        ttk.Button(self.control_panel, text="Clear Rectangle", 
                  command=self.clear_rectangle).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_panel, text="Extract Pixels", 
                  command=self.extract_pixels).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_panel, text="Save Results", 
                  command=self.save_results).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready. Draw rectangle by clicking and dragging.")
        ttk.Label(self.control_panel, textvariable=self.status_var).pack(side=tk.LEFT, padx=20)
        
        # Image display frame
        self.img_frame = ttk.Frame(self.main_frame)
        self.img_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create canvas for image display
        self.canvas = tk.Canvas(self.img_frame, bg="#f0f0f0", width=self.display_width, height=self.display_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Make sure the canvas updates dimensions when resized
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Results display frame with tabs
        self.results_frame = ttk.Frame(self.main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabs for different result views
        self.results_tabs = ttk.Notebook(self.results_frame)
        self.results_tabs.pack(fill=tk.BOTH, expand=True)
        
        # Tab for histograms
        self.hist_tab = ttk.Frame(self.results_tabs)
        self.results_tabs.add(self.hist_tab, text="Histograms")
        
        # Figure for histograms - make it bigger
        self.fig = plt.Figure(figsize=(15, 6), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.hist_tab)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab for waveforms
        self.waveform_tab = ttk.Frame(self.results_tabs)
        self.results_tabs.add(self.waveform_tab, text="Waveforms")
        
        # Figure for waveforms - increased size and higher DPI
        self.waveform_fig = plt.Figure(figsize=(16, 10), dpi=150)
        self.waveform_canvas = FigureCanvasTkAgg(self.waveform_fig, master=self.waveform_tab)
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab for raw pixel data in CSV format
        self.raw_data_tab = ttk.Frame(self.results_tabs)
        self.results_tabs.add(self.raw_data_tab, text="Raw Data")
        
        # Create a frame with scrollbars for the raw data text
        raw_data_frame = ttk.Frame(self.raw_data_tab)
        raw_data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollbars
        raw_data_scroll_y = ttk.Scrollbar(raw_data_frame, orient=tk.VERTICAL)
        raw_data_scroll_x = ttk.Scrollbar(raw_data_frame, orient=tk.HORIZONTAL)
        
        # Create a text widget for the raw data
        self.raw_data_text = tk.Text(raw_data_frame, wrap=tk.NONE, height=25, width=80,
                                   yscrollcommand=raw_data_scroll_y.set,
                                   xscrollcommand=raw_data_scroll_x.set,
                                   font=("Courier New", 10))
        
        # Configure scrollbars
        raw_data_scroll_y.config(command=self.raw_data_text.yview)
        raw_data_scroll_x.config(command=self.raw_data_text.xview)
        
        # Place elements
        raw_data_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        raw_data_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.raw_data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add a dropdown to select the source
        raw_data_controls = ttk.Frame(self.raw_data_tab)
        raw_data_controls.pack(fill=tk.X, pady=5)
        
        ttk.Label(raw_data_controls, text="Source:").pack(side=tk.LEFT, padx=5)
        self.raw_data_source_var = tk.StringVar(value="Digital (DA)")
        raw_data_source_combo = ttk.Combobox(raw_data_controls, textvariable=self.raw_data_source_var, 
                                           values=sources, width=12, state="readonly")
        raw_data_source_combo.pack(side=tk.LEFT, padx=5)
        raw_data_source_combo.bind("<<ComboboxSelected>>", self.update_raw_data_display)
        
        # Add copy to clipboard button
        ttk.Button(raw_data_controls, text="Copy to Clipboard", 
                  command=self.copy_raw_data_to_clipboard).pack(side=tk.LEFT, padx=5)
    
    def load_tiff_image(self, path):
        """Load a TIFF image using tifffile library"""
        try:
            # Use tifffile to read the TIFF image
            img_array = tifffile.imread(path)
            
            # Handle various bit depths and formats
            if img_array.dtype != np.uint8:
                # Convert to 8-bit for display
                if img_array.max() > 0:
                    img_normalized = (img_array / img_array.max() * 255).astype(np.uint8)
                else:
                    img_normalized = img_array.astype(np.uint8)
                img_array = img_normalized
            
            # Ensure it's a 3-channel RGB image
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=2)
            elif len(img_array.shape) == 3 and img_array.shape[2] > 3:
                # More than 3 channels, keep just RGB
                img_array = img_array[:, :, :3]
            
            return img_array
            
        except Exception as e:
            print(f"Error loading TIFF with tifffile: {str(e)}")
            # Return a placeholder colored image (50x50 red square)
            return np.ones((50, 50, 3), dtype=np.uint8) * np.array([255, 0, 0], dtype=np.uint8)
        
    def load_current_set(self):
        try:
            # Get file paths for current set
            self.image_paths = self.image_mappings[self.current_set_id]
            
            # Update set variable
            self.set_var.set(str(self.current_set_id))
            
            # Reset rectangles
            self.rectangles = {0: None, 1: None, 2: None}
            
            # Load current image based on selected source
            self.load_current_image()
            
            # Update status
            self.status_var.set(f"Loaded image set {self.current_set_id}. Draw rectangle by clicking and dragging.")
            
        except Exception as e:
            self.status_var.set(f"Error loading image set: {str(e)}")
            print(f"Error: {str(e)}")
    
    def load_current_image(self):
        try:
            # Get current image path
            img_path = self.image_paths[self.current_img_idx]
            
            # Update source variable
            sources = ["Digital (DA)", "Negative 1", "Negative 2"]
            self.source_var.set(sources[self.current_img_idx])
            
            # Load image with tifffile
            self.current_img_rgb = self.load_tiff_image(img_path)
            
            # Get original image dimensions
            img_h, img_w = self.current_img_rgb.shape[:2]
            
            # Calculate scaling to fit in the fixed display size while preserving aspect ratio
            scale_w = self.display_width / img_w
            scale_h = self.display_height / img_h
            self.scale_factor = min(scale_w, scale_h)
            
            # Calculate new dimensions
            new_width = int(img_w * self.scale_factor)
            new_height = int(img_h * self.scale_factor)
            
            # Convert to PIL for resizing
            pil_img = Image.fromarray(self.current_img_rgb)
            display_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to ImageTk
            self.tk_img = ImageTk.PhotoImage(image=display_img)
            
            # Display on canvas
            self.canvas.delete("all")
            
            # Get the actual canvas size (it might be different from self.display_width/height)
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # If the canvas hasn't been realized yet, use the original dimensions
            if canvas_width <= 1:  # Canvas not yet drawn
                canvas_width = self.display_width
            if canvas_height <= 1:
                canvas_height = self.display_height
                
            # Center the image in the canvas
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            # Create a light gray rectangle to represent the image area
            self.canvas.create_rectangle(
                x_offset-1, y_offset-1, 
                x_offset+new_width+1, y_offset+new_height+1, 
                outline="#999999", width=1, fill="#dddddd"
            )
            
            self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_img)
            
            # Store the offsets for later use in coordinate calculations
            self.img_x_offset = x_offset
            self.img_y_offset = y_offset
            
            # Redraw rectangle if it exists for this image
            if self.rectangles[self.current_img_idx] is not None:
                x1, y1, x2, y2 = self.rectangles[self.current_img_idx]
                x1 = int(x1 * self.scale_factor) + self.img_x_offset
                y1 = int(y1 * self.scale_factor) + self.img_y_offset
                x2 = int(x2 * self.scale_factor) + self.img_x_offset
                y2 = int(y2 * self.scale_factor) + self.img_y_offset
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="rect")
            
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
            print(f"Error: {str(e)}")
    
    def on_set_changed(self, event):
        try:
            self.current_set_id = int(self.set_var.get())
            self.load_current_set()
        except ValueError:
            pass
    
    def on_source_changed(self, event):
        source = self.source_var.get()
        if source == "Digital (DA)":
            self.current_img_idx = 0
            self.rect_placement_mode = False
        elif source == "Negative 1":
            self.current_img_idx = 1
            # Check if DA rectangle exists
            if self.rectangles[0] is not None and self.rectangles[1] is None:
                # Get width and height from DA rectangle
                x1, y1, x2, y2 = self.rectangles[0]
                self.rect_width = x2 - x1
                self.rect_height = y2 - y1
                self.rect_placement_mode = True
                self.status_var.set("Position the rectangle using the same size as DA image")
            else:
                self.rect_placement_mode = False
        elif source == "Negative 2":
            self.current_img_idx = 2
            # Check if DA rectangle exists
            if self.rectangles[0] is not None and self.rectangles[2] is None:
                # Get width and height from DA rectangle
                x1, y1, x2, y2 = self.rectangles[0]
                self.rect_width = x2 - x1
                self.rect_height = y2 - y1
                self.rect_placement_mode = True
                self.status_var.set("Position the rectangle using the same size as DA image")
            else:
                self.rect_placement_mode = False
        
        self.load_current_image()
    
    def on_mouse_down(self, event):
        # Check if the click is within the image area
        if (self.img_x_offset <= event.x < self.img_x_offset + self.tk_img.width() and
            self.img_y_offset <= event.y < self.img_y_offset + self.tk_img.height()):
            self.drawing = True
            self.start_x, self.start_y = event.x, event.y
            
            # Clear existing rectangle
            self.canvas.delete("rect")
            
            # If in placement mode, create the rectangle immediately with fixed size
            if self.rect_placement_mode:
                # Convert the fixed width and height to canvas coordinates
                canvas_width = int(self.rect_width * self.scale_factor)
                canvas_height = int(self.rect_height * self.scale_factor)
                
                # Draw the rectangle centered at the click point
                half_width = canvas_width // 2
                half_height = canvas_height // 2
                self.start_x = event.x - half_width
                self.start_y = event.y - half_height
                
                self.canvas.create_rectangle(
                    self.start_x, self.start_y,
                    self.start_x + canvas_width, self.start_y + canvas_height,
                    outline="red", width=2, tags="rect"
                )
    
    def on_mouse_move(self, event):
        if self.drawing:
            self.canvas.delete("rect")
            
            if self.rect_placement_mode:
                # In placement mode, maintain fixed size from the DA image
                canvas_width = int(self.rect_width * self.scale_factor)
                canvas_height = int(self.rect_height * self.scale_factor)
                
                # Move the rectangle with the mouse, keeping same size
                half_width = canvas_width // 2
                half_height = canvas_height // 2
                rect_x = event.x - half_width
                rect_y = event.y - half_height
                
                self.canvas.create_rectangle(
                    rect_x, rect_y,
                    rect_x + canvas_width, rect_y + canvas_height,
                    outline="red", width=2, tags="rect"
                )
            else:
                # Normal drawing mode, size changes with drag
                self.canvas.create_rectangle(
                    self.start_x, self.start_y, event.x, event.y, 
                    outline="red", width=2, tags="rect"
                )
    
    def on_mouse_up(self, event):
        if self.drawing:
            self.drawing = False
            
            if self.rect_placement_mode:
                # In placement mode, get the final position of the fixed-size rectangle
                canvas_width = int(self.rect_width * self.scale_factor)
                canvas_height = int(self.rect_height * self.scale_factor)
                
                # Calculate final rectangle position
                half_width = canvas_width // 2
                half_height = canvas_height // 2
                rect_x = event.x - half_width
                rect_y = event.y - half_height
                
                x1 = rect_x
                y1 = rect_y
                x2 = rect_x + canvas_width
                y2 = rect_y + canvas_height
            else:
                # Normal drawing mode
                x1, y1 = self.start_x, self.start_y
                x2, y2 = event.x, event.y
                
                # Make sure x1,y1 is top-left and x2,y2 is bottom-right
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
            
            # Convert canvas coordinates to original image coordinates
            # First, adjust for the image offset in the canvas
            x1 -= self.img_x_offset
            y1 -= self.img_y_offset
            x2 -= self.img_x_offset
            y2 -= self.img_y_offset
            
            # Then convert to original image coordinates using scale factor
            orig_x1 = int(x1 / self.scale_factor)
            orig_y1 = int(y1 / self.scale_factor)
            orig_x2 = int(x2 / self.scale_factor)
            orig_y2 = int(y2 / self.scale_factor)
            
            # Check boundaries to ensure we're within the image
            img_h, img_w = self.current_img_rgb.shape[:2]
            
            # For placement mode, we need to preserve the exact rectangle size
            if self.rect_placement_mode:
                # First check if we're going to clip the rectangle
                if orig_x1 < 0 or orig_y1 < 0 or orig_x2 >= img_w or orig_y2 >= img_h:
                    # We need to adjust the position but keep the size
                    
                    # Calculate the desired width and height
                    width = orig_x2 - orig_x1
                    height = orig_y2 - orig_y1
                    
                    # Adjust x position if needed
                    if orig_x1 < 0:
                        orig_x1 = 0
                        orig_x2 = width  # Preserve width
                    elif orig_x2 >= img_w:
                        orig_x2 = img_w - 1
                        orig_x1 = orig_x2 - width  # Preserve width
                        
                    # If still out of bounds, further adjustment
                    if orig_x1 < 0:
                        orig_x1 = 0
                        # We'll need to clip the width in this case
                        
                    # Adjust y position if needed
                    if orig_y1 < 0:
                        orig_y1 = 0
                        orig_y2 = height  # Preserve height
                    elif orig_y2 >= img_h:
                        orig_y2 = img_h - 1
                        orig_y1 = orig_y2 - height  # Preserve height
                        
                    # If still out of bounds, further adjustment
                    if orig_y1 < 0:
                        orig_y1 = 0
                        # We'll need to clip the height in this case
                    
                    # Final check to ensure we don't exceed image boundaries
                    orig_x2 = min(orig_x1 + width, img_w - 1)
                    orig_y2 = min(orig_y1 + height, img_h - 1)
            else:
                # For drawing mode, we can just clip to the image boundaries
                orig_x1 = max(0, min(orig_x1, img_w - 1))
                orig_y1 = max(0, min(orig_y1, img_h - 1))
                orig_x2 = max(0, min(orig_x2, img_w - 1))
                orig_y2 = max(0, min(orig_y2, img_h - 1))
            
            # Store rectangle for current image
            self.rectangles[self.current_img_idx] = (orig_x1, orig_y1, orig_x2, orig_y2)
            
            # If this was the DA image, calculate and store the rectangle size
            if self.current_img_idx == 0:
                self.rect_width = orig_x2 - orig_x1
                self.rect_height = orig_y2 - orig_y1
            
            # Redraw the rectangle with proper boundaries
            x1 = int(orig_x1 * self.scale_factor) + self.img_x_offset
            y1 = int(orig_y1 * self.scale_factor) + self.img_y_offset
            x2 = int(orig_x2 * self.scale_factor) + self.img_x_offset
            y2 = int(orig_y2 * self.scale_factor) + self.img_y_offset
            self.canvas.delete("rect")
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="rect")
            
            # Reset placement mode after drawing
            if self.rect_placement_mode:
                self.rect_placement_mode = False
            
            # Update status
            self.status_var.set(f"Rectangle set for {self.source_var.get()}: "
                              f"({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2}), "
                              f"size: {orig_x2-orig_x1}x{orig_y2-orig_y1} pixels")
            
            # Automatically switch to next source after drawing rectangle
            current_source = self.source_var.get()
            if current_source == "Digital (DA)" and self.rectangles[1] is None:
                # Switch to Negative 1
                self.current_img_idx = 1
                self.source_var.set("Negative 1")
                # Set rectangle placement mode
                x1, y1, x2, y2 = self.rectangles[0]
                self.rect_width = x2 - x1
                self.rect_height = y2 - y1
                self.rect_placement_mode = True
                self.status_var.set("Position the rectangle for Negative 1 using the same size as DA image")
                self.load_current_image()
            elif current_source == "Negative 1" and self.rectangles[2] is None:
                # Switch to Negative 2
                self.current_img_idx = 2
                self.source_var.set("Negative 2")
                # Set rectangle placement mode
                x1, y1, x2, y2 = self.rectangles[0]
                self.rect_width = x2 - x1
                self.rect_height = y2 - y1
                self.rect_placement_mode = True
                self.status_var.set("Position the rectangle for Negative 2 using the same size as DA image")
                self.load_current_image()
    
    def clear_rectangle(self):
        self.canvas.delete("rect")
        self.rectangles[self.current_img_idx] = None
        self.status_var.set(f"Rectangle cleared for {self.source_var.get()}")
    
    def update_raw_data_display(self, event=None):
        """Update the raw data display based on the selected source"""
        if self.current_pixel_data is None:
            self.raw_data_text.delete(1.0, tk.END)
            self.raw_data_text.insert(tk.END, "No pixel data available. Extract pixels first.")
            return
            
        selected_source = self.raw_data_source_var.get()
        if selected_source not in self.current_pixel_data:
            self.raw_data_text.delete(1.0, tk.END)
            self.raw_data_text.insert(tk.END, f"No data available for {selected_source}")
            return
        
        # Get the image data for the selected source
        img_data = self.current_pixel_data[selected_source]
        
        # Generate CSV data
        csv_data = self.generate_csv_data(img_data, selected_source)
        
        # Display in the text widget
        self.raw_data_text.delete(1.0, tk.END)
        self.raw_data_text.insert(tk.END, csv_data)
    
    def generate_csv_data(self, img_data, source_name):
        """Generate CSV formatted data from the image using unique RGB pixels"""
        # Get dimensions
        height, width = img_data.shape[:2]
        
        # Determine the picture ID
        if source_name == "Digital (DA)":
            pic_id = "DA"
        elif source_name == "Negative 1":
            pic_id = "N1"
        elif source_name == "Negative 2":
            pic_id = "N2"
        else:
            pic_id = "UK"  # Unknown
        
        # If we have unique pixels for this source, use those
        if hasattr(self, 'unique_pixels_by_source') and source_name in self.unique_pixels_by_source:
            pixel_positions = self.unique_pixels_by_source[source_name]
            print(f"Using {len(pixel_positions)} pre-selected unique pixels for {source_name}")
        elif hasattr(self, 'selected_pixels') and self.selected_pixels:
            # Fall back to the selected_pixels list
            pixel_positions = self.selected_pixels
            print(f"Using {len(pixel_positions)} general selected pixels for {source_name}")
        else:
            # No pixels selected yet, this shouldn't happen after extract_pixels
            print(f"Warning: No pixels selected for {source_name}")
            pixel_positions = []
        
        # Start with CSV header
        csv_lines = ["pic_id,channel,x_rect,y_rect,pixel_value"]
        
        # Generate data for the selected pixels
        valid_pixel_count = 0
        for y, x in pixel_positions:
            # Make sure the coordinates are within bounds
            if y < height and x < width:
                valid_pixel_count += 1
                pixel = img_data[y, x]
                for c_idx, channel in enumerate(['R', 'G', 'B']):
                    pixel_value = int(pixel[c_idx])
                    csv_lines.append(f"{pic_id},{channel},{x},{y},{pixel_value}")
        
        # Debug info
        if valid_pixel_count < len(pixel_positions):
            print(f"Warning: Only {valid_pixel_count} out of {len(pixel_positions)} pixels were valid for {source_name}")
            print(f"Image dimensions: {width}x{height}")
        
        # Join into a single string
        return "\n".join(csv_lines)
    
    def copy_raw_data_to_clipboard(self):
        """Copy the current raw data to clipboard"""
        if not hasattr(self, 'raw_data_text'):
            return
            
        data = self.raw_data_text.get(1.0, tk.END)
        if data:
            self.root.clipboard_clear()
            self.root.clipboard_append(data)
            self.status_var.set("Raw data copied to clipboard.")
        else:
            self.status_var.set("No data to copy.")
            
    def extract_pixels(self):
        # Check if rectangles are defined for all images
        missing = [i for i in range(3) if self.rectangles[i] is None]
        if missing:
            missing_sources = [["Digital (DA)", "Negative 1", "Negative 2"][i] for i in missing]
            self.status_var.set(f"Missing rectangles for: {', '.join(missing_sources)}")
            return
        
        # Extract pixel values from each image
        pixel_data = {}
        
        # First check all ROI dimensions to see if there are large size discrepancies
        roi_dimensions = []
        for idx, rect in self.rectangles.items():
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
            
            if max_width_diff > 10 or max_height_diff > 10:
                print(f"Warning: Large size discrepancy between ROIs. Width diff: {max_width_diff}, Height diff: {max_height_diff}")
        
        # Now extract the ROIs
        for idx, rect in self.rectangles.items():
            try:
                # Load the original image
                img_path = self.image_paths[idx]
                img_rgb = self.load_tiff_image(img_path)
                
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
                self.status_var.set(f"Error extracting pixels from {img_path}: {str(e)}")
                print(f"Error extracting pixels from {img_path}: {str(e)}")
                return
        
        # Store the current pixel data for saving later
        self.current_pixel_data = pixel_data
        
        # Analyze color diversity in each ROI and find unique pixels
        unique_rgb_by_source = {}
        min_unique_count = float('inf')
        for source, roi in pixel_data.items():
            # Flatten the image and count unique RGB values
            flattened = roi.reshape(-1, roi.shape[2])
            # Store unique RGB values as a set of tuples (for hashability)
            unique_rgb = {tuple(pixel) for pixel in flattened}
            unique_rgb_by_source[source] = unique_rgb
            unique_count = len(unique_rgb)
            min_unique_count = min(min_unique_count, unique_count)
            print(f"{source}: Found {unique_count} unique RGB values in ROI of {flattened.shape[0]} pixels")
        
        # Determine how many unique pixels to use (based on the minimum across all sources)
        k = min(10, min_unique_count)  # Cap at 10 or the minimum available
        print(f"Will select {k} unique pixels based on the minimum available across all images")
        
        # Select unique pixels from each image
        # We'll build positions inside the ROI, not the full image
        self.unique_pixels_by_source = {}
        for source, roi in pixel_data.items():
            # Get the set of unique RGB values for this source
            unique_rgb = unique_rgb_by_source[source]
            # Convert to list for indexing
            unique_list = list(unique_rgb)
            
            # If we have more unique pixels than k, randomly select k of them
            if len(unique_list) > k:
                # Shuffle the unique values
                np.random.shuffle(unique_list)
                selected_rgb = unique_list[:k]
            else:
                selected_rgb = unique_list
            
            # Find the positions of these RGB values in the ROI
            positions = []
            height, width = roi.shape[:2]
            for rgb in selected_rgb:
                found = False
                # Search for this RGB value in the ROI
                for y in range(height):
                    if found:
                        break
                    for x in range(width):
                        if tuple(roi[y, x]) == rgb:
                            positions.append((y, x))
                            found = True
                            break
            
            self.unique_pixels_by_source[source] = positions
            print(f"Selected {len(positions)} unique pixel positions for {source}")
            
        # Update the selected_pixels list to use the Digital (DA) positions as reference
        self.selected_pixels = self.unique_pixels_by_source.get("Digital (DA)", [])
        if not self.selected_pixels and self.unique_pixels_by_source:
            # If no Digital (DA) pixels, use the first available source
            first_source = next(iter(self.unique_pixels_by_source.keys()))
            self.selected_pixels = self.unique_pixels_by_source[first_source]
            
        # Display pixel statistics
        self.display_pixel_stats(pixel_data)
        
        # Display waveforms
        self.display_waveforms(pixel_data)
        
        # Update raw data display
        self.update_raw_data_display()
        
        self.status_var.set(f"Pixel data extracted and displayed. Using {len(self.selected_pixels)} unique pixels.")
    
    def create_waveform(self, image_data, ax, title):
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
    
    def display_waveforms(self, pixel_data):
        """Display waveforms for each image region"""
        # Clear previous plots
        self.waveform_fig.clear()
        self.waveform_fig.patch.set_facecolor('#f0f0f0')  # Set a light gray background for the figure
        
        # Increase figure size for better resolution and clarity
        self.waveform_fig.set_size_inches(16, 10)
        
        # Add a bit more space between subplots
        self.waveform_fig.subplots_adjust(hspace=0.3)
        
        # Create subplots for each image with increased height
        axs = self.waveform_fig.subplots(3, 1, sharex=True)
        
        # Define the sources
        sources = ["Digital (DA)", "Negative 1", "Negative 2"]
        
        # Create a waveform for each source
        for i, source in enumerate(sources):
            if source in pixel_data:
                self.create_waveform(pixel_data[source], axs[i], f"Waveform - {source}")
                # Set background color for each subplot
                axs[i].set_facecolor('#ffffff')  # White background
            else:
                axs[i].text(0.5, 0.5, f"No data for {source}", 
                         ha='center', va='center', fontsize=14, fontname='Arial', fontweight='bold')
                axs[i].set_facecolor('#f8f8f8')  # Light gray background for empty plots
        
        # Add a general information text about waveforms with nicer formatting
        self.waveform_fig.text(0.01, 0.01, 
                            "Waveforms show brightness distribution across the image width.\n"
                            "X-axis: Position from left to right.  Y-axis: Brightness from dark (bottom) to bright (top).\n"
                            "RGB channels are overlaid with their respective colors. Vertical lines mark 25%, 50%, and 75% positions.", 
                            fontsize=10, fontname='Arial', color='#444444')
        
        # Add better spacing between subplots
        self.waveform_fig.tight_layout(pad=3.0)
        self.waveform_canvas.draw()
    
    def display_pixel_stats(self, pixel_data):
        # Clear previous plots
        self.fig.clear()
        
        # Get the number of sources
        num_sources = len(pixel_data)
        
        # Create subplots for each source
        axs = self.fig.subplots(num_sources, 1, sharex=True)
        if num_sources == 1:  # Handle case with single subplot
            axs = [axs]
        
        # Increase figure size
        self.fig.set_size_inches(15, 8)  # More height for vertical layout
        
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
        self.fig.tight_layout(pad=2.0)
        self.canvas_fig.draw()
    
    def save_results(self):
        if self.current_pixel_data is None:
            self.status_var.set("No data to save. Extract pixels first.")
            return
            
        try:
            # Ask user for the directory to save data
            save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
            if not save_dir:  # User canceled
                return
                
            # Create timestamp for filenames
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"pixel_analysis_set{self.current_set_id}_{timestamp}"
            
            # Save histograms as an image
            hist_file = os.path.join(save_dir, f"{base_filename}_histograms.png")
            self.fig.savefig(hist_file, dpi=150, bbox_inches='tight')
            
            # Save waveforms as an image
            waveform_file = os.path.join(save_dir, f"{base_filename}_waveforms.png")
            self.waveform_fig.savefig(waveform_file, dpi=150, bbox_inches='tight')
            
            # Save ROI images and raw CSV data
            for source, roi in self.current_pixel_data.items():
                # Convert source name to a safe filename
                source_name = source.replace(" ", "_").replace("(", "").replace(")", "")
                
                # Save ROI as PNG
                roi_file = os.path.join(save_dir, f"{base_filename}_{source_name}_roi.png")
                pil_roi = Image.fromarray(roi)
                pil_roi.save(roi_file)
                
                # Save raw data as CSV
                csv_data = self.generate_csv_data(roi, source)
                csv_file = os.path.join(save_dir, f"{base_filename}_{source_name}_raw_data.csv")
                with open(csv_file, 'w') as f:
                    f.write(csv_data)
            
            # Save rectangle coordinates
            coords_file = os.path.join(save_dir, f"{base_filename}_coordinates.json")
            coords_data = {
                "set_id": self.current_set_id,
                "image_paths": self.image_paths,
                "rectangles": {
                    "Digital (DA)": self.rectangles[0],
                    "Negative 1": self.rectangles[1],
                    "Negative 2": self.rectangles[2]
                }
            }
            with open(coords_file, 'w') as f:
                json.dump(coords_data, f, indent=2)
            
            self.status_var.set(f"Results saved to {save_dir}")
            
        except Exception as e:
            self.status_var.set(f"Error saving results: {str(e)}")
            print(f"Error: {str(e)}")
    
    def on_frame_configure(self, event):
        # Update the scroll region when the inner frame changes size
        self.canvas_main.configure(scrollregion=self.canvas_main.bbox("all"))
    
    def on_canvas_configure(self, event):
        # Update the inner frame's width to fill the canvas
        canvas_width = event.width
        self.canvas_main.itemconfig(self.canvas_window, width=canvas_width)
    
    def on_canvas_resize(self, event):
        # When the canvas is resized, reload the current image to center it properly
        if hasattr(self, 'current_img_rgb') and self.current_img_rgb is not None:
            self.load_current_image()


def main():
    # Load image mappings
    with open('data/prepro/manual_image_mapping.json', 'r') as f:
        image_mappings = json.load(f)
    
    # Create GUI
    root = tk.Tk()
    app = RectangleDrawingApp(root, image_mappings)
    
    # Configure window to resize with content
    root.update()
    
    # Start the event loop
    root.mainloop()

if __name__ == "__main__":
    main()
