import tkinter as tk
from tkinter import ttk, filedialog
import os
import json
import datetime
import platform
import numpy as np

from ui_components import setup_ui
from image_processor import load_current_set, load_current_image
from data_handler import extract_pixels, generate_combined_csv_data, process_and_save_data

class RectangleDrawingApp:
    def __init__(self, root, image_mappings, num_sets=None, root_directory='.', data_processing_path=None, multiple_rectangles=False):
        self.root = root
        self.root.title("Rectangle Drawing Tool")
        self.root.geometry("1600x1000")
        
        self.root_directory = root_directory
        self.data_processing_path = data_processing_path
        self.multiple_rectangles = multiple_rectangles
        
        self.maximize_window()
        
        # Use the previously createsd mappings to load all mappings if num_sets is None
        # or load only the first num_sets mappings if it has a value
        if num_sets is None:
            self.image_mappings = {int(k): v for k, v in image_mappings.items()}
        else:
            self.image_mappings = {int(k): v for k, v in image_mappings.items() if int(k) < num_sets}
        self.current_set_id = 0
        self.current_img_idx = 0  # 0=da_tiff, 1=source1, 2=source2
        
        # Rectangle coordinates for each image in the current set
        # When multiple_rectangles=False: {0: (x1,y1,x2,y2), 1: (x1,y1,x2,y2), 2: (x1,y1,x2,y2)}
        # When multiple_rectangles=True: {0: [(x1,y1,x2,y2), ...], 1: [(x1,y1,x2,y2), ...], 2: [(x1,y1,x2,y2), ...]}
        if self.multiple_rectangles:
            self.rectangles = {0: [], 1: [], 2: []}  # Lists of (x1, y1, x2, y2)
        else:
            self.rectangles = {0: None, 1: None, 2: None}  # (x1, y1, x2, y2)
        self.drawing = False
        
        self.rect_width = None
        self.rect_height = None
        self.rect_placement_mode = False
        
        # For multiple rectangle mode
        self.all_rects_drawn_on_da = False
        self.multi_rect_placement_mode = False
        self.rect_offsets = []  # [(dx1, dy1, dx2, dy2), ...] relative to first rectangle
        
        # For improved rectangle positioning
        self.adjusted_rectangles = {1: [], 2: []}  # User adjusted rectangles for each source
        self.expected_positions = {1: [], 2: []}   # Expected positions based on DA rectangles
        self.scale_factors = {1: (1.0, 1.0), 2: (1.0, 1.0)}  # (x_scale, y_scale) for each source
        self.translations = {1: (0, 0), 2: (0, 0)}  # (x_trans, y_trans) for each source
        self.current_rect_idx = 0  # Index of rectangle being positioned
        self.placing_first_rectangle = False  # Flag for placing the first rectangle
        self.is_rectangle_placed = False  # Flag to indicate if the current rectangle is placed
        
        self.scale_factor = 1.0
        
        self.current_pixel_data = None
        
        # Fixed display size for all images
        self.display_width = 800
        self.display_height = 600
        
        setup_ui(self)
        
        load_current_set(self)
    
    def maximize_window(self):
        """Maximize the window based on the detected operating system"""
        os_name = platform.system()
        
        try:
            if os_name == "Windows":
                self.root.state('zoomed')
            elif os_name == "Linux":
                self.root.attributes('-zoomed', True)
            elif os_name == "Darwin":
                # I don't have a MAC to check, but code taken from stackoverflow should work
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                self.root.geometry(f"{screen_width}x{screen_height-80}+0+0")
            else:
                raise ValueError(f"Unsupported OS: {os_name}")
            self.status_var.set(f"Window maximized for {os_name} OS")
        except ValueError:
            raise
        except Exception as e:
            print(f"Could not maximize window: {str(e)}")
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.root.geometry(f"{int(screen_width*0.8)}x{int(screen_height*0.8)}+50+50")
    
    def on_set_changed(self, event):
        try:
            # If we have drawn any rectangles but not all three, don't allow changing sets
            if any(self.rectangles.values()) and not self.all_rectangles_drawn:
                self.set_var.set(str(self.current_set_id))
                self.status_var.set("Please complete drawing rectangles on all images before changing sets.")
                return
                
            self.current_set_id = int(self.set_var.get())
            load_current_set(self)
            
            self.reset_workflow_state()
            
        except ValueError:
            pass
    
    def on_source_changed(self, event):
        # Relevant after drawing+placing the rectangle accross sources
        source = self.source_var.get()
        if source == "Digital (DA)":
            self.current_img_idx = 0
        elif source == "Negative 1":
            self.current_img_idx = 1
        elif source == "Negative 2":
            self.current_img_idx = 2
        
        # Simply get the size of the rectangle to be able to place it
        if self.multiple_rectangles:
            # In multiple rectangle mode, we need to ensure we're in placement mode
            # but only when all rectangles are drawn on the first source
            if self.all_rects_drawn_on_da and not self.pixels_extracted:
                self.multi_rect_placement_mode = True
                self.status_var.set(f"Viewing {source}. You can adjust all rectangle positions if needed.")
        else:
            all_drawn = all(rect is not None for rect in self.rectangles.values())
            if all_drawn and not self.pixels_extracted:
                x1, y1, x2, y2 = self.rectangles[self.current_img_idx]
                self.rect_width = x2 - x1
                self.rect_height = y2 - y1
                self.rect_placement_mode = True
                self.status_var.set(f"Viewing {source}. You can adjust the rectangle position if needed.")
        
        load_current_image(self)
        
        if not self.pixels_extracted:
            self.status_var.set(f"Viewing {source}. You can adjust the rectangle position if needed.")
    
    def update_ui_state(self):
        """Update UI elements based on current workflow state"""
        # Check if all rectangles are drawn
        if self.multiple_rectangles:
            all_drawn = (len(self.rectangles[0]) > 0 and 
                         len(self.rectangles[1]) > 0 and 
                         len(self.rectangles[2]) > 0 and
                         len(self.rectangles[0]) == len(self.rectangles[1]) == len(self.rectangles[2]))
        else:
            all_drawn = all(rect is not None for rect in self.rectangles.values())
        
        self.all_rectangles_drawn = all_drawn
        
        # Get current source
        current_source = self.source_var.get()
        
        # Update Set combobox state
        if self.multiple_rectangles:
            if len(self.rectangles[0]) > 0 and not self.pixels_extracted:
                self.set_combo.config(state="disabled")
            elif not any(len(self.rectangles[i]) > 0 for i in range(3)):
                self.set_combo.config(state="readonly")
            else:
                self.set_combo.config(state="disabled")
        else:
            if self.rectangles[0] is not None and not self.pixels_extracted:
                self.set_combo.config(state="disabled")
            elif not any(self.rectangles.values()):
                self.set_combo.config(state="readonly")
            else:
                self.set_combo.config(state="disabled")
        
        # Update Source combobox state
        if self.multiple_rectangles:
            if len(self.rectangles[0]) > 0 and not self.pixels_extracted:
                self.source_combo.config(state="readonly")
                # Only bind the combo box after all rectangles are drawn on DA
                if self.all_rects_drawn_on_da:
                    self.source_combo.bind("<<ComboboxSelected>>", self.on_source_changed)
            else:
                self.source_combo.config(state="disabled")
        else:
            if all_drawn and not self.pixels_extracted:
                self.source_combo.config(state="readonly")
                self.source_combo.bind("<<ComboboxSelected>>", self.on_source_changed)
            elif self.pixels_extracted:
                self.source_combo.config(state="disabled")
            else:
                self.source_combo.config(state="disabled")
        
        # Update multiple rectangle mode specific buttons
        if self.multiple_rectangles:
            # Undo Previous button (only visible when drawing on first source or not extracted)
            if self.current_img_idx == 0 and len(self.rectangles[0]) > 0 and not self.pixels_extracted:
                self.undo_rect_btn.pack(side=tk.LEFT, padx=5)
            else:
                if hasattr(self, 'undo_rect_btn'):
                    self.undo_rect_btn.pack_forget()
            
            # Always hide Next Source button immediately when on Negative 2
            if current_source == "Negative 2":
                if hasattr(self, 'next_source_btn'):
                    self.next_source_btn.pack_forget()
                # Skip the rest of the Next Source button logic
            else:
                # Next Source button logic for other sources
                show_next_source = False
                
                # Show the Next Source button only when:
                # 1. We're on Digital (DA) and have drawn at least one rectangle, or
                # 2. We're on Negative 1 and have at least positioned the first rectangle
                if not self.pixels_extracted:
                    if current_source == "Digital (DA)" and len(self.rectangles[0]) > 0:
                        show_next_source = True
                    elif current_source == "Negative 1" and (
                        hasattr(self, 'adjusted_rectangles') and 
                        1 in self.adjusted_rectangles and 
                        len(self.adjusted_rectangles[1]) > 0
                    ):
                        show_next_source = True
                
                if show_next_source:
                    self.next_source_btn.pack(side=tk.LEFT, padx=5)
                else:
                    # First, make sure the button is created if it doesn't exist yet
                    if not hasattr(self, 'next_source_btn'):
                        self.next_source_btn = ttk.Button(self.control_panel, text="Next Source", 
                                                       command=self.next_source)
                    self.next_source_btn.pack_forget()
        
        # Extract Pixels button (shows for both modes when appropriate)
        # For Negative 2, show the button as soon as at least one rectangle is positioned
        if ((all_drawn and not self.pixels_extracted) or 
            (current_source == "Negative 2" and 
             self.multiple_rectangles and 
             len(self.rectangles[2]) > 0 and 
             not self.pixels_extracted)):
            self.extract_pixels_btn.pack(side=tk.LEFT, padx=5)
            # If we're in data processing mode, bind the Return key when the Next button is visible
            if hasattr(self, 'data_processing_path') and self.data_processing_path and hasattr(self, 'return_key_handler'):
                self.root.bind('<Return>', self.return_key_handler)
        elif not all_drawn or self.pixels_extracted:
            if hasattr(self, 'extract_pixels_btn'):
                self.extract_pixels_btn.pack_forget()
            # Unbind the Return key when the Next button is not visible
            if hasattr(self, 'data_processing_path') and self.data_processing_path and hasattr(self, 'return_key_handler'):
                self.root.unbind('<Return>')
        
        # Clear Rectangle button
        if self.multiple_rectangles:
            if any(len(self.rectangles[i]) > 0 for i in range(3)):
                self.clear_rect_btn.pack(side=tk.LEFT, padx=5)
            else:
                if hasattr(self, 'clear_rect_btn'):
                    self.clear_rect_btn.pack_forget()
        else:
            if self.rectangles[0] is not None:
                self.clear_rect_btn.pack(side=tk.LEFT, padx=5)
            else:
                if hasattr(self, 'clear_rect_btn'):
                    self.clear_rect_btn.pack_forget()
        
        # Save Results button (only in non-data-processing mode)
        if self.pixels_extracted:
            # Only show save button and results in non-data-processing mode
            if not hasattr(self, 'data_processing_path') or not self.data_processing_path:
                self.save_results_btn.pack(side=tk.LEFT, padx=5)
                self.results_tabs.pack(fill=tk.BOTH, expand=True)
        elif hasattr(self, 'save_results_btn'):
            self.save_results_btn.pack_forget()
            self.results_tabs.pack_forget()
    
    def reset_workflow_state(self):
        """Reset the workflow state to initial"""
        if self.multiple_rectangles:
            self.rectangles = {0: [], 1: [], 2: []}
            self.all_rects_drawn_on_da = False
            self.multi_rect_placement_mode = False
            self.rect_offsets = []
        else:
            self.rectangles = {0: None, 1: None, 2: None}
        
        self.all_rectangles_drawn = False
        self.pixels_extracted = False
        self.rect_placement_mode = False
        
        self.current_img_idx = 0
        self.source_var.set("Digital (DA)")
        
        self.current_pixel_data = None
        
        self.canvas.delete("rect")
        self.canvas.delete("preview_rect")
        
        # Ensure Return key is unbound when resetting
        if hasattr(self, 'data_processing_path') and self.data_processing_path and hasattr(self, 'return_key_handler'):
            self.root.unbind('<Return>')
        
        self.update_ui_state()
        
        if self.multiple_rectangles:
            self.status_var.set("Ready. Draw multiple rectangles on the Digital (DA) image.")
        else:
            self.status_var.set("Ready. Draw rectangle on the Digital (DA) image.")
    
    def on_mouse_down(self, event):
        # Only active after drawing the rectangle --> blocks repositioning after pixels are extracted
        if self.pixels_extracted:
            self.status_var.set("Rectangle repositioning is not allowed after extracting pixels. Use 'Clear Rectangle' to start over.")
            return
            
        if (self.img_x_offset <= event.x < self.img_x_offset + self.tk_img.width() and
            self.img_y_offset <= event.y < self.img_y_offset + self.tk_img.height()):
            self.drawing = True
            self.start_x, self.start_y = event.x, event.y
            
            # Different handling for multiple rectangle mode
            if self.multiple_rectangles:
                # If we're in multi-rectangle placement mode (for sources 1 and 2)
                if self.multi_rect_placement_mode and self.current_img_idx != 0:
                    # For the improved positioning system, don't delete existing rectangles
                    pass
                else:
                    # When drawing new rectangles on source 0, don't delete existing ones
                    pass
            else:
                # Standard mode - delete existing rectangle
                self.canvas.delete("rect")
            
            # If all three rectangles are already drawn and we're not extracting pixels yet,
            # always use placement mode with the existing rectangle size
            if not self.multiple_rectangles:
                all_drawn = all(rect is not None for rect in self.rectangles.values())
                if all_drawn and not self.pixels_extracted:
                    x1, y1, x2, y2 = self.rectangles[self.current_img_idx]
                    self.rect_width = x2 - x1
                    self.rect_height = y2 - y1
                    self.rect_placement_mode = True
            elif self.multi_rect_placement_mode and self.current_img_idx != 0:
                # For improved positioning, we'll draw a preview of the current rectangle
                if self.placing_first_rectangle or not self.is_rectangle_placed:
                    # For the first rectangle, or when placing a new one
                    orig_rect = self.rectangles[0][self.current_rect_idx]
                    width = orig_rect[2] - orig_rect[0]
                    height = orig_rect[3] - orig_rect[1]
                    
                    # Convert to canvas coordinates
                    canvas_width = int(width * self.scale_factor)
                    canvas_height = int(height * self.scale_factor)
                    
                    # Center the rectangle at the mouse position
                    half_width = canvas_width // 2
                    half_height = canvas_height // 2
                    
                    x1 = event.x - half_width
                    y1 = event.y - half_height
                    x2 = x1 + canvas_width
                    y2 = y1 + canvas_height
                    
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        outline="blue", width=2, tags="preview_rect"
                    )
                
                # If the user clicks on rectangle with circle, this means we're adjusting it
                # We'll identify this in mouse_up
            
            # If in placement mode, create the rectangle immediately with fixed size
            if self.rect_placement_mode and not self.multiple_rectangles:
                canvas_width = int(self.rect_width * self.scale_factor)
                canvas_height = int(self.rect_height * self.scale_factor)
                
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
        if self.pixels_extracted:
            return
            
        if self.drawing:
            # Different handling for multiple rectangle mode
            if self.multiple_rectangles:
                if self.multi_rect_placement_mode and self.current_img_idx != 0:
                    # For improved positioning, update the preview rectangle
                    if not self.is_rectangle_placed:
                        orig_rect = self.rectangles[0][self.current_rect_idx]
                        width = orig_rect[2] - orig_rect[0]
                        height = orig_rect[3] - orig_rect[1]
                        
                        # Convert to canvas coordinates
                        canvas_width = int(width * self.scale_factor)
                        canvas_height = int(height * self.scale_factor)
                        
                        # Center at mouse position
                        half_width = canvas_width // 2
                        half_height = canvas_height // 2
                        
                        self.canvas.delete("preview_rect")
                        self.canvas.create_rectangle(
                            event.x - half_width, event.y - half_height,
                            event.x + half_width, event.y + half_height,
                            outline="blue", width=2, tags="preview_rect"
                        )
                else:
                    # For drawing new rectangles on source 0
                    self.canvas.delete("preview_rect")
                    self.canvas.create_rectangle(
                        self.start_x, self.start_y, event.x, event.y, 
                        outline="blue", width=2, tags="preview_rect"
                    )
            else:
                # Standard mode
                self.canvas.delete("rect")
                
                if self.rect_placement_mode:
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
        if self.pixels_extracted:
            return
            
        if self.drawing:
            self.drawing = False
            
            # Handle multiple rectangle mode
            if self.multiple_rectangles:
                if self.multi_rect_placement_mode and self.current_img_idx != 0:
                    # For improved positioning, handle the rectangle placement
                    self._handle_rectangle_placement(event.x, event.y)
                else:
                    # For drawing new rectangles on source 0
                    self._save_new_rectangle(event.x, event.y)
            else:
                # Standard single rectangle mode
                if self.rect_placement_mode:
                    # In placement mode, get the final position of the fixed-size rectangle
                    canvas_width = int(self.rect_width * self.scale_factor)
                    canvas_height = int(self.rect_height * self.scale_factor)
                    
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
                    
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                
                x1 -= self.img_x_offset
                y1 -= self.img_y_offset
                x2 -= self.img_x_offset
                y2 -= self.img_y_offset
                
                orig_x1 = int(x1 / self.scale_factor)
                orig_y1 = int(y1 / self.scale_factor)
                orig_x2 = int(x2 / self.scale_factor)
                orig_y2 = int(y2 / self.scale_factor)
                
                img_h, img_w = self.current_img_rgb.shape[:2]
                
                # For placement mode, we need to preserve the exact rectangle size
                if self.rect_placement_mode:
                    if orig_x1 < 0 or orig_y1 < 0 or orig_x2 >= img_w or orig_y2 >= img_h:
                        width = orig_x2 - orig_x1
                        height = orig_y2 - orig_y1
                        
                        if orig_x1 < 0:
                            orig_x1 = 0
                            orig_x2 = width
                        elif orig_x2 >= img_w:
                            orig_x2 = img_w - 1
                            orig_x1 = orig_x2 - width
                            
                        if orig_x1 < 0:
                            orig_x1 = 0
                            
                        if orig_y1 < 0:
                            orig_y1 = 0
                            orig_y2 = height
                        elif orig_y2 >= img_h:
                            orig_y2 = img_h - 1
                            orig_y1 = orig_y2 - height
                            
                        if orig_y1 < 0:
                            orig_y1 = 0
                        
                        orig_x2 = min(orig_x1 + width, img_w - 1)
                        orig_y2 = min(orig_y1 + height, img_h - 1)
                else:
                    # For drawing mode, we can just clip to the image boundaries
                    orig_x1 = max(0, min(orig_x1, img_w - 1))
                    orig_y1 = max(0, min(orig_y1, img_h - 1))
                    orig_x2 = max(0, min(orig_x2, img_w - 1))
                    orig_y2 = max(0, min(orig_y2, img_h - 1))
                
                self.rectangles[self.current_img_idx] = (orig_x1, orig_y1, orig_x2, orig_y2)
                
                if self.current_img_idx == 0:
                    # Drawing mode -> DA -> store the rectangle size
                    self.rect_width = orig_x2 - orig_x1
                    self.rect_height = orig_y2 - orig_y1
                
                # Redraw/clip the rectangle with proper boundaries
                x1 = int(orig_x1 * self.scale_factor) + self.img_x_offset
                y1 = int(orig_y1 * self.scale_factor) + self.img_y_offset
                x2 = int(orig_x2 * self.scale_factor) + self.img_x_offset
                y2 = int(orig_y2 * self.scale_factor) + self.img_y_offset
                self.canvas.delete("rect")
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="rect")
                
                all_drawn = all(rect is not None for rect in self.rectangles.values())
                
                if self.rect_placement_mode and (not all_drawn or self.pixels_extracted):
                    self.rect_placement_mode = False
                
                self.status_var.set(f"Rectangle set for {self.source_var.get()}: "
                                  f"({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2}), "
                                  f"size: {orig_x2-orig_x1}x{orig_y2-orig_y1} pixels")
                
                # Automatically switch to next source after drawing rectangle
                current_source = self.source_var.get()
                if current_source == "Digital (DA)" and self.rectangles[1] is None:
                    self.current_img_idx = 1
                    self.source_var.set("Negative 1")
                    # Set rectangle placement mode
                    x1, y1, x2, y2 = self.rectangles[0]
                    self.rect_width = x2 - x1
                    self.rect_height = y2 - y1
                    self.rect_placement_mode = True
                    self.status_var.set("Position the rectangle for Negative 1 using the same size as DA image")
                    load_current_image(self)
                elif current_source == "Negative 1" and self.rectangles[2] is None:
                    self.current_img_idx = 2
                    self.source_var.set("Negative 2")
                    x1, y1, x2, y2 = self.rectangles[0]
                    self.rect_width = x2 - x1
                    self.rect_height = y2 - y1
                    self.rect_placement_mode = True
                    self.status_var.set("Position the rectangle for Negative 2 using the same size as DA image")
                    load_current_image(self)
                
                self.update_ui_state()
                
                # If all three rectangles are drawn, show guidance
                if all_drawn and not self.pixels_extracted:
                    self.status_var.set("All rectangles placed. You can now select different sources to adjust positions if needed, or extract pixels.")
    
    def _handle_rectangle_placement(self, event_x, event_y):
        """Handle rectangle placement for the improved rectangle positioning system"""
        self.canvas.delete("preview_rect")
        
        # Get the original rectangle from source 0
        if self.current_rect_idx >= len(self.rectangles[0]):
            return
            
        orig_rect = self.rectangles[0][self.current_rect_idx]
        width = orig_rect[2] - orig_rect[0]
        height = orig_rect[3] - orig_rect[1]
        
        # Convert mouse position to image coordinates
        mx = event_x - self.img_x_offset
        my = event_y - self.img_y_offset
        
        img_x = int(mx / self.scale_factor)
        img_y = int(my / self.scale_factor)
        
        # Calculate rectangle coordinates centered at mouse position
        half_width = width / 2
        half_height = height / 2
        
        x1 = img_x - half_width
        y1 = img_y - half_height
        x2 = img_x + half_width
        y2 = img_y + half_height
        
        # Ensure coordinates are within image boundaries
        img_h, img_w = self.current_img_rgb.shape[:2]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        
        # Convert coordinates to integers for slice indexing
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Add to adjusted rectangles
        if self.current_rect_idx < len(self.adjusted_rectangles[self.current_img_idx]):
            # Update existing one
            self.adjusted_rectangles[self.current_img_idx][self.current_rect_idx] = (x1, y1, x2, y2)
        else:
            # Add new one
            self.adjusted_rectangles[self.current_img_idx].append((x1, y1, x2, y2))
        
        # Also update the main rectangles for this source
        if self.current_rect_idx < len(self.rectangles[self.current_img_idx]):
            # Update existing
            self.rectangles[self.current_img_idx][self.current_rect_idx] = (x1, y1, x2, y2)
        else:
            # Add new
            self.rectangles[self.current_img_idx].append((x1, y1, x2, y2))
        
        # Calculate and show the next rectangle if we have more than one
        self.is_rectangle_placed = True
        self.placing_first_rectangle = False
        
        # This is critical: force the Next Source button to be visible after placing the first rectangle
        # BUT NEVER show it for Negative Source 2
        if len(self.adjusted_rectangles[self.current_img_idx]) == 1 and self.current_img_idx != 2:
            if hasattr(self, 'next_source_btn'):
                self.next_source_btn.pack(side=tk.LEFT, padx=5)
        
        # For Negative 2, make sure the Extract Pixels button is visible as soon as one rectangle is placed
        if self.current_img_idx == 2 and not self.pixels_extracted:
            # Show the Extract Pixels button
            if hasattr(self, 'extract_pixels_btn'):
                self.extract_pixels_btn.pack(side=tk.LEFT, padx=5)
        
        # If we have at least one rectangle placed, compute transformation and preview next ones
        if len(self.adjusted_rectangles[self.current_img_idx]) >= 1:
            # Compute transformation for showing the next rectangle
            self._compute_and_apply_transformation(self.current_img_idx)
            
            # Increment to next rectangle if there are more to place
            if len(self.adjusted_rectangles[self.current_img_idx]) < len(self.rectangles[0]):
                self.current_rect_idx = len(self.adjusted_rectangles[self.current_img_idx])
                self.is_rectangle_placed = False
                
                # Show guidance message for next rectangle
                next_rect_idx = self.current_rect_idx + 1
                
                # Different guidance message for Negative Source 2
                if self.current_img_idx == 2:
                    self.status_var.set(f"Positioned rectangle {self.current_rect_idx + 1} of {len(self.rectangles[0])}. "
                                       f"The blue rectangle shows the calculated position for rectangle {next_rect_idx}. "
                                       f"Adjust its position if needed or press 'Next' to extract pixels when finished.")
                else:
                    self.status_var.set(f"Positioned rectangle {self.current_rect_idx + 1} of {len(self.rectangles[0])}. "
                                       f"The blue rectangle shows the calculated position for rectangle {next_rect_idx}. "
                                       f"Adjust its position if needed or press 'Next Source' to accept all calculated positions.")
                
                # Draw preview of the next rectangle if available
                if self.current_rect_idx < len(self.rectangles[self.current_img_idx]):
                    next_rect = self.rectangles[self.current_img_idx][self.current_rect_idx]
                    x1, y1, x2, y2 = next_rect
                    
                    # Convert to canvas coordinates
                    canvas_x1 = int(x1 * self.scale_factor) + self.img_x_offset
                    canvas_y1 = int(y1 * self.scale_factor) + self.img_y_offset
                    canvas_x2 = int(x2 * self.scale_factor) + self.img_x_offset
                    canvas_y2 = int(y2 * self.scale_factor) + self.img_y_offset
                    
                    # Draw the preview rectangle
                    self.canvas.create_rectangle(
                        canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                        outline="blue", width=2, tags="preview_rect"
                    )
                    
                    # Add a circle in the center to indicate it's the next one to position
                    center_x = (canvas_x1 + canvas_x2) / 2
                    center_y = (canvas_y1 + canvas_y2) / 2
                    circle_radius = 5
                    self.canvas.create_oval(
                        center_x - circle_radius, center_y - circle_radius,
                        center_x + circle_radius, center_y + circle_radius,
                        fill="blue", outline="white", tags="preview_rect"
                    )
            else:
                # All rectangles positioned
                if self.current_img_idx == 2:
                    self.status_var.set(f"All rectangles positioned. Press 'Next' to extract pixels.")
                else:
                    self.status_var.set(f"All rectangles positioned. Press 'Next Source' to continue.")
        else:
            # Just the first rectangle placed
            self.current_rect_idx = len(self.adjusted_rectangles[self.current_img_idx])
            self.is_rectangle_placed = False
            
            # If there are more rectangles to place
            if self.current_rect_idx < len(self.rectangles[0]):
                # Calculate and show preview of the next rectangle
                if len(self.adjusted_rectangles[self.current_img_idx]) == 1:
                    # With just one rectangle, we only have translation (no scaling yet)
                    # Use the expected position from DA plus the translation
                    expected_x, expected_y = self.expected_positions[self.current_img_idx][0]
                    x1, y1, x2, y2 = self.adjusted_rectangles[self.current_img_idx][0]
                    placed_x = (x1 + x2) / 2
                    placed_y = (y1 + y2) / 2
                    
                    dx = placed_x - expected_x
                    dy = placed_y - expected_y
                    
                    # Preview the position of the next rectangle
                    if self.current_rect_idx < len(self.rectangles[0]):
                        # Get the expected position of the next rectangle from DA
                        orig_rect = self.rectangles[0][self.current_rect_idx]
                        nx1, ny1, nx2, ny2 = orig_rect
                        
                        # Get center point and dimensions of the original rectangle
                        orig_center_x = (nx1 + nx2) / 2
                        orig_center_y = (ny1 + ny2) / 2
                        orig_width = nx2 - nx1
                        orig_height = ny2 - ny1
                        
                        # Apply translation to center (no scaling)
                        new_center_x = orig_center_x + dx
                        new_center_y = orig_center_y + dy
                        
                        # Calculate new corners using original width and height
                        nx1 = new_center_x - orig_width / 2
                        ny1 = new_center_y - orig_height / 2
                        nx2 = new_center_x + orig_width / 2
                        ny2 = new_center_y + orig_height / 2
                        
                        # Ensure coordinates are within image boundaries
                        nx1 = max(0, min(nx1, img_w - 1))
                        ny1 = max(0, min(ny1, img_h - 1))
                        nx2 = max(0, min(nx2, img_w - 1))
                        ny2 = max(0, min(ny2, img_h - 1))
                        
                        # Convert to integers for slice indexing
                        nx1, ny1, nx2, ny2 = int(nx1), int(ny1), int(nx2), int(ny2)
                        
                        # Update the rectangle in the current source
                        if self.current_rect_idx < len(self.rectangles[self.current_img_idx]):
                            self.rectangles[self.current_img_idx][self.current_rect_idx] = (nx1, ny1, nx2, ny2)
                        else:
                            self.rectangles[self.current_img_idx].append((nx1, ny1, nx2, ny2))
                        
                        # Convert to canvas coordinates
                        canvas_x1 = int(nx1 * self.scale_factor) + self.img_x_offset
                        canvas_y1 = int(ny1 * self.scale_factor) + self.img_y_offset
                        canvas_x2 = int(nx2 * self.scale_factor) + self.img_x_offset
                        canvas_y2 = int(ny2 * self.scale_factor) + self.img_y_offset
                        
                        # Draw the preview rectangle
                        self.canvas.create_rectangle(
                            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                            outline="blue", width=2, tags="preview_rect"
                        )
                        
                        # Add a circle in the center to indicate it's the next one to position
                        center_x = (canvas_x1 + canvas_x2) / 2
                        center_y = (canvas_y1 + canvas_y2) / 2
                        circle_radius = 5
                        self.canvas.create_oval(
                            center_x - circle_radius, center_y - circle_radius,
                            center_x + circle_radius, center_y + circle_radius,
                            fill="blue", outline="white", tags="preview_rect"
                        )
                
                # Different status message for Negative Source 2
                if self.current_img_idx == 2:
                    self.status_var.set(f"Positioned rectangle 1 of {len(self.rectangles[0])}. The blue rectangle shows the calculated position for rectangle 2. "
                                       f"Adjust its position if needed or press 'Next' to extract pixels when finished.")
                else:
                    self.status_var.set(f"Positioned rectangle 1 of {len(self.rectangles[0])}. The blue rectangle shows the calculated position for rectangle 2. "
                                       f"Adjust its position if needed or press 'Next Source' to accept all calculated positions.")
            else:
                # All rectangles positioned
                if self.current_img_idx == 2:
                    self.status_var.set(f"All rectangles positioned. Press 'Next' to extract pixels.")
                else:
                    self.status_var.set(f"All rectangles positioned. Press 'Next Source' to continue.")
        
        # Redraw the canvas to update all rectangles
        load_current_image(self)
        
        # Make sure Next Source button is visible after placing any rectangle
        # BUT NEVER for Negative Source 2
        if len(self.adjusted_rectangles[self.current_img_idx]) > 0 and not self.pixels_extracted and self.current_img_idx != 2:
            if hasattr(self, 'next_source_btn'):
                self.next_source_btn.pack(side=tk.LEFT, padx=5)
        elif self.current_img_idx == 2:
            # Always hide the Next Source button when on Negative Source 2
            if hasattr(self, 'next_source_btn'):
                self.next_source_btn.pack_forget()
        
        # Update the UI state to ensure Extract Pixels button is shown when needed
        self.update_ui_state()
    
    def _save_new_rectangle(self, event_x, event_y):
        """Save a new rectangle when drawing in multiple rectangle mode on source 0"""
        self.canvas.delete("preview_rect")
        
        x1, y1 = self.start_x, self.start_y
        x2, y2 = event_x, event_y
        
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        x1 -= self.img_x_offset
        y1 -= self.img_y_offset
        x2 -= self.img_x_offset
        y2 -= self.img_y_offset
        
        orig_x1 = int(x1 / self.scale_factor)
        orig_y1 = int(y1 / self.scale_factor)
        orig_x2 = int(x2 / self.scale_factor)
        orig_y2 = int(y2 / self.scale_factor)
        
        img_h, img_w = self.current_img_rgb.shape[:2]
        
        # Clip to image boundaries
        orig_x1 = max(0, min(orig_x1, img_w - 1))
        orig_y1 = max(0, min(orig_y1, img_h - 1))
        orig_x2 = max(0, min(orig_x2, img_w - 1))
        orig_y2 = max(0, min(orig_y2, img_h - 1))
        
        # Ensure we have integer coordinates
        orig_x1, orig_y1, orig_x2, orig_y2 = int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)
        
        # Add the new rectangle to the list
        self.rectangles[0].append((orig_x1, orig_y1, orig_x2, orig_y2))
        
        # Draw all rectangles
        self.canvas.delete("rect")
        self._draw_multiple_rectangles()
        
        rect_count = len(self.rectangles[0])
        self.status_var.set(f"Rectangle {rect_count} added to Digital (DA): "
                          f"({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2}), "
                          f"size: {orig_x2-orig_x1}x{orig_y2-orig_y1} pixels")
        
        self.update_ui_state()
    
    def _save_multi_placement_rectangles(self, event_x, event_y):
        """Save all rectangle positions when placing multiple rectangles on sources 1 and 2"""
        # This method is deprecated and replaced by _handle_rectangle_placement
        pass
    
    def clear_rectangle(self):
        """Clear all rectangles and reset the workflow"""
        self.reset_workflow_state()
        
        load_current_image(self)
        
        self.status_var.set("All rectangles cleared. Start again by drawing a rectangle on the Digital (DA) image.")
    
    def update_raw_data_display(self, event=None):
        """Update the raw data display to show data from all three images together"""
        if self.current_pixel_data is None:
            self.raw_data_text.delete(1.0, tk.END)
            self.raw_data_text.insert(tk.END, "No pixel data available. Extract pixels first.")
            return
            
        # We'll combine data from all sources into one display
        all_sources = ["Digital (DA)", "Negative 1", "Negative 2"]
        available_sources = [s for s in all_sources if s in self.current_pixel_data]
        
        if not available_sources:
            self.raw_data_text.delete(1.0, tk.END)
            self.raw_data_text.insert(tk.END, "No data available from any source")
            return
        
        combined_csv_data = generate_combined_csv_data(self, include_stats=True, include_image_id=False)
        
        self.raw_data_text.delete(1.0, tk.END)
        self.raw_data_text.insert(tk.END, combined_csv_data)
    
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
        """Extract pixel data and update the UI state"""
        extract_pixels(self)
        
        self.pixels_extracted = True
        
        self.update_ui_state()
        
        self.status_var.set("Pixels extracted and analysis complete. You can now save the results. Use 'Clear Rectangle' to start over.")
    
    def process_and_save_data(self):
        """Extract pixels and automatically save to the specified data processing file"""
        success = process_and_save_data(self)
        
        if not success:
            # If extraction failed, don't update UI state or clear rectangles
            self.status_var.set(f"Error extracting pixels. Please check the console for details.")
            return
        
        # Only continue if extraction was successful
        self.pixels_extracted = True
        
        # Skip displaying visualizations in data processing mode
        if hasattr(self, 'results_tabs'):
            self.results_tabs.pack_forget()
        
        self.update_ui_state()
        
        # Simply inform that data was processed and allow user to continue with the same set
        self.status_var.set(f"Data processed and saved to {self.data_processing_path}. Press 'Clear Rectangle' to process another region.")
        
        # Automatically clear the rectangle in data processing mode
        if hasattr(self, 'data_processing_path') and self.data_processing_path:
            self.clear_rectangle()
    
    def save_results(self):
        if self.current_pixel_data is None:
            self.status_var.set("No data to save. Extract pixels first.")
            return
            
        try:
            save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
            if not save_dir:
                return
                
            timestamp = datetime.datetime.now().strftime("%d%m_%H%M%S")
            
            subdir = os.path.join(save_dir, f"id_{self.current_set_id}", timestamp)
            os.makedirs(subdir, exist_ok=True)
            
            base_filename = f"pixel_analysis_set{self.current_set_id}"
            
            hist_file = os.path.join(subdir, f"{base_filename}_histograms.png")
            self.fig.savefig(hist_file, dpi=150, bbox_inches='tight')
            
            waveform_file = os.path.join(subdir, f"{base_filename}_waveforms.png")
            self.waveform_fig.savefig(waveform_file, dpi=150, bbox_inches='tight')
            
            for source, roi in self.current_pixel_data.items():
                source_name = source.replace(" ", "_").replace("(", "").replace(")", "")
                
                from PIL import Image
                roi_file_png = os.path.join(subdir, f"{base_filename}_{source_name}_roi.png")
                pil_roi = Image.fromarray(roi)
                pil_roi.save(roi_file_png)
                
                roi_file_tif = os.path.join(subdir, f"{base_filename}_{source_name}_roi.tif")
                pil_roi.save(roi_file_tif)
            
            combined_csv_data = generate_combined_csv_data(self, include_stats=True, include_image_id=False)
            csv_file = os.path.join(subdir, f"{base_filename}_raw_data.csv")
            with open(csv_file, 'w') as f:
                f.write(combined_csv_data)
            
            coords_file = os.path.join(subdir, f"{base_filename}_coordinates.json")
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
            
            self.status_var.set(f"Results saved to {subdir}")
            
        except Exception as e:
            self.status_var.set(f"Error saving results: {str(e)}")
            print(f"Error: {str(e)}")

    def undo_last_rectangle(self):
        """Remove the last drawn rectangle in multiple rectangle mode"""
        if not self.multiple_rectangles:
            return
            
        current_source = self.source_var.get()
        idx = ["Digital (DA)", "Negative 1", "Negative 2"].index(current_source)
        
        if self.rectangles[idx] and len(self.rectangles[idx]) > 0:
            # Remove the last rectangle from the current source
            self.rectangles[idx].pop()
            
            # Also update the rect_offsets if we're in the first source (Digital DA)
            if idx == 0 and len(self.rect_offsets) > 0:
                self.rect_offsets.pop()
                
            # Redraw all rectangles
            self.canvas.delete("rect")
            self._draw_multiple_rectangles()
            
            self.status_var.set(f"Removed last rectangle. {len(self.rectangles[idx])} rectangles remaining on {current_source}.")
            
            # Update the UI state
            self.update_ui_state()
        else:
            self.status_var.set(f"No rectangles to remove on {current_source}.")
    
    def next_source(self):
        """Switch to the next source in multiple rectangle mode"""
        if not self.multiple_rectangles:
            return
            
        current_source = self.source_var.get()
        
        # If we're on Digital (DA)
        if current_source == "Digital (DA)":
            # Check if we have drawn at least one rectangle
            if not self.rectangles[0] or len(self.rectangles[0]) == 0:
                self.status_var.set("Please draw at least one rectangle before moving to the next source.")
                return
            
            # Store the rectangle center points for expected positions
            # We'll use these to calculate the transformation between DA and negative sources
            self.expected_positions[1] = []
            for rect in self.rectangles[0]:
                x1, y1, x2, y2 = rect
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                self.expected_positions[1].append((center_x, center_y))
            
            # Mark that we've drawn all rectangles on DA
            self.all_rects_drawn_on_da = True
            self.multi_rect_placement_mode = True
            self.placing_first_rectangle = True
            self.current_rect_idx = 0
            self.is_rectangle_placed = False
            
            # Reset adjusted rectangles for this source
            self.adjusted_rectangles[1] = []
            
            # Switch to Negative 1
            self.current_img_idx = 1
            self.source_var.set("Negative 1")
            
            # Load the image before setting status
            load_current_image(self)
            self.status_var.set(f"Position the first rectangle on Negative 1. This will calibrate the scale factor.")
            
        # If we're on Negative 1
        elif current_source == "Negative 1":
            # Check if we have positioned at least one rectangle
            if len(self.adjusted_rectangles[1]) == 0:
                self.status_var.set("Please position at least the first rectangle before moving to the next source.")
                return
                
            # For Negative Source 2, we still want to use Digital (DA) as the reference
            # This ensures we have a consistent transformation from the original
            self.expected_positions[2] = []
            for rect in self.rectangles[0]:
                x1, y1, x2, y2 = rect
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                self.expected_positions[2].append((center_x, center_y))
                    
            # Reset for source 2
            self.placing_first_rectangle = True
            self.current_rect_idx = 0
            self.is_rectangle_placed = False
            
            # Reset adjusted rectangles for source 2
            self.adjusted_rectangles[2] = []
                
            # Switch to Negative 2
            self.current_img_idx = 2
            self.source_var.set("Negative 2")
            
            # Explicitly hide the Next Source button when switching to Negative 2
            if hasattr(self, 'next_source_btn'):
                self.next_source_btn.pack_forget()
            
            # Load the image before setting status
            load_current_image(self)
            self.status_var.set(f"Position the first rectangle on Negative 2. This will calibrate the scale factor. When finished, press 'Next' to extract pixels.")
            
        # If we're on Negative 2
        elif current_source == "Negative 2":
            # Check if we've positioned at least the first rectangle on all sources
            if len(self.adjusted_rectangles[2]) == 0:
                self.status_var.set("Please position at least the first rectangle before finalizing.")
                return
                
            # This means we're done with placing rectangles
            self.all_rectangles_drawn = True
            
            # Finalize all rectangles for source 1 if not all were manually placed
            if len(self.adjusted_rectangles[1]) < len(self.rectangles[0]):
                self._compute_and_apply_transformation(1)
                
            # Finalize all rectangles for source 2 if not all were manually placed
            if len(self.adjusted_rectangles[2]) < len(self.rectangles[0]):
                self._compute_and_apply_transformation(2)
                
            self.status_var.set("All rectangles positioned on all sources. You can now press 'Next' to extract pixels.")
        
        # Update UI state
        self.update_ui_state()
        
        # Force Next Source button visibility if we're on Negative 1 with at least one rectangle placed
        # (Don't apply for Negative 2)
        if current_source == "Digital (DA)" and hasattr(self, 'adjusted_rectangles') and 1 in self.adjusted_rectangles:
            if hasattr(self, 'next_source_btn'):
                self.next_source_btn.pack(side=tk.LEFT, padx=5)
                
        # Make sure the Extract Pixels button is shown when on Negative 2 and 
        # at least one rectangle is positioned
        if self.current_img_idx == 2 and len(self.rectangles[2]) > 0 and not self.pixels_extracted:
            if hasattr(self, 'extract_pixels_btn'):
                self.extract_pixels_btn.pack(side=tk.LEFT, padx=5)
    
    def _compute_and_apply_transformation(self, source_idx):
        """Compute and apply transformation from expected to adjusted positions"""
        if len(self.adjusted_rectangles[source_idx]) < 1:
            return
            
        # Get expected and actual centers for the rectangles that have been adjusted
        expected_centers = []
        actual_centers = []
        
        for i, rect in enumerate(self.adjusted_rectangles[source_idx]):
            if i < len(self.expected_positions[source_idx]):
                expected_x, expected_y = self.expected_positions[source_idx][i]
                
                x1, y1, x2, y2 = rect
                actual_x = (x1 + x2) / 2
                actual_y = (y1 + y2) / 2
                
                expected_centers.append((expected_x, expected_y))
                actual_centers.append((actual_x, actual_y))
        
        # Calculate transformation using polynomial fit if we have enough points
        if len(expected_centers) >= 1:
            # For one point, just use translation
            if len(expected_centers) == 1:
                dx = actual_centers[0][0] - expected_centers[0][0]
                dy = actual_centers[0][1] - expected_centers[0][1]
                sx, sy = 1.0, 1.0  # No scaling with just one point
            else:
                # For multiple points, use np.polyfit to get a degree 1 fit (scale + translation)
                expected_x = [p[0] for p in expected_centers]
                expected_y = [p[1] for p in expected_centers]
                actual_x = [p[0] for p in actual_centers]
                actual_y = [p[1] for p in actual_centers]
                
                try:
                    # x = ax + b
                    x_coeffs = np.polyfit(expected_x, actual_x, 1)
                    sx, dx = x_coeffs[0], x_coeffs[1]
                    
                    # y = cy + d
                    y_coeffs = np.polyfit(expected_y, actual_y, 1)
                    sy, dy = y_coeffs[0], y_coeffs[1]
                except Exception as e:
                    print(f"Error in polyfit: {str(e)}")
                    # Fallback to simple translation if polyfit fails
                    dx = actual_centers[0][0] - expected_centers[0][0]
                    dy = actual_centers[0][1] - expected_centers[0][1]
                    sx, sy = 1.0, 1.0
            
            # Store the transformation
            self.scale_factors[source_idx] = (sx, sy)
            self.translations[source_idx] = (dx, dy)
            
            print(f"Transformation for source {source_idx}: Scale={sx:.3f}, {sy:.3f}, Translation={dx:.1f}, {dy:.1f}")
            
            # Apply transformation to all rectangles from DA to create the full set for this source
            transformed_rects = []
            for i, rect in enumerate(self.rectangles[0]):
                if i < len(self.adjusted_rectangles[source_idx]):
                    # Keep manually positioned rectangles
                    # Convert to integers to ensure they work as slice indices
                    x1, y1, x2, y2 = self.adjusted_rectangles[source_idx][i]
                    transformed_rects.append((int(x1), int(y1), int(x2), int(y2)))
                else:
                    # Transform remaining rectangles
                    x1, y1, x2, y2 = rect
                    
                    # Transform centers only, preserving original rectangle size
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    new_center_x = center_x * sx + dx
                    new_center_y = center_y * sy + dy
                    
                    # Preserve original width and height
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Calculate new corners
                    new_x1 = new_center_x - width / 2
                    new_y1 = new_center_y - height / 2
                    new_x2 = new_center_x + width / 2
                    new_y2 = new_center_y + height / 2
                    
                    # Ensure coordinates are within image boundaries
                    img_h, img_w = self.current_img_rgb.shape[:2]
                    new_x1 = max(0, min(new_x1, img_w - 1))
                    new_y1 = max(0, min(new_y1, img_h - 1))
                    new_x2 = max(0, min(new_x2, img_w - 1))
                    new_y2 = max(0, min(new_y2, img_h - 1))
                    
                    # Convert to integers for slice indexing
                    transformed_rects.append((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
            
            # Update the rectangles for this source
            self.rectangles[source_idx] = transformed_rects
            
            # Redraw
            load_current_image(self)
    
    def _draw_multiple_rectangles(self):
        """Helper method to draw all rectangles for the current source in multiple rectangle mode"""
        if not self.multiple_rectangles:
            return
            
        current_source = self.source_var.get()
        idx = ["Digital (DA)", "Negative 1", "Negative 2"].index(current_source)
        
        for rect in self.rectangles[idx]:
            x1, y1, x2, y2 = rect
            
            # Convert to canvas coordinates
            canvas_x1 = int(x1 * self.scale_factor) + self.img_x_offset
            canvas_y1 = int(y1 * self.scale_factor) + self.img_y_offset
            canvas_x2 = int(x2 * self.scale_factor) + self.img_x_offset
            canvas_y2 = int(y2 * self.scale_factor) + self.img_y_offset
            
            self.canvas.create_rectangle(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2, 
                outline="red", width=2, tags="rect"
            )