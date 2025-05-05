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
            
            # Next Source button
            show_next_source = False
            current_source = self.source_var.get()
            if current_source == "Digital (DA)" and len(self.rectangles[0]) > 0 and not self.pixels_extracted:
                show_next_source = True
            elif current_source == "Negative 1" and len(self.rectangles[1]) > 0 and not self.pixels_extracted:
                show_next_source = True
            elif current_source == "Negative 2" and not self.all_rectangles_drawn and not self.pixels_extracted:
                show_next_source = True
                
            if show_next_source:
                self.next_source_btn.pack(side=tk.LEFT, padx=5)
            else:
                if hasattr(self, 'next_source_btn'):
                    self.next_source_btn.pack_forget()
        
        # Extract Pixels button (shows for both modes when appropriate)
        if all_drawn and not self.pixels_extracted:
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
                    # No need to delete existing rectangles in placement mode
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
                # For multiple rectangle mode placement on sources 1 and 2
                # We'll use the first rectangle's position as reference 
                # and draw all rectangles with their relative offsets
                pass
            
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
            elif self.multi_rect_placement_mode and self.current_img_idx != 0:
                # For multiple rectangles mode - draw all rectangles relative to first one
                self._draw_multi_placement_preview(event.x, event.y)
    
    def _draw_multi_placement_preview(self, x, y):
        """Draw all rectangles for multi-rectangle placement mode"""
        if not self.multiple_rectangles or not self.multi_rect_placement_mode:
            return
            
        self.canvas.delete("rect")
        
        # Get the original rectangles (from source 0)
        if not self.rectangles[0] or len(self.rectangles[0]) == 0:
            return
            
        # Assuming the mouse position is the top-left corner of the first rectangle
        dx = x - self.img_x_offset
        dy = y - self.img_y_offset
        
        # Convert to original image coordinates
        base_x = int(dx / self.scale_factor)
        base_y = int(dy / self.scale_factor)
        
        # Draw each rectangle with its offset from the first one
        for i, offset in enumerate(self.rect_offsets):
            dx1, dy1, dx2, dy2 = offset
            orig_rect = self.rectangles[0][i]
            width = orig_rect[2] - orig_rect[0]
            height = orig_rect[3] - orig_rect[1]
            
            # Calculate new rectangle coordinates
            x1 = base_x + dx1
            y1 = base_y + dy1
            x2 = x1 + width
            y2 = y1 + height
            
            # Ensure coordinates are within image boundaries
            img_h, img_w = self.current_img_rgb.shape[:2]
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))
            
            # Convert back to canvas coordinates
            canvas_x1 = int(x1 * self.scale_factor) + self.img_x_offset
            canvas_y1 = int(y1 * self.scale_factor) + self.img_y_offset
            canvas_x2 = int(x2 * self.scale_factor) + self.img_x_offset
            canvas_y2 = int(y2 * self.scale_factor) + self.img_y_offset
            
            # Draw the rectangle
            self.canvas.create_rectangle(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                outline="red", width=2, tags="rect"
            )
    
    def on_mouse_move(self, event):
        if self.pixels_extracted:
            return
            
        if self.drawing:
            # Different handling for multiple rectangle mode
            if self.multiple_rectangles:
                if self.multi_rect_placement_mode and self.current_img_idx != 0:
                    # For multi-rectangle placement mode, preview all rectangles
                    self._draw_multi_placement_preview(event.x, event.y)
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
                    # For multi-rectangle placement mode, save all rectangle positions
                    self._save_multi_placement_rectangles(event.x, event.y)
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
        self.canvas.delete("rect")
        
        # Get the original rectangles (from source 0)
        if not self.rectangles[0] or len(self.rectangles[0]) == 0:
            return
            
        # Calculate the base position from the mouse position
        dx = event_x - self.img_x_offset
        dy = event_y - self.img_y_offset
        
        # Convert to original image coordinates
        base_x = int(dx / self.scale_factor)
        base_y = int(dy / self.scale_factor)
        
        # Create new rectangles for the current source
        new_rects = []
        img_h, img_w = self.current_img_rgb.shape[:2]
        
        for i, offset in enumerate(self.rect_offsets):
            dx1, dy1, dx2, dy2 = offset
            
            # Calculate new rectangle coordinates
            orig_rect = self.rectangles[0][i]
            width = orig_rect[2] - orig_rect[0]
            height = orig_rect[3] - orig_rect[1]
            
            x1 = base_x + dx1
            y1 = base_y + dy1
            x2 = x1 + width
            y2 = y1 + height
            
            # Ensure coordinates are within image boundaries
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))
            
            new_rects.append((x1, y1, x2, y2))
        
        # Update the rectangles for this source
        self.rectangles[self.current_img_idx] = new_rects
        
        # Redraw the rectangles
        self._draw_multiple_rectangles()
        
        self.status_var.set(f"Positioned {len(new_rects)} rectangles on {self.source_var.get()}")
        
        self.update_ui_state()
    
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
        
        self.pixels_extracted = True
        
        # Skip displaying visualizations in data processing mode
        if hasattr(self, 'results_tabs'):
            self.results_tabs.pack_forget()
        
        self.update_ui_state()
        
        # Simply inform that data was processed and allow user to continue with the same set
        self.status_var.set(f"Data processed and saved to {self.data_processing_path}. Press 'Clear Rectangle' to process another region.")
        
        # Automatically clear the rectangle in data processing mode
        if hasattr(self, 'data_processing_path') and self.data_processing_path and success:
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
                
            # Store the rectangle sizes and calculate offsets for relative placement
            # We'll use the first rectangle as the reference point
            if not self.rect_offsets and len(self.rectangles[0]) > 0:
                first_rect = self.rectangles[0][0]
                self.rect_offsets = []
                for rect in self.rectangles[0]:
                    # Calculate offset from the first rectangle
                    x1, y1, x2, y2 = rect
                    fx1, fy1, fx2, fy2 = first_rect
                    offset = (x1 - fx1, y1 - fy1, x2 - fx1, y2 - fy1)
                    self.rect_offsets.append(offset)
            
            # Mark that we've drawn all rectangles on DA
            self.all_rects_drawn_on_da = True
            self.multi_rect_placement_mode = True
            
            # Switch to Negative 1
            self.current_img_idx = 1
            self.source_var.set("Negative 1")
            
            # Load the image before setting status
            load_current_image(self)
            self.status_var.set(f"Position all {len(self.rectangles[0])} rectangles on Negative 1 with one mouse drag")
            
        # If we're on Negative 1
        elif current_source == "Negative 1":
            # Check if we have positioned the rectangles
            if not self.rectangles[1] or len(self.rectangles[1]) == 0:
                self.status_var.set("Please position the rectangles before moving to the next source.")
                return
                
            # Switch to Negative 2
            self.current_img_idx = 2
            self.source_var.set("Negative 2")
            
            # Load the image before setting status
            load_current_image(self)
            self.status_var.set(f"Position all {len(self.rectangles[0])} rectangles on Negative 2 with one mouse drag")
            
        # If we're on Negative 2
        elif current_source == "Negative 2":
            # Check if we've positioned the rectangles on all sources
            all_positioned = (len(self.rectangles[0]) > 0 and 
                              len(self.rectangles[1]) > 0 and 
                              len(self.rectangles[2]) > 0)
            
            if not all_positioned:
                self.status_var.set("Please position the rectangles on all sources.")
                return
                
            # This means we're done with placing rectangles
            self.all_rectangles_drawn = True
            self.status_var.set("All rectangles positioned on all sources. You can now extract pixels or adjust positions if needed.")
        
        self.update_ui_state()

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