import tkinter as tk
from tkinter import ttk, filedialog
import os
import json
import datetime
import platform
import numpy as np

from ui_components import setup_ui
from image_processor import load_current_set, load_current_image
from data_handler import extract_pixels, generate_combined_csv_data

class RectangleDrawingApp:
    def __init__(self, root, image_mappings, num_sets=6, root_directory='.'):
        self.root = root
        self.root.title("Rectangle Drawing Tool")
        self.root.geometry("1600x1000")
        
        self.root_directory = root_directory
        
        self.maximize_window()
        
        # Use the previously createsd mappings to load the `num_sets` mappings 
        self.image_mappings = {int(k): v for k, v in image_mappings.items() if int(k) < num_sets}
        self.current_set_id = 0
        self.current_img_idx = 0  # 0=da_tiff, 1=source1, 2=source2
        
        # Rectangle coordinates for each image in the current set
        self.rectangles = {0: None, 1: None, 2: None}  # (x1, y1, x2, y2)
        self.drawing = False
        
        self.rect_width = None
        self.rect_height = None
        self.rect_placement_mode = False
        
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
        all_drawn = all(rect is not None for rect in self.rectangles.values())
        self.all_rectangles_drawn = all_drawn
        
        if self.rectangles[0] is not None and not self.pixels_extracted:
            self.set_combo.config(state="disabled")
        elif not any(self.rectangles.values()):
            self.set_combo.config(state="readonly")
        else:
            self.set_combo.config(state="disabled")
        
        if all_drawn and not self.pixels_extracted:
            self.source_combo.config(state="readonly")
            self.source_combo.bind("<<ComboboxSelected>>", self.on_source_changed)
        elif self.pixels_extracted:
            self.source_combo.config(state="disabled")
        else:
            self.source_combo.config(state="disabled")
        
        if all_drawn and not self.pixels_extracted:
            self.extract_pixels_btn.pack(side=tk.LEFT, padx=5)
        elif not all_drawn or self.pixels_extracted:
            if hasattr(self, 'extract_pixels_btn'):
                self.extract_pixels_btn.pack_forget()
        
        if self.rectangles[0] is not None:
            self.clear_rect_btn.pack(side=tk.LEFT, padx=5)
        else:
            if hasattr(self, 'clear_rect_btn'):
                self.clear_rect_btn.pack_forget()
        
        if self.pixels_extracted:
            self.save_results_btn.pack(side=tk.LEFT, padx=5)
            self.results_tabs.pack(fill=tk.BOTH, expand=True)
        elif hasattr(self, 'save_results_btn'):
            self.save_results_btn.pack_forget()
            self.results_tabs.pack_forget()
    
    def reset_workflow_state(self):
        """Reset the workflow state to initial"""
        self.rectangles = {0: None, 1: None, 2: None}
        
        self.all_rectangles_drawn = False
        self.pixels_extracted = False
        self.rect_placement_mode = False
        
        self.current_img_idx = 0
        self.source_var.set("Digital (DA)")
        
        self.current_pixel_data = None
        
        self.canvas.delete("rect")
        
        self.update_ui_state()
        
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
            
            self.canvas.delete("rect")
            
            # If all three rectangles are already drawn and we're not extracting pixels yet,
            # always use placement mode with the existing rectangle size
            all_drawn = all(rect is not None for rect in self.rectangles.values())
            if all_drawn and not self.pixels_extracted:
                x1, y1, x2, y2 = self.rectangles[self.current_img_idx]
                self.rect_width = x2 - x1
                self.rect_height = y2 - y1
                self.rect_placement_mode = True
            
            # If in placement mode, create the rectangle immediately with fixed size
            if self.rect_placement_mode:
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
        
        combined_csv_data = generate_combined_csv_data(self)
        
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
            
            combined_csv_data = generate_combined_csv_data(self)
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