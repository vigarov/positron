import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
from pathlib import Path
import argparse
import cv2
import numpy as np
import traceback
import tifffile
from tqdm import tqdm

def load_image_robust(image_path):
    """Load an image using multiple methods for better compatibility."""
    if not os.path.exists(image_path):
        raise ValueError(f"File does not exist: {image_path}")
    if not os.access(image_path, os.R_OK):
        raise ValueError(f"File is not readable: {image_path}")
    
    try:
        # It's weird - some images sometimes get loaded correctly with PIL, and sometimes PIL can't handle them and errors.
        # After trial and error, tifffile manages to always open the images. I thus use it first, and try the rest of the libs afterwards only if it fails (never happened yet)
        img_array = tifffile.imread(image_path)
        
        if img_array.dtype != np.uint8:
            img_array = ((img_array - img_array.min()) / 
                         (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] > 3:
            raise ValueError(f"Image has more than 3 channels: {image_path}")
            
        return Image.fromarray(img_array)
    except Exception as e:
        print(f"Error loading {image_path}: {str(e)}")
    
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            # PIL weirdly uses BGR -> convert it here to RGB
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            return Image.fromarray(img)
    except Exception as e:
        pass
    
    # If both fail, try PIL  as last resort: note this almost always fails for some reason
    try:
        return Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Could not load image with any method: {image_path}")

class ImageMatcher(tk.Tk):
    def __init__(self, dir1=None, dir2=None, dir3=None):
        super().__init__()
        
        self.title("Manual Image Matcher")
        self.geometry("1200x800")
        
        self.directories = [
            dir1 or "data/prepro/da_tiff",
            dir2 or "data/prepro/negative_film_tiff/source1",
            dir3 or "data/prepro/negative_film_tiff/source2"
        ]
        
        self.image_paths = [[], [], []]
        self.thumbnails = [[], [], []]
        self.selected_indices = [-1, -1, -1]
        self.matches = []  # will be tuples (path1, path2, path3)
        
        self.create_menu()
        self.create_ui()
        
        for i, directory in enumerate(self.directories):
            self.load_directory(i, directory)
        
        total_remaining = sum(len(paths) for paths in self.image_paths)
        self.progress_bar["maximum"] = total_remaining + len(self.matches)
        self.progress_bar["value"] = len(self.matches)
        self.update_idletasks()
    
    def create_menu(self):
        menu = tk.Menu(self)
        self.config(menu=menu)
        
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Directory 1", command=lambda: self.browse_directory(0))
        file_menu.add_command(label="Load Directory 2", command=lambda: self.browse_directory(1))
        file_menu.add_command(label="Load Directory 3", command=lambda: self.browse_directory(2))
        file_menu.add_separator()
        file_menu.add_command(label="Save Matches", command=self.save_matches)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
    
    def create_ui(self):
        # Basic UI creation w/ ttk
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.gallery_frames = []
        self.image_listboxes = []
        self.image_previews = []
        
        for i in range(3):
            frame = ttk.LabelFrame(main_frame, text=f"Directory {i+1}")
            frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            
            list_frame = ttk.Frame(frame)
            list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            listbox = tk.Listbox(list_frame, width=30, height=10, exportselection=0, yscrollcommand=scrollbar.set)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=listbox.yview)
            listbox.bind("<<ListboxSelect>>", lambda event, idx=i: self.on_select(event, idx))
            
            preview = ttk.Label(frame)
            preview.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.gallery_frames.append(frame)
            self.image_listboxes.append(listbox)
            self.image_previews.append(preview)
        
        for i in range(3):
            main_frame.columnconfigure(i, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        
        match_button = ttk.Button(button_frame, text="Create Match", command=self.create_match)
        match_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        save_button = ttk.Button(button_frame, text="Save Matches", command=self.save_matches)
        save_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.status_label = ttk.Label(button_frame, text="No matches created")
        self.status_label.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
    
    def browse_directory(self, directory_index):
        directory = filedialog.askdirectory(initialdir=self.directories[directory_index])
        if directory:
            self.directories[directory_index] = directory
            self.load_directory(directory_index, directory)
    
    def load_directory(self, directory_index, directory):
        self.image_paths[directory_index] = []
        self.thumbnails[directory_index] = []
        self.image_listboxes[directory_index].delete(0, tk.END)
        
        if not os.path.exists(directory):
            messagebox.showerror("Error", f"Directory does not exist: {directory}")
            return
        
        image_paths = []
        extensions = ['.tif', '.tiff']
        
        seen_paths = set()
        
        for file_path in Path(directory).iterdir():
            if file_path.is_file():
                for ext in extensions:
                    if file_path.name.lower().endswith(ext):
                        if file_path.name.lower() not in seen_paths:
                            image_paths.append(file_path)
                            seen_paths.add(file_path.name.lower())
                        break
        
        self.gallery_frames[directory_index].config(text=f"Directory {directory_index+1}: {Path(directory).name} ({len(image_paths)} images)")
        
        # Progress bar for loading
        loading_progress = ttk.Progressbar(self, orient="horizontal", length=100, mode="determinate", maximum=len(image_paths))
        loading_progress.pack(fill="x", padx=10, pady=10)
        
        for i, path in enumerate(tqdm(image_paths, desc=f"Loading dir {directory_index+1}")):
            try:
                self.image_paths[directory_index].append(str(path))
                
                # Creating thumbnails so that we do not load full image in memory (my PC crashed several times because of OOM before that haha)
                img = load_image_robust(path)
                img.thumbnail((200, 200))
                img_tk = ImageTk.PhotoImage(img)
                self.thumbnails[directory_index].append(img_tk)
                
                self.image_listboxes[directory_index].insert(tk.END, path.name)
                
                loading_progress["value"] = i + 1
                self.update_idletasks()
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
                traceback.print_exc()
        
        loading_progress.destroy()
        
        total_remaining = sum(len(paths) for paths in self.image_paths)
        self.progress_bar["maximum"] = total_remaining + len(self.matches)
        self.progress_bar["value"] = len(self.matches)
        self.update_idletasks()
    
    def on_select(self, event, directory_index):
        listbox = event.widget
        if not listbox.curselection():
            return
        
        index = listbox.curselection()[0]
        self.selected_indices[directory_index] = index
        
        # Update preview
        if 0 <= index < len(self.thumbnails[directory_index]):
            self.image_previews[directory_index].config(image=self.thumbnails[directory_index][index])
    
    def create_match(self):
        if -1 in self.selected_indices:
            messagebox.showwarning("Warning", "Please select an image from each directory")
            return
        
        # Create match tuple
        match = tuple(self.image_paths[i][self.selected_indices[i]] for i in range(3))
        self.matches.append(match)
        
        self.status_label.config(text=f"Matches: {len(self.matches)}")
        
        # Remove selected items from each list (in reverse to maintain indices)
        for i in range(3):
            if self.selected_indices[i] >= 0:
                self.image_paths[i].pop(self.selected_indices[i])
                self.thumbnails[i].pop(self.selected_indices[i])
                
                self.image_listboxes[i].delete(self.selected_indices[i])
                
                self.selected_indices[i] = -1
                self.image_previews[i].config(image="")
        
        # Update progress bar to show remaining images across all directories
        total_remaining = sum(len(paths) for paths in self.image_paths)
        self.progress_bar["maximum"] = total_remaining + len(self.matches)
        self.progress_bar["value"] = len(self.matches)
        self.update_idletasks()
        
        messagebox.showinfo("Success", "Match created")
    
    def save_matches(self):
        if not self.matches:
            messagebox.showwarning("Warning", "No matches to save")
            return
        
        output_file = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="data/prepro",
            initialfile="manual_image_mapping.json"
        )
        
        if not output_file:
            return
        
        mapping = {}
        for i, match in enumerate(self.matches):
            mapping[i] = match
        
        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        messagebox.showinfo("Success", f"Matches saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Manual Image Matcher GUI')
    parser.add_argument('--dir1', type=str, default=None,
                      help='Path to directory 1 (default: data/prepro/da_tiff)')
    parser.add_argument('--dir2', type=str, default=None,
                      help='Path to directory 2 (default: data/prepro/negative_film_tiff/source1)')
    parser.add_argument('--dir3', type=str, default=None,
                      help='Path to directory 3 (default: data/prepro/negative_film_tiff/source2)')
    args = parser.parse_args()
    
    app = ImageMatcher(args.dir1, args.dir2, args.dir3)
    app.mainloop()

if __name__ == "__main__":
    main() 