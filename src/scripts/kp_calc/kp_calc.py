import tkinter as tk
from app import RectangleDrawingApp
import json
import os

def main():
    # Determine the root directory based on current working directory
    current_dir = os.getcwd()
    if "kp_calc" in current_dir:
        root_directory = '../../../'
    else:
        root_directory = '.'
    
    mapping_path = os.path.join(root_directory, 'data/prepro/manual_image_mapping.json')
    with open(mapping_path, 'r') as f:
        image_mappings = json.load(f)
    
    root = tk.Tk()
    app = RectangleDrawingApp(root, image_mappings, root_directory=root_directory)
    
    root.mainloop()

if __name__ == "__main__":
    main()
