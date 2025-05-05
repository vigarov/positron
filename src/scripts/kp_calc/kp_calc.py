import tkinter as tk
from app import RectangleDrawingApp
import json
import os
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Key Point Calculator')
    parser.add_argument('--data_processing', type=str, help='Path to file for storing processed data')
    parser.add_argument('--multiple', action='store_true', help='Enable drawing multiple rectangles (only with --data_processing)')
    args = parser.parse_args()
    
    # Validate that --multiple is only used with --data_processing
    if args.multiple and not args.data_processing:
        parser.error("The --multiple flag can only be used with --data_processing")
    
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
    app = RectangleDrawingApp(root, image_mappings, root_directory=root_directory, 
                             data_processing_path=args.data_processing,
                             multiple_rectangles=args.multiple)
    
    root.mainloop()

if __name__ == "__main__":
    main()
