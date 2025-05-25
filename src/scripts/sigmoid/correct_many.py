import argparse
import json
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import itertools
from math import ceil
from matplotlib.backends.backend_pdf import PdfPages
import cv2
from tqdm import tqdm
# Assuming the script is in positron/scripts/sigmoid/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.processing.sigmoid_correct import SigmoidCorrect
import multiprocessing as mp
from PyPDF2 import PdfMerger


N_COLS = 6

def load_and_normalize_tiff(image_path):
    try:
        img = tifffile.imread(image_path)
    except FileNotFoundError:
        # Running the script from a different dir might cause this, trying to load from project root
        print(f"Error: Image file not found at {image_path}")
        image_path_obj = Path(image_path)
        if not image_path_obj.is_absolute() and str(image_path_obj).startswith("data"):
             alt_path = Path(project_root) / image_path.replace('\\', os.sep)
             print(f"Attempting to load from: {alt_path}")
             try:
                 img = tifffile.imread(alt_path)
             except FileNotFoundError:
                 print(f"Error: Image file still not found at {alt_path}")
                 raise
        else:
            raise    
    if np.min(img) < -1e-5 or np.max(img) > 1.0 + 1e-5:
        print(f"Warning: Input image values are outside the expected [0,1] range. Min: {np.min(img)}, Max: {np.max(img)} ; 99th percentile: {np.percentile(img,99)}")
        img = (img - np.min(img))/(np.max(img) - np.min(img))
    if img.ndim == 2 or img.shape[-1] >= 4:
        print(f"Error: Input image is not a 3-channel RGB image")
        exit(-1)
    return img

def parse_args():
    parser = argparse.ArgumentParser(description="Apply Sigmoid Correction with various parameters to all images. Creates a pdf where each page is dedicated to one image, and each page contains plots for processing that image with different parameters.")
    parser.add_argument("-i", "--index", default=None,type=str,help="Select which picture (index inside mappings file) to parse. Use comma-seperatedd digits for multiple indices. If this argument is not specified, applies to all images.")
    parser.add_argument("-o", "--output_dir",default="outs", type=str, help="Path to the output directory.")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX document instead of matplotlib plots. Images will be saved in lower resolution in an 'images' folder.")
    parser.add_argument("-P","--processed_path",type=str, default=None, help="Only generate the LaTeX file (don't process the images) using the images that are in the parent folder specified by this argument. Only useful with combination with --latex")
    parser.add_argument("mappings", type=str, help="Path to the image mapping JSON file.")
    parser.add_argument("config", type=str, help="Path to the PDF \"experiment\"-generating config JSON file used to define the constant and variable parameters for generating the different plots on each page of the output PDF.")
    args = parser.parse_args()
    if args.index:
        args.index = [int(idx) for idx in args.index.split(',')]
    return args

def parse_param(param):
    if isinstance(param, str):
        if param.startswith('linspace:'):
            splitted = param.split(':')
            assert len(splitted) == 2,f"linspace param must have 4 parts, got {splitted}"
            start, end, n_steps = splitted[1].split(',')
            return list(np.linspace(float(start), float(end), int(n_steps)))
        elif ":" in param:
            assert '=' in param
            value, sub_param = param.split(':')
            assert '=' in sub_param and len(sub_param.split('=')) == 2
            sub_key, sub_value = sub_param.split('=')
            parsed_sub_value = parse_param(sub_value)
            assert not isinstance(parsed_sub_value, dict),f"sub_value {sub_value} is a dict"
            return {value: {sub_key: parsed_sub_value}}
        else:
            if param.lower().isdigit():
                return int(param)
            elif param.lower() == 'true':
                return True
            elif param.lower() == 'false':
                return False
            else:
                try:
                    flt = float(param)
                    return flt
                except ValueError:
                    return param
    elif isinstance(param, list):
        return [parse_param(p) for p in param]
    else:
        return param

def shorten_key(key):
    if key == 'wb_method':
        return 'wb'
    elif key == 'correction_mode':
        return 'mode'
    elif key == 'lumi_factor':
        return 'lumi'
    else:
        if isinstance(key, str) and '_' in key:
            keys_splitted = key.split('_')
            return '_'.join([k[:2] for k in keys_splitted])
        else:
            return key    

def shorten_value(value):
    if isinstance(value, str) and '_' in value:
        values_splitted = value.split('_')
        return '_'.join([v[:2] for v in values_splitted])
    elif isinstance(value, float):
        return f"{value:.2f}"
    return value

def parse_experiment_config(config):
    constant_params = config['constant_params']
    variable_params = config['variable_params']
    set_of_SigmoidCorrect_configs = {}

    base_config = {}
    for k,v in constant_params.items():
        param = parse_param(v)
        assert isinstance(param,(str,float,int,bool)) 
        base_config[k] = param


    for k,v in variable_params.items():
        param_values = parse_param(v)
        assert isinstance(param_values, (list,dict))
        variable_params[k] = param_values

    param_keys,param_values = zip(*variable_params.items())
    all_variable_param_permutations = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]
    
    for perm in all_variable_param_permutations:
        current_config = base_config.copy()
        current_config_name_components = []
        for k,v in perm.items():
            if isinstance(v, dict):
                assert len(v) == 1
                value = list(v.keys())[0]
                sub_params = list(v.values())[0]
                current_config_name_components.append(f"{shorten_key(k)}={shorten_value(value)}")
                assert isinstance(sub_params, dict) and len(sub_params) == 1
                sub_k, sub_v = list(sub_params.items())[0]
                current_config_name_components.append(f"{shorten_key(sub_k)}={shorten_value(sub_v)}")
            else:
                current_config_name_components.append(f"{shorten_key(k)}={shorten_value(v)}")
        
        final_config = {}
        current_config.update(perm)
        for k,v in current_config.items():
            if isinstance(v, dict):
                assert len(v) == 1
                value = list(v.keys())[0]
                sub_params = list(v.values())[0]
                final_config[k] = value
                assert isinstance(sub_params, dict) and len(sub_params) == 1
                sub_k, sub_v = list(sub_params.items())[0]
                final_config[sub_k] = sub_v    
            else:
                final_config[k] = v
        set_of_SigmoidCorrect_configs['\n'.join(current_config_name_components)] = final_config
    return set_of_SigmoidCorrect_configs

def print_format_value(v):
    if isinstance(v,float):
        return f"{v:.3f}"
    elif isinstance(v,dict):
        assert len(v) == 1
        true_v = list(v.keys())[0]
        next_params = v[true_v]
        assert isinstance(next_params,dict) and len(next_params)==1
        sub_param_k,sub_param_v = list(next_params.items())[0]
        return f"{print_format_value(true_v)}:{str(sub_param_k)}={print_format_value(sub_param_v)}"
    if isinstance(v,list):
        return " OR ".join([print_format_value(e) for e in v])
    return str(v)

def save_image_for_latex(img, filename, output_dir):
    if img.dtype != np.uint8:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img
    
    filepath = Path(output_dir) / filename
    cv2.imwrite(str(filepath), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

def process_single_image_latex(args_tuple):
    idx, files, set_of_SigmoidCorrect_configs, constant_params_config, variable_params_config, output_dir = args_tuple
    common_result = process_images_common(idx, files, set_of_SigmoidCorrect_configs)
    
    img_dir = Path(output_dir) / "images" / f"img_{idx}"
    img_dir.mkdir(parents=True, exist_ok=True)
    save_image_for_latex(common_result['da_tiff'], "ground_truth.png", str(img_dir))
    
    config_images = []
    for proc_config in common_result['processed_configs']:
        i = proc_config['config_index']
        config_name = proc_config['config_name']
        
        img_filename_src1 = f"config_{i:03d}_source1.png"
        img_filename_src2 = f"config_{i:03d}_source2.png"
        save_image_for_latex(proc_config['resized_src1'], img_filename_src1, str(img_dir))
        save_image_for_latex(proc_config['resized_src2'], img_filename_src2, str(img_dir))
        
        config_images.append({
            'filename_src1': img_filename_src1,
            'filename_src2': img_filename_src2,
            'config_name': config_name,
            'config_index': i
        })
    
    return {
        'idx': idx,
        'ground_truth_path': f"images/img_{idx}/ground_truth.png",
        'config_images': config_images,
        'constant_params': constant_params_config,
        'variable_params': variable_params_config,
        'is_landscape': common_result['is_landscape']
    }

# Common image processing logic shared by both matplotlib and LaTeX modes
def process_images_common(idx, files, set_of_SigmoidCorrect_configs):
    print(f"Processing image {idx}... from process {os.getpid()}")
    assert isinstance(files, list) and len(files) == 3
    da_tiff_path = files[0]
    source1_path = files[1]
    source2_path = files[2]
    
    da_tiff = load_and_normalize_tiff(da_tiff_path)
    source1 = load_and_normalize_tiff(source1_path)
    source2 = load_and_normalize_tiff(source2_path)
    
    h, w = da_tiff.shape[:2]
    is_landscape = w > h
    
    aspect_ratio = h / w
    new_width = 512
    new_height = int(new_width * aspect_ratio)
    resized_da_tiff = cv2.resize(da_tiff, (new_width, new_height))
    
    processed_configs = []
    for i, (config_name, config) in tqdm(enumerate(set_of_SigmoidCorrect_configs.items()),total=len(set_of_SigmoidCorrect_configs),desc=f"Processing img {idx} configurations"):
        processor = SigmoidCorrect(config)
        processed_src1 = processor.apply(source1)
        processed_src2 = processor.apply(source2)
        # processed_src1, processed_src2 = source1, source2
        
        h1, w1 = processed_src1.shape[:2]
        h2, w2 = processed_src2.shape[:2]
        new_width = 256
        new_height1 = int(new_width * (h1 / w1))
        new_height2 = int(new_width * (h2 / w2))
        
        resized_src1 = cv2.resize(processed_src1, (new_width, new_height1))
        resized_src2 = cv2.resize(processed_src2, (new_width, new_height2))
        
        processed_configs.append({
            'config_name': config_name,
            'config_index': i,
            'resized_src1': resized_src1,
            'resized_src2': resized_src2
        })
    
    return {
        'idx': idx,
        'da_tiff': resized_da_tiff,
        'is_landscape': is_landscape,
        'processed_configs': processed_configs
    }

def process_single_image(args_tuple):
    idx, files, set_of_SigmoidCorrect_configs, constant_params_config, variable_params_config, output_dir = args_tuple
    
    # Use common processing logic
    common_result = process_images_common(idx, files, set_of_SigmoidCorrect_configs)
    processed_configs = common_result['processed_configs']
    da_tiff = common_result['da_tiff']
    
    n_configs = len(processed_configs)
    center_col = N_COLS // 2
    num_meta_rows = 2
    n_rows = num_meta_rows + ceil(n_configs/N_COLS)  
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(20, 5*n_rows))
    
    for ax_idx in range(N_COLS):
        axes[0, ax_idx].axis('off')

    param_ax = axes[0,center_col]
    param_text = "Config Parameters\nConstant: "
    if constant_params_config:
        param_text += "//\t".join([f"{k}: {print_format_value(v)}" for k,v in constant_params_config.items()])
    else:
        param_text += "None\n"
    
    param_text += "\nVariable: "
    if variable_params_config:
        param_text += "//\t".join([f"{k}: {print_format_value(v)}" for k,v in variable_params_config.items()])
    else:
        param_text += "None\n"
    
    param_ax.text(0,1, param_text, transform=param_ax.transAxes, fontsize=8,
                  va='top', ha='center', wrap=True,
                  bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', alpha=0.7))

    center_ax = axes[1,center_col] 
    center_ax.imshow(da_tiff)
    center_ax.set_title('DSLR image ("Ground Truth")')
    center_ax.axis('off')
    # Turn off any unused axes
    for i in range(N_COLS):
        axes[1,i].axis('off')
        

    for proc_config in tqdm(processed_configs, desc=f"Processing img {idx} configurations"):
        i = proc_config['config_index']
        config_name = proc_config['config_name']
        resized_src1 = proc_config['resized_src1']
        resized_src2 = proc_config['resized_src2']
        
        row = (i // N_COLS) + num_meta_rows
        col = i % N_COLS
        
        ax = axes[row,col]
        ax.imshow(np.hstack([resized_src1, resized_src2]))
        ax.set_title(config_name, fontsize=8)
        ax.axis('off')
        
    fig.tight_layout(pad=1.0, h_pad=2.0, w_pad=1.0)

    # Save to a temporary PDF - will merge after all processes are done
    temp_pdf_path = Path(output_dir) / f"temp_image_{idx}.pdf"
    with PdfPages(temp_pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return str(temp_pdf_path)

def escape_latex_string(text):
    # Replace special LaTeX characters
    text = text.replace('\\', '\\textbackslash{}')
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('$', '\\$')
    text = text.replace('#', '\\#')
    text = text.replace('^', '\\textasciicircum{}')
    text = text.replace('_', '\\_')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    text = text.replace('~', '\\textasciitilde{}')
    text = text.replace('\n', ';')
    # if len(text) > 50:
    #     text = text[:47] + "..."
    return text

def generate_latex_document(image_results, output_path):
    """Generate a LaTeX document that displays the processed images."""
    
    latex_content = r"""
\documentclass[a4paper,11pt]{article}
\usepackage[margin=1cm]{geometry}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{float}
\usepackage{amsmath}
\usepackage{array}
\usepackage{longtable}
\usepackage{pdflscape}
\linespread{0.5}

\title{Sigmoid Correction Results}
\author{Generated by correct\_many.py}
\date{\today}
\pagenumbering{gobble}
\begin{document}
\maketitle

"""
    assert image_results and len(image_results) > 0
    
    first_result = image_results[0]
    constant_params = first_result['constant_params']
    variable_params = first_result['variable_params']
    
    latex_content += "\\section{Configuration}\n\n"
    latex_content += "\\subsection{Constant Parameters:}\n"
    if constant_params:
        for k, v in constant_params.items():
            formatted_value = escape_latex_string(str(print_format_value(v)))
            latex_content += f"{escape_latex_string(str(k))}: {formatted_value}\\\\"
    else:
        latex_content += "None\\\\"
    
    latex_content += "\n\\subsection{Variable Parameters:}\n"
    if variable_params:
        for k, v in variable_params.items():
            formatted_value = escape_latex_string(str(print_format_value(v)))
            latex_content += f"{escape_latex_string(str(k))}: {formatted_value}\\\\"
    else:
        latex_content += "None\\\\"
    
    latex_content += "\n"
    
    # Images
    latex_content += "\\section{Images}\n\\newpage\n"
    for result in image_results:
        idx = result['idx']
        ground_truth_path = result['ground_truth_path']
        config_images = result['config_images']
        is_landscape = result.get('is_landscape', False)
        
        if is_landscape:
            latex_content += "\\begin{landscape}\n"
        
        latex_content += f"\\subsection*{{\\centering Image {idx}}}\n\n"
        # latex_content += "\\textbf{DSLR image (Ground Truth):}\\\\\n"
        latex_content += "\\begin{center}\n"
        latex_content += f"\\includegraphics[width=0.2\\textheight]{{{ground_truth_path}}}\n"
        latex_content += "\\end{center}\n\n"
        
        n_configs = len(config_images)
        n_rows = ceil(n_configs / N_COLS)
        
        if is_landscape:
            available_height = "\\dimexpr\\textwidth\\relax"  # Reserve space for titles and ground truth
            height_per_row = f"\\dimexpr({available_height})\\relax"
        else:
            available_height = "\\dimexpr\\textheight-6cm\\relax"  # Reserve ~6cm for titles, ground truth, and margins
            height_per_row = f"\\dimexpr({available_height})/{n_rows}\\relax"


        
        # latex_content += "\\textbf{Processed Results:}\\\\\n"
        latex_content += "\\hspace{-0.72cm}\n"
        # latex_content += "\\setlength{\\columnsep}{2pt}"
        for i, config_img in enumerate(config_images):
            if i % N_COLS == 0 and i > 0:
                latex_content += "\\\\\n"  # Remove vspace between rows
            
            config_name_comps = [escape_latex_string(comp) for comp in config_img['config_name'].split()]
            half = len(config_name_comps)//2
            config_name = ";".join(config_name_comps[:half])+"\\\\"+ ";".join(config_name_comps[half:])
            
            latex_content += f"\\begin{{minipage}}{{\\dimexpr\\linewidth/{N_COLS}\\relax}}%\n"
            latex_content += "\\centering\n"
            latex_content += f"{{\\tiny {config_name}}}\\\\\n"
            
            latex_content += f"\\includegraphics[width=0.49\\linewidth, height={height_per_row}, keepaspectratio]{{images/img_{idx}/{config_img['filename_src1']}}}%\n"
            latex_content += f"\\includegraphics[width=0.49\\linewidth, height={height_per_row}, keepaspectratio]{{images/img_{idx}/{config_img['filename_src2']}}}\n"
            latex_content += "\\end{minipage}%\n"  # Add % to prevent spacing between minipages
            
            # # Add horizontal spacing between columns (except for last column)
            # if (i + 1) % N_COLS != 0 and i < n_configs - 1:
            #     latex_content += "\\hspace{\\columnsep}\n"
        
        if is_landscape:
            latex_content += "\\end{landscape}\n\n"
        else:
            latex_content += "\n\n\\newpage\n\n"
    
    latex_content += "\\end{document}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"LaTeX document saved to {output_path}")
    print("To compile: pdflatex document.tex")

def build_synthetic_image_results(processed_path, image_mappings, original_constant_params, original_variable_params,set_of_SigmoidCorrect_configs):
    # Traverese the parent output dir and simply build the output dictionary
    images_dir = Path(processed_path)
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found at {images_dir}")
    
    config_index_to_name = {}
    for i, (config_name, _) in enumerate(set_of_SigmoidCorrect_configs.items()):
        config_index_to_name[i] = config_name
    
    image_results = []
    for idx_str in sorted(image_mappings.keys(), key=int):
        idx = int(idx_str)
        img_dir = images_dir / f"img_{idx}"
        
        if not img_dir.exists():
            print(f"Warning: Image directory {img_dir} not found, skipping...")
            continue
        ground_truth_path = img_dir / "ground_truth.png"
        if not ground_truth_path.exists():
            print(f"Warning: Ground truth image not found at {ground_truth_path}, skipping...")
            continue
    
        config_images = []
        config_files = {}
        
        # Group config files by index
        for img_file in img_dir.glob("config_*_source*.png"):
            filename = img_file.name
            # config_XXX_sourceY.png
            parts = filename.replace('.png', '').split('_')
            if len(parts) >= 3 and parts[0] == 'config' and parts[2].startswith('source'):
                try:
                    config_idx = int(parts[1])
                    source_num = parts[2]  # 'source1' or 'source2'
                    
                    if config_idx not in config_files:
                        config_files[config_idx] = {}
                    config_files[config_idx][source_num] = filename
                except ValueError:
                    continue
        
        for config_idx in sorted(config_files.keys()):
            sources = config_files[config_idx]
            if 'source1' in sources and 'source2' in sources:
                config_name = config_index_to_name.get(config_idx, f"Config_{config_idx:03d}")
                
                config_images.append({
                    'filename_src1': sources['source1'],
                    'filename_src2': sources['source2'],
                    'config_name': config_name,
                    'config_index': config_idx
                })
        
        # Set is_landscape to True (alway the case for our data; can use heuristics on ground truth if we want to generalize this code I guess)
        
        image_result = {
            'idx': idx,
            'ground_truth_path': f"images/img_{idx}/ground_truth.png",
            'config_images': config_images,
            'constant_params': original_constant_params,
            'variable_params': original_variable_params,
            'is_landscape': True
        }
        
        image_results.append(image_result)
        print(f"Discovered {len(config_images)} configurations for image {idx}")
    
    return image_results

def main():
    args = parse_args()
    with open(args.mappings, 'r') as f: 
        image_mappings = json.load(f)
    if args.index is None:
        args.index = list(range(len(image_mappings)))

    #Filter image mappings
    image_mappings = {k:v for k,v in image_mappings.items() if int(k) in args.index}

    with open(args.config, 'r') as f:
        config_json_content = json.load(f)
    
    original_constant_params = config_json_content.get('constant_params', {})
    original_variable_params = config_json_content.get('variable_params', {})

    set_of_SigmoidCorrect_configs = parse_experiment_config(config_json_content)
    print(f"There are {len(set_of_SigmoidCorrect_configs)} different configurations to be applied per image.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_workers = min(10, os.cpu_count() if os.cpu_count() else 1) # Use available CPUs, cap at 8
    pool = mp.Pool(processes=n_workers)

    process_args = [(idx, files, set_of_SigmoidCorrect_configs, 
                     original_constant_params, original_variable_params,
                     str(output_dir)) 
                   for idx, files in image_mappings.items()]

    print(f"Starting processing for {len(process_args)} images using {n_workers} worker(s)...")
    
    if args.latex:
        print("Running in LaTeX mode...")
        if args.processed_path is None:
            image_results = pool.map(process_single_image_latex, process_args) # Process single image_latex simply creates and saves the images to the correct directory --> all that is left is to create the .tex
        else:
            # Synthetic results
            image_results = build_synthetic_image_results(args.processed_path,image_mappings,original_constant_params,original_variable_params,set_of_SigmoidCorrect_configs)

        latex_output_path = output_dir / "document.tex"
        generate_latex_document(image_results, str(latex_output_path))
    else:
        print("Running in matplotlib mode...")
        pdf_output_path = output_dir / "all_image_comparisons.pdf"
        temp_pdf_paths = pool.map(process_single_image, process_args) # Process single image returns the path to the pdf w/ the image output
        
        merger = PdfMerger()
        for pdf_path in temp_pdf_paths:
            merger.append(pdf_path)
        merger.write(str(pdf_output_path))
        merger.close()
        
        # Clean up temp files
        for pdf_path in temp_pdf_paths:
            Path(pdf_path).unlink()
        
        print(f"PDF saved to {pdf_output_path}")
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()