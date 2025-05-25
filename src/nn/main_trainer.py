import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import pickle as pkl
from skimage import color
import argparse
import random
import matplotlib.pyplot as plt
import sys
import torch.nn as nn
import importlib

# Add the parent directory to path to make imports work when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
MODEL_DIR = Path(__file__).resolve().parent / "models"
AVAILABLE_MODELS = [f.stem for f in MODEL_DIR.iterdir() if f.is_file() and f.suffix == '.py' and f.name not in ['__init__.py']]

from src.nn.utils.config import Config
from src.nn.utils.datasets import get_lab_dataloaders, get_rgb_dataloaders
from src.nn.utils.train import train_predictor
from src.nn.utils.constants import (
    REPRO_SEED, DEFAULT_CHECKPOINT_PATH, DATA_PATH, GRAYS_DATA,
    DEFAULT_EVAL_PATH, DEFAULT_NUM_SAMPLES, DEFAULT_SOURCE_IMG, DEFAULT_GT_IMG
)
from src.nn.eval import *

# Dynamically imports the model class and its config function
def _get_model_and_config_fn(model_type: str):
    try:
        module_name = f"src.nn.models.{model_type}"
        model_module = importlib.import_module(module_name)
    except ImportError:
        raise ValueError(f"Model type '{model_type}' not found or module could not be imported. Searched in {module_name}")

    # Construct class name (e.g., enhanced_comb -> EnhCombHiddenLayerNN)
    # This assumes snake_case_model.py -> SnakeCaseModelNN
    class_name_parts = [part.capitalize() for part in model_type.split('_')]
    if not class_name_parts[-1].endswith("NN"): 
        class_name_parts[-1] += "NN" 
    # Handle cases like "one_hidden" -> "OneHiddenNN" vs "OneHiddenLayerNN"
    # For now, we'll search for a class ending with NN in the module.
    model_class = None
    for attr_name in dir(model_module):
        attr = getattr(model_module, attr_name)
        if isinstance(attr, type) and issubclass(attr, nn.Module) and attr_name.endswith("NN"):
            model_class = attr
            break
    if model_class is None:
        # Try to guess from model_type for more complex names
        # e.g. enhanced_comb -> EnhCombHiddenLayerNN
        model_type_camel = "".join(word.capitalize() for word in model_type.split("_"))
        if not model_type_camel.endswith("NN"):
            # Hard code
            if model_type == "enhanced_comb":
                class_name_guess = "EnhCombHiddenLayerNN"
            elif model_type == "one_hidden":
                class_name_guess = "OneHiddenLayerNN"
            elif model_type == "combhidden":
                class_name_guess = "CombHiddenLayerNN"
            else:
                class_name_guess = model_type_camel + "NN"
        else:
            class_name_guess = model_type_camel

        try:
            model_class = getattr(model_module, class_name_guess)
        except AttributeError:
            raise ValueError(f"Could not find a suitable model class (e.g., {class_name_guess} or ending with NN) in {module_name}.")

    config_fn_name = f"get_{model_type}_model_config"
    try:
        config_fn = getattr(model_module, config_fn_name)
    except AttributeError:
        raise ValueError(f"Config function '{config_fn_name}' not found in {module_name}")
    
    return model_class, config_fn

def get_per_pixel_data(dataset, mean=True, use_rgb=False):
    if mean:
        per_pixel = dataset.groupby(by=["image_id", "pixel_num"]).mean()[["DA_R", "DA_G", "DA_B", "N1_R", "N1_G", "N1_B", "N2_R", "N2_G", "N2_B"]]
    else:
        per_pixel = dataset[["DA_R", "DA_G", "DA_B", "N1_R", "N1_G", "N1_B", "N2_R", "N2_G", "N2_B"]].copy()
    
    per_pixel['DA_RGB_RAW'] = list(zip(per_pixel.DA_R, per_pixel.DA_G, per_pixel.DA_B))
    per_pixel['N1_RGB_RAW'] = list(zip(per_pixel.N1_R, per_pixel.N1_G, per_pixel.N1_B))
    per_pixel['N2_RGB_RAW'] = list(zip(per_pixel.N2_R, per_pixel.N2_G, per_pixel.N2_B))
    
    per_pixel = per_pixel.drop(columns=['DA_R', 'DA_G', 'DA_B', 
                                      'N1_R', 'N1_G', 'N1_B',
                                      'N2_R', 'N2_G', 'N2_B'])
    
    for col_prefix in ["DA", "N1", "N2"]:
        rgb_col = f"{col_prefix}_RGB_RAW"
        target_col = f"{col_prefix}_RGB"
        per_pixel[target_col] = per_pixel[rgb_col].apply(lambda rgb_tuple: np.array(rgb_tuple)/255.0)
    
    per_pixel = per_pixel.drop(columns=["DA_RGB_RAW", "N1_RGB_RAW", "N2_RGB_RAW"])

    if not use_rgb:
        for col_prefix in ["DA", "N1", "N2"]:
            rgb_col = f"{col_prefix}_RGB"
            lab_col = f"{col_prefix}_LAB"
            per_pixel[lab_col] = per_pixel[rgb_col].apply(color.rgb2lab)
        per_pixel = per_pixel.drop(columns=["DA_RGB", "N1_RGB", "N2_RGB"])
    # If use_rgb is True, the columns DA_RGB, N1_RGB, N2_RGB (containing np.array of 0-1 RGB) are kept.
    
    return per_pixel

def load_model(model_type: str, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        specific_model_checkpoint_path = DEFAULT_CHECKPOINT_PATH / model_type
        if specific_model_checkpoint_path.exists():
            checkpoint_path = specific_model_checkpoint_path
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path} or {specific_model_checkpoint_path}")
    
    config_path = checkpoint_path / "config.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "rb") as f:
        config = pkl.load(f)

    model_class, _ = _get_model_and_config_fn(model_type)
    model = model_class(config)
    
    model_pt_path = checkpoint_path / "model.pt"
    if not model_pt_path.exists():
        raise FileNotFoundError(f"Model weights (model.pt) not found at {model_pt_path}")
    model.load_state_dict(torch.load(model_pt_path))
    model.eval() 
    return model

def train_model(model_type: str, data_path=None, override_previous_dir=False, model_name = None, epochs=None, use_rgb=False):
    """
    Train a new model
    
    Args:
        model_type: The type of model to train (e.g., "enhanced_comb")
        data_path: Path to the dataset CSV (defaults to constants.DATA_PATH)
        override_previous_dir: Whether to override existing model directory
        model_name: Custom name for the model config/directory
        epochs: Number of epochs to train for (overrides config default)
        use_rgb: Whether to train on RGB data instead of LAB
    
    Returns:
        model: Trained model
        test_dataset: The test dataset used
    """
    if data_path is None:
        data_path = DATA_PATH
        grays_path = GRAYS_DATA
    else:
        data_path = Path(data_path)
        grays_path = str(data_path).replace("new_color_checkers.csv", "grays.csv")
        if not Path(grays_path).exists():
            grays_path = GRAYS_DATA
    
    dataset = pd.read_csv(data_path).dropna()
    grays_df = pd.read_csv(grays_path).dropna()
    
    # Process grays data
    grays_df = grays_df.groupby(by=["image_id", "pixel_num"]).nth([0,1,4]).reset_index()
    grays_df["pixel_num"] = grays_df.pixel_num.apply(lambda n: n + dataset.pixel_num.max())
    full_dataset = pd.concat([dataset, grays_df]).sort_values(by=["image_id", "pixel_num"])
    
    # Convert to per-pixel LAB or RGB data
    per_pixel = get_per_pixel_data(full_dataset, mean=False, use_rgb=use_rgb)
    
    model_class, config_fn = _get_model_and_config_fn(model_type)

    config_params = {}
    if model_name:
        config_params['model_name'] = model_name
    if epochs is not None: # Only pass epochs if provided
        config_params['epochs'] = epochs
        
    config = config_fn(**config_params)
    if epochs is not None:
        config.update({'epochs': epochs})
    
    config.update({'color_space':'RGB' if use_rgb else 'LAB'}) # Store color space in config
    
    generator = torch.Generator().manual_seed(REPRO_SEED)
    
    dataloader_fn_to_use = get_rgb_dataloaders if use_rgb else get_lab_dataloaders
    
    model, _, _, _, test_dataset = train_predictor(
        config=config,
        df=per_pixel, # This df will contain either LAB or RGB columns based on use_rgb
        model_fn=model_class,
        dataloader_fn=dataloader_fn_to_use,
        override_previous_dir=override_previous_dir,
        generator=generator
    )
    
    return model, test_dataset

def get_model(model_type: str, train_new=False, data_path=None, checkpoint_path=None, model_name=None, epochs=None, use_rgb=False):
    if train_new:
        return train_model(model_type, data_path, model_name=model_name, epochs=epochs, use_rgb=use_rgb)
    else:
        model = load_model(model_type, checkpoint_path)
        return model, None

def evaluate_model(model, test_dataset, num_samples=DEFAULT_NUM_SAMPLES, 
                  source_img_path=None, gt_img_path=None, 
                  output_dir=DEFAULT_EVAL_PATH, display_width=800, color_space = 'LAB'):
    # Evaluate the model and generate visualizations
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating model in {color_space} color space.")
    print("Generating weight visualizations...")
    layer_names = [name for name, layer in model.named_children() 
                  if isinstance(layer, nn.Linear)]
    for layer_name in layer_names:
        try:
            fig, _ = visualize_layer_weights(model, layer_name)
            fig.savefig(output_dir / f"{layer_name}_weights.png", bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"  - Saved {layer_name} weights visualization")
        except (AttributeError, ValueError) as e:
            print(f"  - Couldn't visualize {layer_name}: {e}")
    
    # 2. Visualize random samples
    if test_dataset is not None:
        print("Generating sample visualizations...")
        for i in range(num_samples):
            idx = random.randint(0, len(test_dataset) - 1)
            fig, _ = viz_sample(model, test_dataset, idx, color_space=color_space)
            fig.savefig(output_dir / f"sample_{i}.png", bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"  - Saved sample visualization {i+1}/{num_samples}")
    
    # 3. Visualize image transformation
    if source_img_path is not None and gt_img_path is not None:
        print("Generating image transformation visualization...")
        try:
            fig = show_image_transform(model, source_img_path, gt_img_path, display_width=display_width, color_space=color_space)
            fig.savefig(output_dir / "image_transform.png", bbox_inches="tight", dpi=300)
            plt.close(fig)
            print("  - Saved image transformation visualization")
        except Exception as e:
            print(f"  - Couldn't visualize image transform: {e}")
    
    print(f"Evaluation complete. Results saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generic Model Trainer for color transformation")
    parser.add_argument("--model_type", type=str, required=True, choices=AVAILABLE_MODELS, help="Type of model to train/load.")
    parser.add_argument("--train", action="store_true", help="Train a new model instead of loading existing one")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train for (overrides config default if provided)")
    parser.add_argument("--data_path", type=str, help="Path to the dataset CSV")
    parser.add_argument("--rgb", action="store_true", help="Train using RGB values instead of LAB values")
    parser.add_argument("--checkpoint_path", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to the model checkpoint directory. If loading, can be specific model dir or parent dir.")
    parser.add_argument("--no-eval", action="store_true", help="Skip model evaluation and visualization") 
    parser.add_argument("--eval_path", type=str, default=DEFAULT_EVAL_PATH, help="Path to save evaluation outputs")
    parser.add_argument("--source_img", type=str, default=DEFAULT_SOURCE_IMG, help="Path to source image for transformation")
    parser.add_argument("--gt_img", type=str, default=DEFAULT_GT_IMG, help="Path to ground truth image for comparison")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of random samples to visualize")
    parser.add_argument("--display_width", type=int, default=800, help="Width to downsample images to for display (height will be calculated to maintain aspect ratio)")
    parser.add_argument("--model_name", type=str, default=None, help="Override default model name for saving/config")
    
    args = parser.parse_args()
    args.eval = not args.no_eval
    return args

if __name__ == "__main__":
    args = parse_args()
    
    effective_checkpoint_path = Path(args.checkpoint_path)
    if args.train:
        print(f"Training new {args.model_type} model...")
        model, test_dataset = get_model(model_type=args.model_type, 
                                        train_new=True, 
                                        data_path=args.data_path, 
                                        model_name=args.model_name, 
                                        epochs=args.epochs,
                                        use_rgb=args.rgb)
        print("Model training complete")
    else:
        print(f"Loading pre-trained {args.model_type} model...")
        model, _ = get_model(model_type=args.model_type, 
                                        checkpoint_path=effective_checkpoint_path,
                                        use_rgb=args.rgb)

        # Re-create a test_dataset if evaluating a loaded model
        test_dataset_for_eval = None
        if args.eval:
            data_path_for_eval = args.data_path if args.data_path else DATA_PATH
            grays_path_for_eval = str(data_path_for_eval).replace("new_color_checkers.csv", "grays.csv")
            if not Path(grays_path_for_eval).exists(): grays_path_for_eval = GRAYS_DATA
            
            dataset_eval = pd.read_csv(data_path_for_eval).dropna()
            grays_df_eval = pd.read_csv(grays_path_for_eval).dropna()
            grays_df_eval = grays_df_eval.groupby(by=["image_id", "pixel_num"]).nth([0,1,4]).reset_index()
            grays_df_eval["pixel_num"] = grays_df_eval.pixel_num.apply(lambda n: n + dataset_eval.pixel_num.max())
            full_dataset_eval = pd.concat([dataset_eval, grays_df_eval]).sort_values(by=["image_id", "pixel_num"])
            
            model_color_space = model.config.get('color_space', 'LAB')
            per_pixel_eval = get_per_pixel_data(full_dataset_eval, mean=False, use_rgb=(model_color_space == 'RGB'))
            
            temp_config_for_dataloader = Config(tt_split=model.config.get('tt_split',0.8), 
                                                bs=model.config.get('bs',64), 
                                                keep_one_source=model.config.get('keep_one_source',False))

            dataloader_fn_eval = get_rgb_dataloaders if model_color_space == 'RGB' else get_lab_dataloaders
            _, _, test_dataset_for_eval = dataloader_fn_eval(config=temp_config_for_dataloader, df=per_pixel_eval)
        test_dataset = test_dataset_for_eval

        print("Model loaded successfully")
    
    if args.eval:
        source_img_path = args.source_img if Path(args.source_img).exists() else None
        gt_img_path = args.gt_img if Path(args.gt_img).exists() else None
        
        if source_img_path is None or gt_img_path is None:
            print(f"Warning: Source image or ground truth image not found. Image transformation will be skipped.")
            if source_img_path is None:
                print(f"  Source image not found at: {args.source_img}")
            if gt_img_path is None:
                print(f"  Ground truth image not found at: {args.gt_img}")
        
        eval_output_dir = Path(args.eval_path)
        if args.model_name:
            eval_output_dir = eval_output_dir / args.model_name
        elif model.config.get('name'):
            eval_output_dir = eval_output_dir / model.config.get('name')
        else:
            eval_output_dir = eval_output_dir / args.model_type


        print(f"Evaluating model and saving results to {eval_output_dir}...")
        evaluate_model(
            model, 
            test_dataset=test_dataset,
            source_img_path=source_img_path,
            gt_img_path=gt_img_path,
            output_dir=eval_output_dir,
            num_samples=args.num_samples,
            display_width=args.display_width,
            color_space='RGB' if args.rgb else 'LAB'
        )
    
    print("Done!")
