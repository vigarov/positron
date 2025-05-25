import pandas as pd
import numpy as np
from skimage import color
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import statsmodels.formula.api as smf
import sys
from pathlib import Path
import argparse
from prettytable import PrettyTable

LOG_EPSILON = 1e-5 

def rgb_to_log_density(rgb_norm_array):
    return -np.log10(np.maximum(rgb_norm_array, LOG_EPSILON))

def log_density_to_rgb(log_density_array):
    return np.clip(10**(-log_density_array), 0, 1)

def rgb_array_to_lab_array(rgb_norm_array):
    if rgb_norm_array.ndim == 1:
        return color.rgb2lab(rgb_norm_array)
    return np.array([color.rgb2lab(sample) for sample in rgb_norm_array])

# Adjust path to import from src.nn.utils
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.nn.utils.constants import DATA_PATH, GRAYS_DATA
    print(f"Successfully imported DATA_PATH: {DATA_PATH} and GRAYS_DATA: {GRAYS_DATA}")
except ImportError as e:
    print(f"Warning: Could not import constants from src.nn.utils.constants ({e}). Using default relative paths for data.")
    current_script_dir = Path(__file__).resolve().parent
    project_root_fallback = current_script_dir.parent.parent 
    DATA_PATH = str(project_root_fallback / "data" / "new_color_checkers.csv")
    GRAYS_DATA = str(project_root_fallback / "data" / "grays.csv")
    print(f"Using fallback DATA_PATH: {DATA_PATH}")
    print(f"Using fallback GRAYS_DATA: {GRAYS_DATA}")
except Exception as e:
    print(f"An unexpected error occurred during constants import: {e}")
    DATA_PATH, GRAYS_DATA = None, None

def load_and_prepare_data(data_path_str, grays_path_str, use_mean_data=False, regression_type='lab', calculate_log_roundtrip=False):
    if not data_path_str or not Path(data_path_str).exists():
        raise FileNotFoundError(f"Data file not found: {data_path_str}")
    if not grays_path_str or not Path(grays_path_str).exists():
        raise FileNotFoundError(f"Grays data file not found: {grays_path_str}")

    dataset_df = pd.read_csv(data_path_str).dropna()
    grays_df = pd.read_csv(grays_path_str).dropna()

    # Process grays data (same as main_trainer.py)
    if "pixel_num" not in dataset_df.columns:
        raise ValueError("'pixel_num' column missing from the main dataset.")
    if "pixel_num" not in grays_df.columns or "image_id" not in grays_df.columns:
        raise ValueError("'pixel_num' or 'image_id' column missing from the grays dataset.")
        
    max_pixel_num_dataset = dataset_df.pixel_num.max() if not dataset_df.empty else 0
    if not grays_df.empty and {"image_id", "pixel_num"}.issubset(grays_df.columns):
        try:
            grays_df_processed = grays_df.groupby(by=["image_id", "pixel_num"]).nth([0,1,4]).reset_index()
            grays_df_processed["pixel_num"] = grays_df_processed.pixel_num.apply(lambda n: n + max_pixel_num_dataset)
            full_dataset = pd.concat([dataset_df, grays_df_processed]).sort_values(by=["image_id", "pixel_num"]).reset_index(drop=True)
        except IndexError:
            print("Warning: Not enough rows in some groups of grays_df for .nth([0,1,4]). Using grays_df as is for concatenation.")
            grays_df["pixel_num"] = grays_df.pixel_num.apply(lambda n: n + max_pixel_num_dataset)
            full_dataset = pd.concat([dataset_df, grays_df]).sort_values(by=["image_id", "pixel_num"]).reset_index(drop=True)

    else:
        print("Warning: Grays dataset is empty or missing key columns. Proceeding with main dataset only.")
        full_dataset = dataset_df.sort_values(by=["image_id", "pixel_num"]).reset_index(drop=True)
    
    # Per-pixel data processing
    required_rgb_cols = ["DA_R", "DA_G", "DA_B", "N1_R", "N1_G", "N1_B", "N2_R", "N2_G", "N2_B"]
    if not all(col in full_dataset.columns for col in required_rgb_cols):
        missing_cols = [col for col in required_rgb_cols if col not in full_dataset.columns]
        raise ValueError(f"Missing required RGB columns in the combined dataset: {missing_cols}. Available: {full_dataset.columns.tolist()}")

    if use_mean_data:
        print("Averaging RGB data per image_id and pixel_num.")
        group_by_cols = ["image_id", "pixel_num"]
        if not all(col in full_dataset.columns for col in group_by_cols):
            raise ValueError(f"Missing groupby columns for mean op. Available: {full_dataset.columns.tolist()}")
        columns_to_average = [col for col in required_rgb_cols if col in full_dataset.columns]
        per_pixel_df = full_dataset.groupby(by=group_by_cols, as_index=False)[columns_to_average].mean()
    else:
        per_pixel_df = full_dataset[required_rgb_cols + ["image_id", "pixel_num"]].copy()

    if not all(col in per_pixel_df.columns for col in required_rgb_cols):
        missing_cols = [col for col in required_rgb_cols if col not in per_pixel_df.columns]
        raise ValueError(f"Missing required RGB columns after processing for mean: {missing_cols}.")
        
    # Create and normalize RGB tuple columns (0-1 range)
    for prefix in ['DA', 'N1', 'N2']:
        r, g, b = per_pixel_df[f'{prefix}_R']/255.0, per_pixel_df[f'{prefix}_G']/255.0, per_pixel_df[f'{prefix}_B']/255.0
        per_pixel_df[f'{prefix}_RGB_NORM'] = list(zip(r, g, b))
        per_pixel_df[f'{prefix}_RGB_NORM'] = per_pixel_df[f'{prefix}_RGB_NORM'].apply(np.array)

    da_rgb_norm_all = np.array(per_pixel_df['DA_RGB_NORM'].tolist())
    n1_rgb_norm_all = np.array(per_pixel_df['N1_RGB_NORM'].tolist())
    n2_rgb_norm_all = np.array(per_pixel_df['N2_RGB_NORM'].tolist())

    # Determine primary data space
    if regression_type == 'lab':
        print("Primary data space: LAB")
        da_data = rgb_array_to_lab_array(da_rgb_norm_all)
        n1_data = rgb_array_to_lab_array(n1_rgb_norm_all)
        n2_data = rgb_array_to_lab_array(n2_rgb_norm_all)
    elif regression_type == 'log_density':
        print("Primary data space: Log-Density RGB")
        da_data = rgb_to_log_density(da_rgb_norm_all)
        n1_data = rgb_to_log_density(n1_rgb_norm_all)
        n2_data = rgb_to_log_density(n2_rgb_norm_all)
    else:
        raise ValueError(f"Unknown regression_type: {regression_type}")

    n1_data_log_rt, n2_data_log_rt = None, None
    if calculate_log_roundtrip:
        print("Calculating Log-Density roundtrip LAB features.")
        n1_log_density = rgb_to_log_density(n1_rgb_norm_all)
        n1_rgb_from_log = log_density_to_rgb(n1_log_density)
        n1_data_log_rt = rgb_array_to_lab_array(n1_rgb_from_log)

        n2_log_density = rgb_to_log_density(n2_rgb_norm_all)
        n2_rgb_from_log = log_density_to_rgb(n2_log_density)
        n2_data_log_rt = rgb_array_to_lab_array(n2_rgb_from_log)
        
    return da_data, n1_data, n2_data, n1_data_log_rt, n2_data_log_rt

# Performs linear regression. Metrics (RMSE, R2) are always computed in LAB space
# X_log_rt are features from log-density roundtrip, expected in LAB space if use_comb_formula is True
def perform_linear_regression(X, y, feature_name, regression_type, use_smf_ols=False, X_log_rt=None, use_comb_formula=False):
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")
    if use_comb_formula and (X_log_rt is None or X.shape[0] != X_log_rt.shape[0]):
        raise ValueError("X_log_rt must be provided and have same samples as X for combined formula.")
    if X.shape[0] == 0:
        print(f"Skipping regression for {feature_name} as there is no data.")
        return None, None

    # Split data
    if X_log_rt is not None:
        # Stratify if possible? For now, simple split.
        X_train, X_test, y_train_reg_space, y_test_reg_space, X_log_rt_train, X_log_rt_test = train_test_split(
            X, y, X_log_rt, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train_reg_space, y_test_reg_space = train_test_split(X, y, test_size=0.2, random_state=42)
        X_log_rt_train, X_log_rt_test = None, None # Ensure they are defined

    y_pred_reg_space = np.zeros_like(y_test_reg_space)

    comb_str = "+ Combined Formula" if use_comb_formula else ""
    regression_method_name = f"SMF OLS{comb_str}" if use_smf_ols else "Scikit-learn LinearRegression"
    
    print(f"--- Linear Regression ({regression_method_name}) for {feature_name} -> DA_DATA (in {regression_type} space) ---")
    print(f"Input X shape: {X_train.shape}, Output y shape: {y_train_reg_space.shape}")
    if X_log_rt_train is not None:
        print(f"Input X_log_rt shape: {X_log_rt_train.shape}")

    if use_smf_ols:
        all_params_smf = []
        for i in range(y_train_reg_space.shape[1]): # For each output channel (L, A, B)
            df_train_smf = pd.DataFrame()
            df_test_smf = pd.DataFrame()
            formula_parts = []

            if use_comb_formula:
                if regression_type != 'lab': # Should be caught by main but double check
                    raise ValueError("Combined formula requires 'lab' regression type for primary features.")
                # Assumes X_train and X_log_rt_train are LAB, 3 channels each
                df_train_smf[f'input_ch{i}'] = X_train[:, i]
                df_test_smf[f'input_ch{i}'] = X_test[:, i]
                formula_parts.append(f'input_ch{i}')
                
                df_train_smf[f'log_rt_ch{i}'] = X_log_rt_train[:, i]
                df_test_smf[f'log_rt_ch{i}'] = X_log_rt_test[:, i]
                formula_parts.append(f'log_rt_ch{i}')
            else:
                # Original SMF: use all input features from X_train for each target channel
                num_input_features = X_train.shape[1]
                input_col_names = [f'px{j}' for j in range(num_input_features)]
                for j in range(num_input_features):
                    df_train_smf[input_col_names[j]] = X_train[:, j]
                    df_test_smf[input_col_names[j]] = X_test[:, j]
                formula_parts.extend(input_col_names)
            
            df_train_smf['target_channel'] = y_train_reg_space[:, i]
            formula = 'target_channel ~ ' + ' + '.join(formula_parts)
            
            model_smf_channel = smf.ols(formula=formula, data=df_train_smf).fit()
            all_params_smf.append(model_smf_channel.params)
            y_pred_reg_space[:, i] = model_smf_channel.predict(df_test_smf)
        
        print(f"Model Parameters (SMF OLS, for each output channel from formula: {formula_parts}):")
        for idx, params in enumerate(all_params_smf):
            print(f"  Target Output Channel {idx}:")
            print(params)
    else: # Use scikit-learn (does not use comb formula directly, X would need pre-combination)
        if use_comb_formula:
            print("Warning: --comb flag is active but scikit-learn regression is selected. Combined features are not used by scikit-learn in this setup.")
        model_sklearn = LinearRegression()
        model_sklearn.fit(X_train, y_train_reg_space)
        y_pred_reg_space = model_sklearn.predict(X_test)
        print(f"Model Coefficients (Scikit-learn, trained in {regression_type} space):\\n{model_sklearn.coef_}")
        print(f"Model Intercept (Scikit-learn, trained in {regression_type} space):\\n{model_sklearn.intercept_}")

    if regression_type == 'log_density':
        y_true_rgb_norm = log_density_to_rgb(y_test_reg_space)
        y_pred_rgb_norm = log_density_to_rgb(y_pred_reg_space)
        y_true_lab = rgb_array_to_lab_array(y_true_rgb_norm)
        y_pred_lab = rgb_array_to_lab_array(y_pred_rgb_norm)
    elif regression_type == 'lab':
        y_true_lab = y_test_reg_space
        y_pred_lab = y_pred_reg_space
    else:
        raise ValueError(f"Unknown regression_type: {regression_type}")

    rmse_lab = root_mean_squared_error(y_true_lab, y_pred_lab)
    r2_lab = r2_score(y_true_lab, y_pred_lab)
    
    print(f"Delta Lab (RMSE): {rmse_lab:.4f}")
    print(f"R^2 Score on Test Set (evaluated in LAB space): {r2_lab:.4f}\\n")
    
    return rmse_lab, r2_lab

def main():
    parser = argparse.ArgumentParser(description="Perform simple linear regression on color values.")
    parser.add_argument("--mean", action="store_true", help="Use mean RGB values per pixel before conversion to regression space.")
    parser.add_argument("--type", type=str, default='lab', choices=['lab', 'log_density'], help="Primary regression space for features: 'lab' or 'log_density'.")
    parser.add_argument("--smf", action="store_true", help="Use statsmodels.formula.api OLS.")
    parser.add_argument("--comb", action="store_true", help="Use combined formula (input_channel + log_roundtrip_channel) with SMF OLS. Requires --type lab.")
    args = parser.parse_args()

    if args.comb and not args.smf:
        parser.error("--comb flag requires --smf flag to be specified.")
    if args.comb and args.type != 'lab':
        parser.error("--comb flag requires --type lab for the primary features.")

    regression_method_str = "SMF OLS"
    if args.smf and args.comb:
        regression_method_str += " (Combined Formula)"
    elif not args.smf:
        regression_method_str = "Scikit-learn LinearRegression"
        if args.comb:
             print("Warning: --comb flag specified without --smf. --comb will be ignored for Scikit-learn.")

    print(f"Starting linear regression analysis (Type: {args.type}, Method: {regression_method_str})...")
    if args.mean:
        print("Using --mean flag: data will be averaged per pixel.")
    
    if DATA_PATH is None or GRAYS_DATA is None:
        print("Error: Critical data paths not initialized. Exiting.")
        return

    print("Loading and preparing data...")
    should_calc_log_rt = args.smf and args.comb
    try:
        da_data, n1_data, n2_data, n1_data_log_rt, n2_data_log_rt = load_and_prepare_data(
            DATA_PATH, GRAYS_DATA, 
            use_mean_data=args.mean, 
            regression_type=args.type, 
            calculate_log_roundtrip=should_calc_log_rt
        )
    except FileNotFoundError as e: print(f"Error: Data file not found. {e}"); return
    except ValueError as e: print(f"Error during data preparation: {e}"); return
    except Exception as e: print(f"An unexpected error: {e}"); return

    if da_data.size == 0 or n1_data.size == 0 or n2_data.size == 0:
        print("Error: One or more data arrays are empty. Cannot proceed."); return

    print(f"Data shapes: DA: {da_data.shape}, N1: {n1_data.shape}, N2: {n2_data.shape}")
    if n1_data_log_rt is not None : print(f"N1_log_rt: {n1_data_log_rt.shape}, N2_log_rt: {n2_data_log_rt.shape}")

    if not (da_data.shape[1] == 3 and n1_data.shape[1] == 3 and n2_data.shape[1] == 3 and da_data.shape[0] == n1_data.shape[0] == n2_data.shape[0]):
        print("Error: Data arrays do not have consistent shapes (num_samples x 3)."); return
    if should_calc_log_rt and not (n1_data_log_rt.shape == n1_data.shape and n2_data_log_rt.shape == n2_data.shape):
        print("Error: Log roundtrip data arrays do not match primary data shapes."); return
    if da_data.shape[0] == 0: print("Error: No data points loaded."); return

    rmse1, r2_1, rmse2, r2_2, rmse3, r2_3 = [None]*6

    active_comb_formula = args.smf and args.comb

    print("\nPerforming regression for N1_DATA on DA_DATA...")
    rmse1, r2_1 = perform_linear_regression(n1_data, da_data, "N1", args.type, use_smf_ols=args.smf, X_log_rt=n1_data_log_rt if active_comb_formula else None, use_comb_formula=active_comb_formula)

    print("\nPerforming regression for N2_DATA on DA_DATA...")
    rmse2, r2_2 = perform_linear_regression(n2_data, da_data, "N2", args.type, use_smf_ols=args.smf, X_log_rt=n2_data_log_rt if active_comb_formula else None, use_comb_formula=active_comb_formula)

    print("\nPerforming regression for [N1_DATA or N2_DATA (VStacked)] on DA_DATA...")
    X_vstack = np.vstack((n1_data, n2_data))
    y_vstack = np.vstack((da_data, da_data))
    X_log_rt_vstack = None
    if active_comb_formula and n1_data_log_rt is not None and n2_data_log_rt is not None:
        X_log_rt_vstack = np.vstack((n1_data_log_rt, n2_data_log_rt))
    
    rmse3, r2_3 = perform_linear_regression(X_vstack, y_vstack, "[N1 or N2 (VStacked)]", args.type, use_smf_ols=args.smf, X_log_rt=X_log_rt_vstack, use_comb_formula=active_comb_formula)
    
    print("Linear regression analysis complete.")
    summary_method_detail = regression_method_str
    summary_title = f"--- Summary (Reg. Type: {args.type}, Method: {summary_method_detail}, Metrics in LAB space) ---"
    print(f"\n{summary_title}")
    
    val_na = "N/A"
    rmse1_str = f"{rmse1:.4f}" if rmse1 is not None else val_na
    rmse2_str = f"{rmse2:.4f}" if rmse2 is not None else val_na
    rmse3_str = f"{rmse3:.4f}" if rmse3 is not None else val_na
    r2_1_str = f"{r2_1:.4f}" if r2_1 is not None else val_na
    r2_2_str = f"{r2_2:.4f}" if r2_2 is not None else val_na
    r2_3_str = f"{r2_3:.4f}" if r2_3 is not None else val_na
    
    table = PrettyTable()
    table.field_names = ["Metric", "Test 1 (N1->DA)", "Test 2 (N2->DA)", "Test 3 (N1/N2->DA)"]
    table.add_row(["RMSE (LAB)", rmse1_str, rmse2_str, rmse3_str])
    table.add_row(["R2 (LAB)", r2_1_str, r2_2_str, r2_3_str])
    print(table)

if __name__ == "__main__":
    main() 