import numpy as np
from src.utils import sigmoid
from src.processing.white_balance import White_Balance

class SigmoidCorrect:
    def __init__(self, config:dict):
        """
        Initializes the SigmoidCorrect processor.

        Args:
            correction_mode (str): One of 'center_mean', 'stretch_percentiles', 'dual_percentile_wb_stretch'.
            **config: Configuration dictionary for the selected mode and white balance.
                Common WB keys (used by White_Balance.set_params):
                - 'wb_method': (str) e.g., 'gray_world', 'percentile_reference_balance'. Defaults to 'gray_world'.
                - 'white_patch_percentile': (float)
                - 'shades_of_gray_p': (float)
                - 'percentile_ref_lower': (float)
                - 'percentile_ref_upper': (float)
                - 'percentile_ref_target_point': (str) 'white' or 'black'

                Common SigmoidCorrect keys (used based on correction_mode):
                - 'map_to_z': (float) Defines the target range for sigmoid input (e.g., [-map_to_z, map_to_z]). Default 4.0.
                - 'lumi_factor': (float) Luminance adjustment. Default 0.0.
                
                Mode-specific SigmoidCorrect keys:
                For 'center_mean':
                - 'center_loc': (float) Target mean location after centering. Default 0.0.
                - 'center_upper_percentile': (float) Percentile for scaling. Default 99.0.
                For 'stretch_percentiles' (for the stretching part):
                - 'stretch_lower_percentile': (float) Lower percentile for contrast stretching. Default 1.0.
                - 'stretch_upper_percentile': (float) Upper percentile for contrast stretching. Default 99.0.
        """
        self.correction_mode = config.get("correction_mode","center_mean")
        self._validate_correction_mode()
        wb_method = config.get('wb_method', 'gray_world')
        self.white_balancer = White_Balance(method=wb_method)
        self.white_balancer.set_params(**config)

        self.map_to_z = float(config.get('map_to_z', 4.0))
        self.lumi_factor = float(config.get('lumi_factor', 0.0)) # Used by stretch modes

        if self.correction_mode == 'center_mean':
            self.center_loc = float(config.get('center_loc', 0.0))
            self.center_upper_percentile = float(config.get('center_upper_percentile', 99.0))
        
        elif self.correction_mode == 'stretch_percentiles':
            self.stretch_lower_percentile = float(config.get('stretch_lower_percentile', 1.0))
            self.stretch_upper_percentile = float(config.get('stretch_upper_percentile', 99.0))

        self.rotate_image = config.get('rotate_image', True)

    def _validate_correction_mode(self):
        valid_modes = ['center_mean', 'stretch_percentiles']
        if self.correction_mode not in valid_modes:
            raise ValueError(f"Unknown correction_mode: {self.correction_mode}. Valid modes are {valid_modes}")

    def apply(self, input_image_float):
        # Input image range [0, 1]
        # Returns processed image as numpy array, float, range [0, 1]
        if not (input_image_float.dtype == np.float32 or input_image_float.dtype == np.float64):
            raise ValueError("Input image must be a float32 or float64 numpy array.")
        if np.min(input_image_float) < -1e-5 or np.max(input_image_float) > 1.0 + 1e-5: # Allow small epsilon
            print(f"Warning: Input image values are outside the expected [0,1] range. Min: {np.min(input_image_float)}, Max: {np.max(input_image_float)}")
        mean_g_before_wb = np.mean(input_image_float,axis=(0,1))[1]
        img_wb = self.white_balancer.apply(input_image_float) 

        if self.correction_mode == 'center_mean':
            out = self._apply_center_mean(img_wb, mean_g_before_wb)
        elif self.correction_mode == 'stretch_percentiles':
            out = self._apply_stretch_percentiles(img_wb)
        else:
            raise ValueError(f"Unknown correction_mode: {self.correction_mode}")

        if self.rotate_image:
            out = np.rot90(out.transpose(1,0,2), k=1)
        return out

    def _apply_center_mean(self, img_wb_float, mean_g_before_wb):
        input_image_centered = img_wb_float - mean_g_before_wb + self.center_loc
        
        all_values_abs = np.abs(input_image_centered).reshape(-1)
        p_high = np.percentile(all_values_abs, self.center_upper_percentile)
        p_high = max(p_high, 1e-5) 

        input_image_scaled = -input_image_centered * (self.map_to_z / p_high)
        
        sigmoid_corrected = sigmoid(input_image_scaled)
        return np.clip(sigmoid_corrected, 0.0, 1.0)

    def _apply_stretch_percentiles(self, img_wb_float,alternative_method=False):
        all_values = img_wb_float.reshape(-1)
        p_high = np.percentile(all_values, self.stretch_upper_percentile)
        p_low = np.percentile(all_values, self.stretch_lower_percentile)
        
        # Ensure p_high is greater than p_low to avoid division by zero or negative range.
        if p_high <= p_low:
            print(f"Warning: p_high <= p_low. Setting p_high to p_low + 1e-5. p_high: {p_high}, p_low: {p_low}")
            p_high = p_low + 1e-5 

        if alternative_method:
            # Map [p_low, p_high] to [-self.map_to_z, +self.map_to_z], then add lumi_factor
            if (p_high - p_low) < 1e-5:
                input_image_centered = img_wb_float - np.mean(img_wb_float) + self.lumi_factor
            else:
                # Scale to [0, 2 * map_to_z]
                input_image_scaled = (img_wb_float - p_low) * (2.0 * self.map_to_z) / (p_high - p_low) 
                # Shift to [-map_to_z, map_to_z] and add lumi_factor
                input_image_centered = input_image_scaled - self.map_to_z + self.lumi_factor
        else:
            input_image_stretched = img_wb_float * ( 2.0 * self.map_to_z / p_high)
            input_image_centered = input_image_stretched - 4.0 - p_low * self.map_to_z + self.lumi_factor
        
        sigmoid_corrected = sigmoid(-input_image_centered)
        return np.clip(sigmoid_corrected, 0.0, 1.0)

    def __call__(self, input_image_float):
        return self.apply(input_image_float)
