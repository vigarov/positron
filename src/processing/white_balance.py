#!/usr/bin/env python3

import numpy as np
import cv2
from skimage import img_as_float, img_as_ubyte
from src.utils import apply_channel_scaling

class White_Balance:
    # Apply several white balance algorithms based on selection
    
    def __init__(self, method='gray_world'):
        self.__method = method
        self.white_patch_percentile = 99.0
        self.shades_of_gray_p = 6.0
        
        # Parameters for percentile_reference_balance
        self.percentile_ref_lower = 1.0
        self.percentile_ref_upper = 99.0
        self.percentile_ref_target_point = 'white'
        
        self._validate_method()
        
    # Validate that the selected method is available
    def _validate_method(self):
        valid_methods = ['gray_world', 'white_patch', 'shades_of_gray', 
                         'retinex', 'opencv', 'scale_to_green_mean', 
                         'percentile_reference_balance']
        
        if self.__method not in valid_methods:
            raise ValueError(f"Method '{self.__method}' not recognized. Available methods: {valid_methods}")
            
        if self.__method == 'opencv' and not hasattr(cv2, 'xphoto'):
            raise RuntimeError("OpenCV xphoto module not available. Cannot use 'opencv' method.")

    def set_method(self, method):
        self.__method = method
        self._validate_method()
        return self
        
    # Set parameters for white balance methods
    # Possible parameters: white_patch_percentile, shades_of_gray_p, percentile_ref_lower, percentile_ref_upper, percentile_ref_target_point
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'white_patch_percentile':
                self.white_patch_percentile = float(value)
            elif key == 'shades_of_gray_p':
                self.shades_of_gray_p = float(value)
            elif key == 'percentile_ref_lower':
                self.percentile_ref_lower = float(value)
            elif key == 'percentile_ref_upper':
                self.percentile_ref_upper = float(value)
            elif key == 'percentile_ref_target_point':
                if value not in ['white', 'black']:
                    raise ValueError("percentile_ref_target_point must be 'white' or 'black'")
                self.percentile_ref_target_point = value
        return self
    
    def apply(self, img):
        if self.__method == 'gray_world':
            balanced_img_float = self._gray_world(img)
        elif self.__method == 'white_patch':
            balanced_img_float = self._white_patch(img)
        elif self.__method == 'shades_of_gray':
            balanced_img_float = self._shades_of_gray(img)
        elif self.__method == 'retinex':
            balanced_img_uint8 = self._retinex_white_balance(img)
            balanced_img_float = img_as_float(balanced_img_uint8)
        elif self.__method == 'opencv':
            balanced_img_uint8 = self._opencv_wb(img)
            balanced_img_float = img_as_float(balanced_img_uint8)
        elif self.__method == 'scale_to_green_mean':
            balanced_img_float = self._scale_to_green_mean(img)
        elif self.__method == 'percentile_reference_balance':
            balanced_img_float = self._percentile_reference_balance(img)
        else:
            balanced_img_float = img_as_float(img)
        return balanced_img_float

    def __call__(self, img):
        return self.apply(img)

    def _gray_world(self, img):
        # Calculate the average values for each channel
        avg_channels = [np.mean(img[:,:,i]) for i in range(3)]
        avg_gray = sum(avg_channels) / 3.0
        
        scale_factors = [avg_gray / avg if avg > 1e-6 else 1.0 for avg in avg_channels]
        
        return apply_channel_scaling(img, scale_factors)

    def _white_patch(self, img):
        max_channels = [np.percentile(img[:,:,i], self.white_patch_percentile) for i in range(3)]
        scale_factors = [1.0 / max_val if max_val > 1e-6 else 1.0 for max_val in max_channels]
        return apply_channel_scaling(img, scale_factors)

    def _shades_of_gray(self, img):
        p_float = float(self.shades_of_gray_p)
        norms = [np.mean(np.power(img[:,:,i], p_float)) ** (1.0/p_float) for i in range(3)]
        avg_norm = sum(norms) / 3.0
        
        scale_factors = [avg_norm / norm if norm > 1e-6 else 1.0 for norm in norms]
        
        return apply_channel_scaling(img, scale_factors)

    def _retinex_white_balance(self, img_uint8):
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Ensure the L channel is of type CV_8UC1 (8-bit unsigned single-channel)
        l_channel = lab[:,:,0]
        l_channel = clahe.apply(l_channel)
        lab[:,:,0] = l_channel
        
        # Convert back to RGB
        result_uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result_uint8

    def _opencv_wb(self, img_uint8):
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        
        wb = cv2.xphoto.createSimpleWB()
        result_bgr = wb.balanceWhite(img_bgr)
        result_rgb_uint8 = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_rgb_uint8 = np.clip(result_rgb_uint8, 0, 255).astype(np.uint8)
        return result_rgb_uint8

    def _scale_to_green_mean(self, img):
        means = np.mean(img,axis=(0,1))
        
        mean_r = means[0]
        mean_g = means[1]
        mean_b = means[2]

        scale_r = mean_g / mean_r
        scale_b = mean_g / mean_b
        
        scale_factors = [scale_r, 1.0, scale_b] # [scale_r, 1.0, scale_b]
        return apply_channel_scaling(img, scale_factors)

    def _percentile_reference_balance(self, img):
        if self.percentile_ref_target_point == 'white':
            percentile_val = self.percentile_ref_upper
        else:
            percentile_val = self.percentile_ref_lower
            
        channel_percentiles = [np.percentile(img[:,:,i], percentile_val) for i in range(3)]
        
        perc_r = channel_percentiles[0] if channel_percentiles[0] > 1e-6 else 1.0
        perc_g = channel_percentiles[1]
        perc_b = channel_percentiles[2] if channel_percentiles[2] > 1e-6 else 1.0

        scale_r = perc_g / perc_r
        scale_b = perc_g / perc_b
        
        scale_factors = [scale_r, 1.0, scale_b]
        return apply_channel_scaling(img, scale_factors)