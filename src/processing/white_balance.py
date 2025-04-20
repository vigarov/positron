#!/usr/bin/env python3

import numpy as np
import cv2
from skimage import img_as_float
from src.utils import apply_channel_scaling

class White_Balance:
    # Apply several white balance algorithms based on selection
    
    def __init__(self, method='gray_world'):
        self.method = method
        self._validate_method()
        
        # Parameters for different methods
        self.white_patch_percentile = 99
        self.shades_of_gray_p = 6
        
    def _validate_method(self):
        """Validate that the selected method is available"""
        valid_methods = ['gray_world', 'white_patch', 'shades_of_gray', 'retinex', 'opencv']
        
        if self.method not in valid_methods:
            raise ValueError(f"Method '{self.method}' not recognized. Available methods: {valid_methods}")
            
        # Check if OpenCV xphoto is available for 'opencv' method
        if self.method == 'opencv' and not hasattr(cv2, 'xphoto'):
            raise RuntimeError("OpenCV xphoto module not available. Cannot use 'opencv' method.")
    
    def set_method(self, method):
        self.method = method
        self._validate_method()
        return self
        
    def set_params(self, **kwargs):
        """
        Set parameters for white balance methods.
        
        Possible parameters:
        - white_patch_percentile: Percentile value for white patch method
        - shades_of_gray_p: p-norm value for shades of gray method
        """
        for key, value in kwargs.items():
            if key == 'white_patch_percentile':
                self.white_patch_percentile = value
            elif key == 'shades_of_gray_p':
                self.shades_of_gray_p = value
        return self
    
    def apply(self, img):
        """
        Apply the selected white balance algorithm to an image.
        
        Parameters:
        - img: Input image (RGB, uint8)
        
        Returns:
        - White-balanced image (RGB, uint8)
        """
        if self.method == 'gray_world':
            return self._gray_world(img)
        elif self.method == 'white_patch':
            return self._white_patch(img)
        elif self.method == 'shades_of_gray':
            return self._shades_of_gray(img)
        elif self.method == 'retinex':
            return self._retinex_white_balance(img)
        elif self.method == 'opencv':
            return self._opencv_wb(img)
        else:
            # This should never happen because of the validation
            return img

    def _gray_world(self, img):
        """
        Applies the Gray World algorithm for white balance correction.
        Assumes that the average color in a natural image is gray.
        """
        img_float = img_as_float(img)
        
        # Calculate the average values for each channel
        avg_channels = [np.mean(img_float[:,:,i]) for i in range(3)]
        avg_gray = sum(avg_channels) / 3
        
        # Calculate scaling factors
        scale_factors = [avg_gray / avg if avg > 0 else 1.0 for avg in avg_channels]
        
        return apply_channel_scaling(img_float, scale_factors)

    def _white_patch(self, img):
        """
        Applies the White Patch algorithm (Perfect Reflector assumption).
        Assumes that the brightest pixels in the image should be white.
        """
        img_float = img_as_float(img)
        
        # Find the maximum values for each channel (using percentile to avoid outliers)
        max_channels = [np.percentile(img_float[:,:,i], self.white_patch_percentile) for i in range(3)]
        
        # Calculate scaling factors
        scale_factors = [1.0 / max_val if max_val > 0 else 1.0 for max_val in max_channels]
        
        return apply_channel_scaling(img_float, scale_factors)

    def _shades_of_gray(self, img):
        """
        Applies the Shades of Gray algorithm.
        A generalization of the Gray World (p=1) and White Patch (p=∞) methods.
        """
        img_float = img_as_float(img)
        
        # Calculate the p-norm for each channel
        norms = [np.mean(np.power(img_float[:,:,i], self.shades_of_gray_p)) ** (1/self.shades_of_gray_p) for i in range(3)]
        avg_norm = sum(norms) / 3
        
        # Calculate scaling factors
        scale_factors = [avg_norm / norm if norm > 0 else 1.0 for norm in norms]
        
        return apply_channel_scaling(img_float, scale_factors)

    def _retinex_white_balance(self, img):
        """
        Applies a simple implementation of Retinex-based white balancing
        using OpenCV's CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Ensure the L channel is of type CV_8UC1 (8-bit unsigned single-channel)
        l_channel = lab[:,:,0].astype(np.uint8)
        l_channel = clahe.apply(l_channel)
        lab[:,:,0] = l_channel
        
        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result

    def _opencv_wb(self, img):
        """
        Uses OpenCV's automatic white balance algorithm.
        """
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Create a white balance object
        wb = cv2.xphoto.createSimpleWB()
        
        # Apply white balance
        result_bgr = wb.balanceWhite(img_bgr)
        
        # Convert back to RGB
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        # Ensure values are in valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result