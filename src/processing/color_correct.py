#!/usr/bin/env python3

import numpy as np
import cv2
from skimage import img_as_float, img_as_ubyte, exposure

# Import utils from parent package
from src.utils import apply_channel_scaling

class Color_Correction:
    """
    Apply different color correction methods to an image.
    See _validate_method() for available methods.
    """
    
    def __init__(self, method='histogram_equalization'):
        self.method = method
        self._validate_method()
        
        # Defaults
        self.params = {
            'gamma_correction': {
                'gamma': 1.0
            },
            'opencv_tonemap': {
                'opencv_tonemap': 1.0
            }
        }
        
    def _validate_method(self):
        """Validate that the selected method is available"""
        valid_methods = ['histogram_equalization', 'gamma_correction', 
                         'opencv_tonemap']
        
        if self.method not in valid_methods:
            raise ValueError(f"Method '{self.method}' not recognized. Available methods: {valid_methods}")
            
        # Check if OpenCV has the required tonemap module
        if self.method == 'opencv_tonemap' and not hasattr(cv2, 'createTonemapDrago'):
            raise RuntimeError("OpenCV Tonemap module not available. Cannot use 'opencv_tonemap' method.")
    
    def set_method(self, method):
        #Change the color correction method
        self.method = method
        self._validate_method()
        return self
        
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            for method_name, method_params in self.params.items():
                if key in method_params:
                    self.params[method_name][key] = value
                    break
        return self
    
    def apply(self, img):
        match self.method:
            case 'histogram_equalization':
                return self._histogram_equalization(img)
            case 'gamma_correction':
                return self._gamma_correction(img)
            case 'opencv_tonemap':
                return self._opencv_tonemap(img)
            case _:
                raise ValueError(f"Method '{self.method}' not recognized")

    def _histogram_equalization(self, img):
        # Convert to LAB first
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # And equalize only on L
        l_channel = lab[:,:,0]
        l_channel_eq = cv2.equalizeHist(l_channel.astype(np.uint8))
        lab[:,:,0] = l_channel_eq
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _gamma_correction(self, img):
        # gamma < 1 brightens; > 1 darkens
        img_float = img_as_float(img)
        
        gamma = self.params['gamma_correction']['gamma']
        corrected = np.power(img_float, 1/gamma)
        return img_as_ubyte(corrected)

    def _opencv_tonemap(self, img):
        """
        Uses OpenCV's tonemapping algorithms to enhance contrast and colors.
        Good for HDR-like effects and bringing out details in shadows.
        """
        img_hdr = img.astype(np.float32) / 255.0  # Normalize to [0,1] required for tone mapping
        
        tonemap = cv2.createTonemapDrago(2.2, 0.5)
        result = tonemap.process(img_hdr)
        return apply_channel_scaling(result)
