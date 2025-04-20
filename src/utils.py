import numpy as np
from skimage import img_as_ubyte, img_as_float

def apply_channel_scaling(img, scale_factors=None):
    # Some images are not in float (caused errors in underlying processing functions like cv2)
    # --> make sure to convert them here so scaling by non-int works as expected
    img_float = img if img.dtype == np.float32 or img.dtype == np.float64 else img_as_float(img)
    result = np.zeros_like(img_float)
    
    if scale_factors is None:
        scale_factors = [1.0, 1.0, 1.0]
    
    for i, scale in enumerate(scale_factors):
        result[:,:,i] = np.clip(img_float[:,:,i] * scale, 0, 1)
    
    return img_as_ubyte(result) 