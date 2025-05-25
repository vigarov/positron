import numpy as np

def apply_channel_scaling(img, scale_factors=None):
    # Some images are not in float (caused errors in underlying processing functions like cv2)
    # --> make sure to convert them here so scaling by non-int works as expected
    result = img.copy()
    if scale_factors is None:
        scale_factors = [1.0, 1.0, 1.0]
    
    for i, scale in enumerate(scale_factors):
        result[:,:,i] = result[:,:,i] * scale
    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
