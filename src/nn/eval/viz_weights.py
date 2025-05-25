import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

def visualize_layer_weights(model: nn.Module, layer_name: str):
    # Get the specified layer's weights
    layer = getattr(model, layer_name, None)
    if layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")
        
    weights = layer.weight.data.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(weights)
    plt.colorbar(im, ax=ax)
    
    ax.set_xlabel("Input Feature Index (L,a,b channels)")
    ax.set_ylabel("Output Feature Index (L,a,b channels)")
    
    ax.set_xticks(np.arange(weights.shape[1]))
    ax.set_xticklabels([f"{i}" for i in range(weights.shape[1])])
    ax.set_yticks(np.arange(weights.shape[0]))
    ax.set_yticklabels([f"{i}" for i in range(weights.shape[0])])
    
    ax.set_title(f"Weight Matrix for {layer_name}")
    
    return fig, ax
