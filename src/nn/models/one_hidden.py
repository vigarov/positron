import torch.nn as nn
from utils import *

class OneHiddenLayerNN(nn.Module):
    def __init__(self,config:Config):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(3,config['hidden_size']),
            nn.Tanh(),
            nn.Linear(config['hidden_size'],3)
        )
        self.linear = nn.Linear(3,3)

    def forward(self,x):
        seq = self.seq(x)
        lin = self.linear(x)
        return lin + seq
    
def get_linear_model(config:Config):
    return OneHiddenLayerNN(config)

def get_linear_model_config(model_name=None, epochs=100):
    return Config({
            'tt_split': 0.8, # Train-test split
            'bs': 1, # Training batch size --> 1 = GD (not SGD)
            'base_lr': 1*(10**-1), # Starting learning rate,
            'end_lr': 1*(10**-4), # Smallest lr you will converge to
            'epochs': epochs, # epochs
            'warmup_epochs': 2, # number of warmup epochs
            'hidden_size': 15,
            "keep_one_source": True,
            "scheduler_type": "reduce_on_plateau",
            "keep_one_source":True,
            },name="gr3_tanh_hidden" if model_name is None else model_name,name_features=["hidden_size","keep_one_source","scheduler_type","epochs"])
