import torch
import torch.nn as nn
from ..utils import torch_lab2rgb, torch_rgb2lab, Config

class EnhCombHiddenLayerNN(nn.Module):
    def __init__(self,config:Config):
        super().__init__()

        self.lab_seq = nn.Sequential(
            nn.Linear(3,config['hidden_size']),
            nn.Tanh(),
            nn.Linear(config['hidden_size'],3)
        )
        self.lab_linear = nn.Linear(3,3)
        self.lab_comb = nn.Linear(6,3)

        self.log_dens_linear = nn.Linear(3,3)
        self.final_comb = nn.Linear(6,3)

    def forward(self,x):
        seq = self.lab_seq(x)
        lin = self.lab_linear(x)
        stacked = torch.hstack([lin,seq])#.reshape(-1,6)
        final_lab = self.lab_comb(stacked)

        x_rgb = torch_lab2rgb(x)
        log_density = -torch.log10(x_rgb)
        model_processed = self.log_dens_linear(log_density)
        back_to_lab = torch_rgb2lab(torch.pow(10,model_processed))
        return self.final_comb(torch.hstack([final_lab,back_to_lab]))

def get_enhcomb_model(config:Config):
    return EnhCombHiddenLayerNN(config)

def get_enhanced_comb_model_config(model_name = None, epochs=100):
    return Config({
            'tt_split': 0.8, # Train-test split
            'bs': 1, # Training batch size --> 1 = GD (not SGD)
            'base_lr': 1*(10**-1), # Starting learning rate,
            'end_lr': 1*(10**-4), # Smallest lr you will converge to
            'epochs': epochs, # epochs
            'warmup_epochs': 2, # number of warmup epochs
            'hidden_size': 10,
            "keep_one_source": True,
            "scheduler_type": "reduce_on_plateau",
            "keep_one_source":True,
            },name="enhcombgr3_tanh_hidden" if model_name is None else model_name,name_features=["hidden_size","keep_one_source","scheduler_type","epochs"])