import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from .config import Config
from .constants import REPRO_SEED

class ColorCheckerLabDataset(Dataset):
    def __init__(self, lab_df:pd.DataFrame,keep_one_source = False):
        assert all(col_name in lab_df.columns for col_name in ["DA_LAB","N1_LAB","N2_LAB"])
        n1 = np.array(lab_df["N1_LAB"].tolist())
        n2 = np.array(lab_df["N2_LAB"].tolist())
        da = np.array(lab_df["DA_LAB"].tolist())
        assert n1.shape == n2.shape == da.shape
        assert len(n1.shape) == 2, f"n1.shape: {n1.shape}, n1[0]: {n1[0]}, {type(n1)}, {type(n1[0])}"
        assert n1.shape[1] == 3
        if not keep_one_source:
            self.x = np.hstack([n1,n2]).reshape(-1,3)
            self.y = np.hstack([da,da]).reshape(-1,3)
        else:
            self.x = n1
            self.y = da

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'x':torch.Tensor(self.x[idx]),
                'y':torch.Tensor(self.y[idx])}

class ColorCheckerRGBDataset(Dataset):
    def __init__(self, rgb_df:pd.DataFrame,keep_one_source = False):
        assert all(col_name in rgb_df.columns for col_name in ["DA_RGB","N1_RGB","N2_RGB"])
        n1 = np.array(rgb_df["N1_RGB"].tolist())
        n2 = np.array(rgb_df["N2_RGB"].tolist())
        da = np.array(rgb_df["DA_RGB"].tolist())
        assert n1.shape == n2.shape == da.shape
        assert len(n1.shape) == 2, f"n1.shape: {n1.shape}, n1[0]: {n1[0]}, {type(n1)}, {type(n1[0])}"
        assert n1.shape[1] == 3 # R, G, B
        if not keep_one_source:
            self.x = np.hstack([n1,n2]).reshape(-1,3)
            self.y = np.hstack([da,da]).reshape(-1,3)
        else:
            self.x = n1
            self.y = da

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'x':torch.Tensor(self.x[idx]),
                'y':torch.Tensor(self.y[idx])}

def get_lab_dataloaders(config:Config,df:pd.DataFrame,generator:torch.Generator|None=None) -> tuple[DataLoader,DataLoader,Dataset]:
    tt_split = config["tt_split"]
    if isinstance(tt_split,float):
        tt_split = [tt_split,1-tt_split]
    assert isinstance(tt_split,list), f"{type(tt_split)} {tt_split}"
    assert len(tt_split) == 2
    assert sum(tt_split) == 1
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(REPRO_SEED)
    dataset = ColorCheckerLabDataset(df[["DA_LAB","N1_LAB","N2_LAB"]],keep_one_source=config["keep_one_source"])
    train_dataset, test_dataset = random_split(dataset, tt_split, generator=generator)
    bs = config["bs"]
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    return train_dataloader, test_dataloader, test_dataset

def get_rgb_dataloaders(config:Config,df:pd.DataFrame,generator:torch.Generator|None=None) -> tuple[DataLoader,DataLoader,Dataset]:
    tt_split = config["tt_split"]
    if isinstance(tt_split,float):
        tt_split = [tt_split,1-tt_split]
    assert isinstance(tt_split,list), f"{type(tt_split)} {tt_split}"
    assert len(tt_split) == 2
    assert sum(tt_split) == 1
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(REPRO_SEED)
    # Ensure the input df has the required RGB columns
    dataset = ColorCheckerRGBDataset(df[["DA_RGB","N1_RGB","N2_RGB"]],keep_one_source=config["keep_one_source"])
    train_dataset, test_dataset = random_split(dataset, tt_split, generator=generator)
    bs = config["bs"]
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    return train_dataloader, test_dataloader, test_dataset
