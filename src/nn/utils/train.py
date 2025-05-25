import pickle as pkl
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import math

from pathlib import Path
from shutil import rmtree
from collections import defaultdict

from .config import Config
from .constants import ALL_METRICS, REPRO_SEED, MODEL_BASE_DIR, TEXT_SEPARATOR

# Typing
from torch.utils.data import DataLoader,Dataset 
from typing import Callable
import pandas as pd


def validate_model(model,eval_dataloader,device,metrics: list|None=None, print_validation=True):
    model.eval()
    if metrics is None:
        metrics = ALL_METRICS
    gt = []
    preds = []
    with torch.no_grad():
        for batch in (tqdm(eval_dataloader,desc="Evaluating model",total=len(eval_dataloader)) if print_validation else eval_dataloader):
            x,gt = batch['x'].to(device),batch['y'].detach().cpu().numpy().tolist()
            preds = model(x).detach().cpu().numpy().tolist()
    all_res = [metric(gt,preds) for metric in metrics]
    return all_res

def maybe_update_result(current_value,new_result,better="lower"):
    assert better in ["lower","higher"]
    updated = False
    new_value = current_value
    if current_value == -np.inf or (better == "lower" and new_result < current_value) or (better == "higher" and new_result > current_value):
        updated = True
        new_value = new_result
    return updated,new_value
                
def get_scheduler(config, optimizer, train_len):
    epochs = config['epochs']
    warmup_epochs = config['warmup_epochs']
    base_lr = config['base_lr']
    end_lr = config['end_lr']
    scheduler_type = config['scheduler_type']
    if scheduler_type == 'multistep':
        # We do the following technique : high LR in the beginning, low towards the end
        # starting from base_lr we decrease up to e-5, by a factor of 1/sqrt(10) ~0.3162  k times
        fct = 1/np.sqrt(10)
        end_epoch = min(32, epochs)
        num_groups = math.ceil(math.log(end_lr / base_lr, fct))
        group_size = max(1, end_epoch // num_groups)
        milestones = [warmup_epochs + group_size * i for i in range(1, num_groups)]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, fct)
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs - warmup_epochs, 
            eta_min=end_lr
        ) 
    elif scheduler_type == 'exponential':
        gamma = (end_lr / base_lr) ** (1 / (epochs - warmup_epochs))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=8,
            min_lr=end_lr,
        )
    elif scheduler_type == 'one_cycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr * 10,
            total_steps=epochs * train_len,
            pct_start=0.3,
            final_div_factor=base_lr / end_lr
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
def train_predictor(config:Config,df,model_fn:Callable[[Config],nn.Module],
                    dataloader_fn:Callable[[Config,pd.DataFrame,torch.Generator],tuple[DataLoader,DataLoader,Dataset]],
                    override_previous_dir=False,print_validation = True,tqdm_train=True, generator=None) -> tuple[nn.Module,Path,dict,list]:
    # Trains a model
    # When `override_previous_dir` = True, destroy the directory of the previous saved run with the same config id (if it exists)
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    device = torch.device(device)
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(REPRO_SEED)
    train_dataloader, eval_dataloader, test_dataset = dataloader_fn(config,df,generator)
    epochs = config["epochs"]
    model = model_fn(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['base_lr'], eps=1e-6,amsgrad=True)
    loss_fn = torch.nn.MSELoss(reduction="sum") # Minimizing the MSE loss corresponds to minimizing Delta E LAB      

    # LR scheduler
    lr_scheduler = get_scheduler(config, optimizer, len(train_dataloader))
    metrics_to_use = ALL_METRICS
    best_results = {metric:-np.inf for metric in metrics_to_use}
    model_bases = Path(MODEL_BASE_DIR)
    model_bases.mkdir(parents=True,exist_ok=True)
    save_dir = model_bases / config.ident
    if override_previous_dir and save_dir.exists():
        assert save_dir.is_dir(),f"{save_dir.absolute().as_posix()} exists and is not a directory!"
        rmtree(save_dir.absolute().as_posix())
    save_dir.mkdir(parents = False,exist_ok=False)
    for metric in metrics_to_use:
        (save_dir/metric.name).mkdir(parents = False,exist_ok=False)

    all_losses = []
    all_results = defaultdict(list)
    worse_dlab_count = 0

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        model.train()
        current_lr = optimizer.param_groups[0]['lr'] if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else lr_scheduler.get_last_lr()[0]
        train_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d} w/ lr ({current_lr:.6f})", total=len(train_dataloader)) if tqdm_train else train_dataloader
        c = 0
        gl = 0

        # Train
        for batch in train_iterator:
            x,y = batch['x'].to(device),batch['y'].to(device)
            prediction = model(x)
            loss = loss_fn(prediction,y)
            c+=1
            cl = loss.item()
            gl += cl
            all_losses.append(cl)
            if tqdm_train:
                train_iterator.set_postfix({"loss": f"{gl/c:6.3f}"})
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Validate
        print_val = print_validation and (epoch % 20 == 0)
        if print_val:
            print(TEXT_SEPARATOR)
        all_current_results = validate_model(model,eval_dataloader,device,metrics_to_use,print_validation=print_val)
        
        primary_metric_idx = next((i for i, m in enumerate(metrics_to_use) if m.name.lower() == "d_lab"), 0)
        primary_metric_result = all_current_results[primary_metric_idx]
        
        
        for metric,new_metric_result in zip(metrics_to_use,all_current_results):    
            if print_val:
                print(f"{metric.name}: {new_metric_result}\n")
            all_results[metric].append(new_metric_result)
            # Save best models
            curr_metric_res = best_results[metric]
            updated,new_res = maybe_update_result(curr_metric_res,new_metric_result,metric.better_direction)
            best_results[metric] = new_res
            if updated:
                if metric.name.lower() == "d_lab": 
                    worse_dlab_count = 0
                # Save the model
                metric_dir:Path = save_dir/metric.name
                fname:Path = metric_dir/"model.pt"
                if fname.exists():
                    assert fname.is_file()
                    fname.unlink()
                torch.save(model.state_dict(),fname.absolute().as_posix())
                with open((metric_dir/"config.pkl").absolute().as_posix(),"wb") as f:
                    pkl.dump(config,f,pkl.HIGHEST_PROTOCOL)
                # save the loss!
                with open((metric_dir/"loss.txt").absolute().as_posix(),"w") as f:
                    f.write(f"Loss = {new_res:.5f}")
            elif metric.name.lower() == "d_lab": 
                worse_dlab_count += 1
        if print_val:
            print(TEXT_SEPARATOR)
        
        # Update LR
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(primary_metric_result)
        else:
            lr_scheduler.step()

        if worse_dlab_count >=3:
            if print_val:
                print(f"Reached worse D_Lab on validation set three epochs in a row, early stopping the training to avoid overfitting!")
            # break
        
    return model,save_dir, all_results, all_losses, test_dataset