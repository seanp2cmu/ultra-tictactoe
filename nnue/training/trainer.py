"""NNUE training loop with AMP, LR scheduling, gradient clipping, and tqdm."""
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from nnue.core.model import NNUE
from nnue.training.dataset import NNUEDataset
from nnue.config import NNUEConfig, TrainingConfig


def _build_scheduler(optimizer, config, steps_per_epoch):
    """Build LR scheduler with linear warmup + cosine decay."""
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.num_epochs * steps_per_epoch
    
    if config.lr_schedule == "none":
        return None
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_nnue(data_path, output_path='nnue/model/nnue_model.pt', 
               net_config=None, train_config=None, device='cuda'):
    """Train NNUE model from .npz data.
    
    Returns:
        (model, best_val_loss, epoch_metrics_list)
    """
    net_config = net_config or NNUEConfig()
    train_config = train_config or TrainingConfig()
    use_amp = train_config.use_amp and device != 'cpu'
    
    # Load data
    data = np.load(data_path)
    boards, values = data['boards'], data['values']
    
    # Split
    t0 = time.time()
    train_ds, val_ds = NNUEDataset.train_val_split(boards, values)
    feat_time = time.time() - t0
    
    train_loader = DataLoader(
        train_ds, batch_size=train_config.batch_size,
        shuffle=True, num_workers=train_config.num_workers,
        pin_memory=True, persistent_workers=train_config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_config.batch_size,
        shuffle=False, num_workers=train_config.num_workers,
        pin_memory=True, persistent_workers=train_config.num_workers > 0,
    )
    
    # Model
    model = NNUE(
        accumulator_size=net_config.accumulator_size,
        hidden1_size=net_config.hidden1_size,
        hidden2_size=net_config.hidden2_size,
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=use_amp)
    scheduler = _build_scheduler(optimizer, train_config, len(train_loader))
    
    # Info bar
    tqdm.write(f"  Data: {len(train_ds):,} train, {len(val_ds):,} val "
               f"(feat: {feat_time:.1f}s) | Model: {param_count:,} params | "
               f"AMP={use_amp}")
    
    # Training loop
    best_val_loss = float('inf')
    all_metrics = []
    
    epoch_pbar = tqdm(range(train_config.num_epochs), desc="Train",
                      ncols=100, leave=False)
    
    for epoch in epoch_pbar:
        t0 = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        batch_pbar = tqdm(train_loader, desc=f"Ep {epoch+1}",
                          ncols=90, leave=False)
        for stm, nstm, target in batch_pbar:
            stm = stm.to(device, non_blocking=True)
            nstm = nstm.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            with autocast(enabled=use_amp):
                pred = model(stm, nstm).squeeze(-1)
                loss = criterion(pred, target)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            if train_config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()
            
            train_loss += loss.item()
            train_batches += 1
            batch_pbar.set_postfix(loss=f"{loss.item():.5f}")
        batch_pbar.close()
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for stm, nstm, target in val_loader:
                stm = stm.to(device, non_blocking=True)
                nstm = nstm.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                with autocast(enabled=use_amp):
                    pred = model(stm, nstm).squeeze(-1)
                    loss = criterion(pred, target)
                
                val_loss += loss.item()
                val_batches += 1
        
        train_loss /= max(1, train_batches)
        val_loss /= max(1, val_batches)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        
        # Epoch metrics
        metrics = {
            'train/loss': train_loss,
            'train/val_loss': val_loss,
            'train/lr': lr_now,
            'train/epoch_time_s': elapsed,
            'train/epoch': epoch + 1,
        }
        all_metrics.append(metrics)
        
        # Update outer progress bar
        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            model.save(output_path)
            saved = " ★"
        
        epoch_pbar.set_postfix_str(
            f"loss={train_loss:.5f} val={val_loss:.5f} lr={lr_now:.1e}{saved}"
        )
    
    epoch_pbar.close()
    tqdm.write(f"  Best val_loss: {best_val_loss:.6f}")
    
    return model, best_val_loss, all_metrics
