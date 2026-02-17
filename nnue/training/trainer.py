"""NNUE training loop with AMP, LR scheduling, and gradient clipping."""
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

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
    
    Args:
        data_path: Path to .npz training data
        output_path: Path to save trained model
        net_config: NNUEConfig (network architecture)
        train_config: TrainingConfig (training hyperparams)
        device: torch device
    """
    net_config = net_config or NNUEConfig()
    train_config = train_config or TrainingConfig()
    use_amp = train_config.use_amp and device != 'cpu'
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    boards, values = data['boards'], data['values']
    print(f"  Positions: {len(values):,}")
    print(f"  Value range: [{values.min():.3f}, {values.max():.3f}]")
    
    # Split
    t0 = time.time()
    train_ds, val_ds = NNUEDataset.train_val_split(boards, values)
    print(f"  Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    print(f"  Feature extraction: {time.time() - t0:.1f}s")
    
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
    print(f"\nModel: {param_count:,} parameters")
    print(f"AMP: {use_amp}, LR schedule: {train_config.lr_schedule}, "
          f"grad_clip: {train_config.grad_clip}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=use_amp)
    scheduler = _build_scheduler(optimizer, train_config, len(train_loader))
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(train_config.num_epochs):
        t0 = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for stm, nstm, target in train_loader:
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
        
        print(f"Epoch {epoch+1:>3}/{train_config.num_epochs} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {lr_now:.2e} | Time: {elapsed:.1f}s")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            model.save(output_path)
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")
    
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    return model, best_val_loss
