"""NNUE training loop."""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nnue.core.model import NNUE
from nnue.training.dataset import NNUEDataset
from nnue.config import NNUEConfig, TrainingConfig


def train_nnue(data_path, output_path='nnue_model.pt', 
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
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    boards, values = data['boards'], data['values']
    print(f"  Positions: {len(values)}")
    print(f"  Value range: [{values.min():.3f}, {values.max():.3f}]")
    
    # Split
    train_ds, val_ds = NNUEDataset.train_val_split(boards, values)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, 
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=train_config.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    
    # Model
    model = NNUE(
        accumulator_size=net_config.accumulator_size,
        hidden1_size=net_config.hidden1_size,
        hidden2_size=net_config.hidden2_size,
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {param_count:,} parameters")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(train_config.num_epochs):
        t0 = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        for stm, nstm, target in train_loader:
            stm, nstm, target = stm.to(device), nstm.to(device), target.to(device)
            
            pred = model(stm, nstm).squeeze(-1)
            loss = criterion(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for stm, nstm, target in val_loader:
                stm, nstm, target = stm.to(device), nstm.to(device), target.to(device)
                pred = model(stm, nstm).squeeze(-1)
                loss = criterion(pred, target)
                val_loss += loss.item()
                val_batches += 1
        
        train_loss /= max(1, train_batches)
        val_loss /= max(1, val_batches)
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch+1}/{train_config.num_epochs} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"Time: {elapsed:.1f}s")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(output_path)
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")
    
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    return model
