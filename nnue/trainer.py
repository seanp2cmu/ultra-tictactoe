"""
NNUE Training Script

Train NNUE network using data generated from AlphaZero.
Uses knowledge distillation - AlphaZero is the teacher, NNUE is the student.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Tuple
from .network import NNUE
from .data_generator import NNUEDataGenerator


class NNUEDataset(Dataset):
    """PyTorch Dataset for NNUE training data."""
    
    def __init__(self, data: List[Tuple[np.ndarray, float, float]], use_mcts_value: bool = True):
        """
        Args:
            data: List of (features, mcts_value, game_result) tuples
            use_mcts_value: If True, use MCTS value as target; else use game result
        """
        self.features = np.array([d[0] for d in data], dtype=np.float32)
        
        if use_mcts_value:
            self.values = np.array([d[1] for d in data], dtype=np.float32)
        else:
            self.values = np.array([d[2] for d in data], dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.values[idx]


class NNUETrainer:
    """
    Trainer for NNUE network.
    
    Training strategy:
    1. Use MCTS values as primary target (knowledge distillation)
    2. Optionally mix in game results for regularization
    3. Early stopping based on validation loss
    """
    
    def __init__(
        self,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        batch_size: int = 256,
        weight_decay: float = 1e-4
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        self.model = NNUE(device=device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        
        print(f"NNUE Parameters: {self.model.count_parameters():,}")
        print(f"Model Size: {self.model.get_model_size_kb():.1f} KB")
    
    def train(
        self,
        train_data: list,
        val_data: Optional[list] = None,
        epochs: int = 100,
        use_mcts_value: bool = True,
        save_path: Optional[str] = None,
        patience: int = 10
    ) -> dict:
        """
        Train NNUE network.
        
        Args:
            train_data: Training data from NNUEDataGenerator
            val_data: Optional validation data
            epochs: Number of training epochs
            use_mcts_value: Use MCTS values (True) or game results (False)
            save_path: Path to save best model
            patience: Early stopping patience
            
        Returns:
            Training history dict
        """
        # Create datasets
        train_dataset = NNUEDataset(train_data, use_mcts_value)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = None
        if val_data:
            val_dataset = NNUEDataset(val_data, use_mcts_value)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for features, values in train_loader:
                features = features.to(self.device)
                values = values.to(self.device).unsqueeze(-1)
                
                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions, values)
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_loss_str = ""
            if val_loader:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for features, values in val_loader:
                        features = features.to(self.device)
                        values = values.to(self.device).unsqueeze(-1)
                        
                        predictions = self.model(features)
                        loss = self.criterion(predictions, values)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                val_loss_str = f" | Val: {avg_val_loss:.6f}"
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if save_path:
                        self.model.save(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs} | Train: {avg_train_loss:.6f}{val_loss_str}")
        
        # Save final model if no validation
        if save_path and not val_data:
            self.model.save(save_path)
        
        print(f"\n✓ Training complete!")
        print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
        if history['val_loss']:
            print(f"  Best val loss: {best_val_loss:.6f}")
        
        return history
    
    def evaluate(self, test_data: list) -> dict:
        """
        Evaluate model on test data.
        
        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()
        
        dataset = NNUEDataset(test_data, use_mcts_value=True)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        all_preds = []
        all_targets = []
        all_results = []
        
        with torch.no_grad():
            for features, values in loader:
                features = features.to(self.device)
                predictions = self.model(features)
                
                all_preds.extend(predictions.squeeze().cpu().numpy())
                all_targets.extend(values.numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Compute metrics
        mse = np.mean((all_preds - all_targets) ** 2)
        mae = np.mean(np.abs(all_preds - all_targets))
        
        # Correlation
        correlation = np.corrcoef(all_preds, all_targets)[0, 1]
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'num_samples': len(all_preds)
        }


def main():
    """Train NNUE from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NNUE network')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output', type=str, default='nnue.pt', help='Output model path')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split')
    
    args = parser.parse_args()
    
    # Load data
    data = NNUEDataGenerator.load_data(args.data)
    
    # Split into train/val
    n_val = int(len(data) * args.val_split)
    np.random.shuffle(data)
    val_data = data[:n_val]
    train_data = data[n_val:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Train
    trainer = NNUETrainer(
        device=args.device,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        save_path=args.output
    )
    
    # Evaluate
    metrics = trainer.evaluate(val_data)
    print(f"\nValidation Metrics:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Correlation: {metrics['correlation']:.4f}")


if __name__ == '__main__':
    main()
