"""NNUE model for Ultimate Tic-Tac-Toe.

Architecture (Stockfish NNUE style):
  Input (199 sparse) → Accumulator (199→256, shared) → ClippedReLU
  Concat STM + NSTM (512) → Linear 512→32 → CReLU → Linear 32→32 → CReLU → Linear 32→1 → tanh
"""
import torch
import torch.nn as nn
import numpy as np

from nnue.core.features import NUM_FEATURES, extract_features, features_to_tensor


class NNUE(nn.Module):
    """NNUE network with shared perspective accumulator."""
    
    def __init__(self, accumulator_size=256, hidden1_size=32, hidden2_size=32):
        super().__init__()
        self.accumulator_size = accumulator_size
        
        # Shared accumulator (same weights for both perspectives)
        self.accumulator = nn.Linear(NUM_FEATURES, accumulator_size)
        
        # Final layers (input = concat of both perspectives)
        self.fc1 = nn.Linear(accumulator_size * 2, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)
    
    def forward(self, stm_features, nstm_features):
        """Forward pass.
        
        Args:
            stm_features: (batch, NUM_FEATURES) side-to-move features
            nstm_features: (batch, NUM_FEATURES) not-side-to-move features
            
        Returns:
            (batch, 1) evaluation in [-1, 1]
        """
        # Accumulator + ClippedReLU
        acc_stm = torch.clamp(self.accumulator(stm_features), 0.0, 1.0)
        acc_nstm = torch.clamp(self.accumulator(nstm_features), 0.0, 1.0)
        
        # Concat and final layers
        x = torch.cat([acc_stm, acc_nstm], dim=1)
        x = torch.clamp(self.fc1(x), 0.0, 1.0)
        x = torch.clamp(self.fc2(x), 0.0, 1.0)
        x = torch.tanh(self.fc3(x))
        return x
    
    def evaluate(self, board):
        """Evaluate a single board position.
        
        Args:
            board: Board object
            
        Returns:
            float: evaluation from current player's perspective [-1, 1]
        """
        stm_idx, nstm_idx = extract_features(board)
        stm_t, nstm_t = features_to_tensor(stm_idx, nstm_idx)
        
        device = next(self.parameters()).device
        stm_t = torch.from_numpy(stm_t).unsqueeze(0).to(device)
        nstm_t = torch.from_numpy(nstm_t).unsqueeze(0).to(device)
        
        with torch.no_grad():
            value = self.forward(stm_t, nstm_t)
        
        return value.item()
    
    def save(self, path):
        """Save model weights."""
        torch.save({
            'state_dict': self.state_dict(),
            'accumulator_size': self.accumulator_size,
            'hidden1_size': self.fc1.out_features,
            'hidden2_size': self.fc2.out_features,
        }, path)
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            accumulator_size=checkpoint['accumulator_size'],
            hidden1_size=checkpoint['hidden1_size'],
            hidden2_size=checkpoint['hidden2_size'],
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        return model
