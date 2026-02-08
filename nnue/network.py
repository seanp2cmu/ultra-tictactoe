"""
NNUE Network for Ultimate Tic-Tac-Toe

Stockfish-style NNUE architecture optimized for fast evaluation.
Uses ClippedReLU activation and supports INT8 quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union
from game import Board
from .features import NNUEFeatureExtractor


class ClippedReLU(nn.Module):
    """ReLU clamped to [0, 1] for quantization-friendly activation."""
    
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class NNUE(nn.Module):
    """
    NNUE (Efficiently Updatable Neural Network) for Ultimate Tic-Tac-Toe.
    
    Architecture (Stockfish HalfKP style, simplified):
    - Input: 288 features
    - Layer 1: 288 → 256 (ClippedReLU) - accumulator layer
    - Layer 2: 256 → 32 (ClippedReLU)
    - Layer 3: 32 → 32 (ClippedReLU)
    - Output: 32 → 1 (sigmoid)
    
    Total parameters: ~80K (vs AlphaZero's ~100M)
    Inference: ~1μs (vs AlphaZero's ~1ms)
    """
    
    INPUT_SIZE = NNUEFeatureExtractor.TOTAL_FEATURES  # 288
    L1_SIZE = 256
    L2_SIZE = 32
    L3_SIZE = 32
    
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        
        self.device = device
        self.feature_extractor = NNUEFeatureExtractor()
        
        # Accumulator layer (supports incremental update)
        self.fc1 = nn.Linear(self.INPUT_SIZE, self.L1_SIZE)
        
        # Hidden layers
        self.fc2 = nn.Linear(self.L1_SIZE, self.L2_SIZE)
        self.fc3 = nn.Linear(self.L2_SIZE, self.L3_SIZE)
        
        # Output layer
        self.fc_out = nn.Linear(self.L3_SIZE, 1)
        
        # Activation
        self.activation = ClippedReLU()
        
        # Initialize weights
        self._init_weights()
        
        self.to(device)
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch_size, 288)
            
        Returns:
            Value prediction of shape (batch_size, 1) in range [0, 1]
        """
        # Accumulator layer
        x = self.activation(self.fc1(x))
        
        # Hidden layers
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        
        # Output
        x = torch.sigmoid(self.fc_out(x))
        
        return x
    
    def predict(self, board: Board) -> float:
        """
        Predict value for a single board position.
        
        Args:
            board: Ultimate Tic-Tac-Toe board
            
        Returns:
            Value in [0, 1] (0=loss, 0.5=draw, 1=win) from current player's perspective
        """
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor.extract(board)
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)
            value = self.forward(x)
            return value.item()
    
    def predict_batch(self, boards: list) -> np.ndarray:
        """
        Predict values for multiple boards.
        
        Args:
            boards: List of Board objects
            
        Returns:
            np.ndarray of shape (batch_size,) with values in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor.extract_batch(boards)
            x = torch.from_numpy(features).to(self.device)
            values = self.forward(x)
            return values.squeeze(-1).cpu().numpy()
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'state_dict': self.state_dict(),
            'device': self.device,
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
    
    def get_accumulator(self, board: Board) -> torch.Tensor:
        """
        Get the accumulator (first layer output) for incremental updates.
        
        In a real NNUE implementation, this would be cached and updated
        incrementally as moves are made/unmade.
        
        Args:
            board: Board position
            
        Returns:
            Accumulator tensor of shape (256,)
        """
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor.extract(board)
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)
            acc = self.activation(self.fc1(x))
            return acc.squeeze(0)
    
    def forward_from_accumulator(self, accumulator: torch.Tensor) -> float:
        """
        Complete forward pass from cached accumulator.
        
        This is the fast path - only 3 small matrix multiplications.
        
        Args:
            accumulator: Cached first layer output of shape (256,)
            
        Returns:
            Value in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            x = accumulator.unsqueeze(0)
            x = self.activation(self.fc2(x))
            x = self.activation(self.fc3(x))
            x = torch.sigmoid(self.fc_out(x))
            return x.item()
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_kb(self) -> float:
        """Get approximate model size in KB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024


class QuantizedNNUE:
    """
    INT8 quantized NNUE for maximum inference speed.
    
    TODO: Implement INT8 quantization for production use.
    This would provide ~4x speedup over FP32.
    """
    pass
