"""Neural network architecture for Ultimate Tic-Tac-Toe."""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast

from game import Board
from utils import BoardEncoder


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with SE attention."""
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction=16)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)


class Model(nn.Module):
    """AlphaZero-style CNN with dual heads (policy + value)."""
    
    def __init__(self, num_res_blocks: int = 10, num_channels: int = 256) -> None:
        super().__init__()
        self.num_channels = num_channels
        self._device_cache = None
        
        self.input = nn.Sequential(
            nn.Conv2d(7, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 9 * 9, 81)
        
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    @property
    def _device(self):
        if self._device_cache is None:
            self._device_cache = next(self.parameters()).device
        return self._device_cache
    
    def to(self, device):
        self._device_cache = None
        return super().to(device)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.input(x))
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 9 * 9)
        policy = self.policy_fc(policy)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 9 * 9)
        value = F.relu(self.value_fc1(value))
        value = torch.sigmoid(self.value_fc2(value)) 
        
        return policy.clone(), value.clone()
    
    def predict(self, board_state) -> Tuple[np.ndarray, float]:
        """Single board prediction using BoardEncoder."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            # Use BoardEncoder for consistent tensor conversion
            tensor = BoardEncoder.board_to_tensor(board_state)
            board_tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)
            
            if device.type == 'cuda':
                with autocast(device_type='cuda'):
                    policy_logits, value = self.forward(board_tensor)
            else:
                policy_logits, value = self.forward(board_tensor)
            policy_probs = F.softmax(policy_logits, dim=1)
            return policy_probs.cpu().numpy()[0], value.cpu().numpy()[0][0]
    
