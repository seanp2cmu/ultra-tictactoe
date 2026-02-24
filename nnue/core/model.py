"""NNUE model for Ultimate Tic-Tac-Toe.

Architecture (Stockfish SFNNv5+ style):
  Input (199 sparse) → Accumulator (199 → acc_size+1, shared)
    ├── acc_size hidden → SCReLU (squared clipped relu)
    └── 1 PSQT output (direct to final output)
  Concat STM+NSTM (acc_size*2) → LayerStack[bucket] (→h1→CReLU→h2→CReLU→1)
  output = layer_stack_result + (psqt_stm - psqt_nstm) / 2

Key improvements over v1:
  - SCReLU: clamp(x,0,1)² on accumulator — stronger nonlinearity
  - PSQT: 1 extra accumulator neuron bypasses layer stack — captures gross eval
  - Layer Stack Buckets: different weights per game phase (piece count buckets)
  - Raw output: no tanh, use sigmoid WDL loss externally
"""
import torch
import torch.nn as nn
import numpy as np

from nnue.core.features import NUM_FEATURES, extract_features, features_to_tensor


def _count_pieces_from_features(stm_features):
    """Count filled cells from dense feature tensor for bucket selection.
    
    Features [0-80] = my pieces, [81-161] = opponent pieces.
    Returns tensor of piece counts per batch element.
    """
    my_pieces = stm_features[:, :81].sum(dim=1)
    opp_pieces = stm_features[:, 81:162].sum(dim=1)
    return (my_pieces + opp_pieces).long()


class LayerStack(nn.Module):
    """Multiple layer stacks selected by bucket index (Stockfish SFNNv4+).
    
    Evaluates all stacks in parallel, then gathers the correct one per sample.
    """
    
    def __init__(self, input_size, hidden1_size, hidden2_size, num_buckets):
        super().__init__()
        self.num_buckets = num_buckets
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        
        self.fc1 = nn.Linear(input_size, hidden1_size * num_buckets)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size * num_buckets)
        self.fc3 = nn.Linear(hidden2_size, 1 * num_buckets)
    
    def forward(self, x, bucket_indices):
        """Forward pass with bucket selection.
        
        Args:
            x: (batch, input_size)
            bucket_indices: (batch,) int tensor, values in [0, num_buckets)
        Returns:
            (batch, 1)
        """
        batch = x.shape[0]
        nb = self.num_buckets
        
        # Precompute offset for gather: nb * i + bucket
        offset = torch.arange(0, batch * nb, nb, device=x.device)
        indices = bucket_indices + offset
        
        # fc1 → gather → CReLU
        h1_all = self.fc1(x)                                    # (batch, h1*nb)
        h1_sel = h1_all.view(-1, self.hidden1_size)[indices]     # (batch, h1)
        h1 = torch.clamp(h1_sel, 0.0, 1.0)
        
        # fc2 → gather → CReLU
        h2_all = self.fc2(h1)                                    # (batch, h2*nb)
        h2_sel = h2_all.view(-1, self.hidden2_size)[indices]     # (batch, h2)
        h2 = torch.clamp(h2_sel, 0.0, 1.0)
        
        # fc3 → gather
        out_all = self.fc3(h2)                                   # (batch, 1*nb)
        out = out_all.view(-1, 1)[indices]                       # (batch, 1)
        
        return out.unsqueeze(-1)


class NNUE(nn.Module):
    """NNUE network with SCReLU, PSQT, and Layer Stack Buckets."""
    
    def __init__(self, accumulator_size=256, hidden1_size=32, hidden2_size=32,
                 num_buckets=4, bucket_divisor=20):
        super().__init__()
        self.accumulator_size = accumulator_size
        self.num_buckets = num_buckets
        self.bucket_divisor = bucket_divisor
        
        # Shared accumulator: acc_size hidden + 1 PSQT
        self.accumulator = nn.Linear(NUM_FEATURES, accumulator_size + 1)
        
        # Layer stack with buckets
        self.layer_stack = LayerStack(
            input_size=accumulator_size * 2,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            num_buckets=num_buckets,
        )
    
    def _get_buckets(self, stm_features):
        """Compute bucket indices from piece count."""
        piece_count = _count_pieces_from_features(stm_features)
        return torch.clamp(piece_count // self.bucket_divisor, 0, self.num_buckets - 1)
    
    def forward(self, stm_features, nstm_features):
        """Forward pass.
        
        Args:
            stm_features: (batch, NUM_FEATURES) side-to-move features
            nstm_features: (batch, NUM_FEATURES) not-side-to-move features
            
        Returns:
            (batch, 1) raw evaluation (unbounded, use sigmoid for WDL)
        """
        # Accumulator
        raw_stm = self.accumulator(stm_features)     # (batch, acc_size+1)
        raw_nstm = self.accumulator(nstm_features)   # (batch, acc_size+1)
        
        # Split hidden and PSQT
        hidden_stm, psqt_stm = raw_stm[:, :-1], raw_stm[:, -1]
        hidden_nstm, psqt_nstm = raw_nstm[:, :-1], raw_nstm[:, -1]
        
        # SCReLU on hidden: clamp(x, 0, 1)²
        act_stm = torch.clamp(hidden_stm, 0.0, 1.0) ** 2
        act_nstm = torch.clamp(hidden_nstm, 0.0, 1.0) ** 2
        
        # Concat perspectives
        x = torch.cat([act_stm, act_nstm], dim=1)    # (batch, acc_size*2)
        
        # Bucket selection
        buckets = self._get_buckets(stm_features)
        
        # Layer stack
        ls_out = self.layer_stack(x, buckets).squeeze(-1)  # (batch, 1)
        
        # PSQT: perspective difference
        psqt = (psqt_stm - psqt_nstm) * 0.5               # (batch,)
        
        # Final output: raw (unbounded)
        output = ls_out.squeeze(-1) + psqt
        return output.unsqueeze(-1)
    
    def evaluate(self, board):
        """Evaluate a single board position.
        
        Args:
            board: Board object
            
        Returns:
            float: raw evaluation (use sigmoid(val/scaling) for win probability)
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
            'version': 2,
            'state_dict': self.state_dict(),
            'accumulator_size': self.accumulator_size,
            'hidden1_size': self.layer_stack.hidden1_size,
            'hidden2_size': self.layer_stack.hidden2_size,
            'num_buckets': self.num_buckets,
            'bucket_divisor': self.bucket_divisor,
        }, path)
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            accumulator_size=checkpoint['accumulator_size'],
            hidden1_size=checkpoint['hidden1_size'],
            hidden2_size=checkpoint['hidden2_size'],
            num_buckets=checkpoint.get('num_buckets', 4),
            bucket_divisor=checkpoint.get('bucket_divisor', 20),
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        return model
