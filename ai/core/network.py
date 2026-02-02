"""Neural network architecture for Ultimate Tic-Tac-Toe."""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from game import Board


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
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_state) -> Tuple[np.ndarray, float]:
        """Single board prediction."""
        self.eval()
        with torch.no_grad():
            board_tensor = self._board_to_tensor(board_state)
            policy_logits, value = self.forward(board_tensor)
            policy_probs = F.softmax(policy_logits, dim=1)
            return policy_probs.cpu().numpy()[0], value.cpu().numpy()[0][0]
    
    def _board_to_tensor(self, board_state: Board) -> torch.Tensor:
        """Convert board to 7-channel tensor input (최적화: np.kron 대신 슬라이싱)."""
        if isinstance(board_state, np.ndarray):
            boards = board_state.copy()
        else:
            boards = np.array(board_state.boards, dtype=np.float32)
        
        current_player = board_state.current_player if hasattr(board_state, 'current_player') else 1
        opponent_player = 3 - current_player
        
        # 최적화: planes 미리 초기화
        my_completed_plane = np.zeros((9, 9), dtype=np.float32)
        opponent_completed_plane = np.zeros((9, 9), dtype=np.float32)
        draw_completed_plane = np.zeros((9, 9), dtype=np.float32)
        valid_board_mask = np.zeros((9, 9), dtype=np.float32)
        
        if hasattr(board_state, 'completed_boards'):
            completed = board_state.completed_boards
            
            # 최적화: np.kron 대신 직접 슬라이싱으로 3x3 블록 채우기
            for br in range(3):
                for bc in range(3):
                    start_r, start_c = br * 3, bc * 3
                    status = completed[br][bc]
                    
                    if status != 0:
                        # 완료된 보드: 해당 영역 빈 칸으로
                        boards[start_r:start_r+3, start_c:start_c+3] = 0
                        
                        if status == current_player:
                            my_completed_plane[start_r:start_r+3, start_c:start_c+3] = 1.0
                        elif status == opponent_player:
                            opponent_completed_plane[start_r:start_r+3, start_c:start_c+3] = 1.0
                        elif status == 3:
                            draw_completed_plane[start_r:start_r+3, start_c:start_c+3] = 1.0
        
        # player planes
        my_plane = (boards == current_player).astype(np.float32)
        opponent_plane = (boards == opponent_player).astype(np.float32)
        
        # last move plane
        last_move_plane = np.zeros((9, 9), dtype=np.float32)
        if hasattr(board_state, 'last_move') and board_state.last_move is not None:
            last_r, last_c = board_state.last_move
            last_move_plane[last_r, last_c] = 1.0
            
            # valid board mask 계산
            target_board_r, target_board_c = last_r % 3, last_c % 3
            
            if hasattr(board_state, 'completed_boards'):
                if board_state.completed_boards[target_board_r][target_board_c] == 0:
                    start_r, start_c = target_board_r * 3, target_board_c * 3
                    valid_board_mask[start_r:start_r+3, start_c:start_c+3] = 1.0
                else:
                    # 완료되지 않은 보드에 마스크 적용
                    for br in range(3):
                        for bc in range(3):
                            if board_state.completed_boards[br][bc] == 0:
                                sr, sc = br * 3, bc * 3
                                valid_board_mask[sr:sr+3, sc:sc+3] = 1.0
            else:
                start_r, start_c = target_board_r * 3, target_board_c * 3
                valid_board_mask[start_r:start_r+3, start_c:start_c+3] = 1.0
        else:
            valid_board_mask[:] = 1.0
        
        board_tensor = np.stack([
            my_plane, opponent_plane,
            my_completed_plane, opponent_completed_plane, draw_completed_plane,
            last_move_plane, valid_board_mask
        ], axis=0)
        board_tensor = torch.FloatTensor(board_tensor).unsqueeze(0)
        
        device = next(self.parameters()).device
        return board_tensor.to(device)
