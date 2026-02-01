import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import threading
from torchsummaryX import summary
from torch.amp import autocast, GradScaler

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        residual = x
        out = self.residual(x)
        out += residual
        out = F.relu(out)
        return out


class Model(nn.Module):
    def __init__(self, num_res_blocks=10, num_channels=256):
        super(Model, self).__init__()
        
        self.num_channels = num_channels
        
        self.input = nn.Sequential(
          nn.Conv2d(6, num_channels, kernel_size=3, padding=1),
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
        
    def forward(self, x):
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
    
    def predict(self, board_state):
        self.eval()
        with torch.no_grad():
            board_tensor = self._board_to_tensor(board_state)
            policy_logits, value = self.forward(board_tensor)
            policy_probs = F.softmax(policy_logits, dim=1)
            return policy_probs.cpu().numpy()[0], value.cpu().numpy()[0][0]
    
    def _board_to_tensor(self, board_state):
        if isinstance(board_state, np.ndarray):
            boards = board_state
        else:
            boards = np.array(board_state.boards, dtype=np.float32)
        
        player1_plane = (boards == 1).astype(np.float32)
        player2_plane = (boards == 2).astype(np.float32)
        current_player_plane = np.ones((9, 9), dtype=np.float32)
        
        if hasattr(board_state, 'current_player'):
            if board_state.current_player == 2:
                current_player_plane = np.zeros((9, 9), dtype=np.float32)
        
        completed_p1_plane = np.zeros((9, 9), dtype=np.float32)
        completed_p2_plane = np.zeros((9, 9), dtype=np.float32)
        completed_draw_plane = np.zeros((9, 9), dtype=np.float32)
        
        if hasattr(board_state, 'completed_boards'):
            for br in range(3):
                for bc in range(3):
                    if board_state.completed_boards[br][bc] == 1:
                        completed_p1_plane[br*3:(br+1)*3, bc*3:(bc+1)*3] = 1
                    elif board_state.completed_boards[br][bc] == 2:
                        completed_p2_plane[br*3:(br+1)*3, bc*3:(bc+1)*3] = 1
                    elif board_state.completed_boards[br][bc] == 3:
                        completed_draw_plane[br*3:(br+1)*3, bc*3:(bc+1)*3] = 1
        
        board_tensor = np.stack([
            player1_plane, player2_plane, current_player_plane,
            completed_p1_plane, completed_p2_plane, completed_draw_plane
        ], axis=0)
        board_tensor = torch.FloatTensor(board_tensor).unsqueeze(0)
        
        device = next(self.parameters()).device
        board_tensor = board_tensor.to(device)
        
        return board_tensor


class AlphaZeroNet:
    def __init__(self, model=None, lr=0.001, weight_decay=1e-4, device=None, use_amp=True):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
        
        if model is None:
            self.model = Model().to(self.device)
        else:
            self.model = model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None
        
        self.predict_lock = threading.Lock()
        
    def predict(self, board_state):
        with self.predict_lock:
            return self.model.predict(board_state)
    
    def predict_batch(self, board_states):
        """여러 board states를 batch로 처리"""
        with self.predict_lock:
            self.model.eval()
            with torch.no_grad():
                # 모든 board states를 tensor로 변환
                batch_tensors = []
                for board_state in board_states:
                    board_tensor = self.model._board_to_tensor(board_state)
                    batch_tensors.append(board_tensor)
                
                # Batch로 합치기 (batch_size, channels, height, width)
                batch_input = torch.cat(batch_tensors, dim=0).to(self.device)
                
                # Forward pass
                policy_logits, values = self.model(batch_input)
                
                # Softmax for policy
                policy_probs = torch.softmax(policy_logits, dim=1)
                
                # CPU로 이동 및 numpy 변환
                policy_probs = policy_probs.cpu().numpy()
                values = values.cpu().numpy()
                
                return policy_probs, values.flatten()
    
    def train_step(self, boards, policies, values):
        self.model.train()
        
        boards_tensor = torch.FloatTensor(boards).to(self.device)
        policies_tensor = torch.FloatTensor(policies).to(self.device)
        values_tensor = torch.FloatTensor(values).to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with autocast(device_type=self.device.type):
                policy_logits, value_preds = self.model(boards_tensor)
                policy_loss = F.cross_entropy(policy_logits, policies_tensor)
                value_loss = F.mse_loss(value_preds.squeeze(), values_tensor)
                total_loss = policy_loss + value_loss
            
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_logits, value_preds = self.model(boards_tensor)
            policy_loss = F.cross_entropy(policy_logits, policies_tensor)
            value_loss = F.mse_loss(value_preds.squeeze(), values_tensor)
            total_loss = policy_loss + value_loss
            
            total_loss.backward()
            self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save(self, filepath):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_res_blocks': len(self.model.res_blocks),  # 모델 구조 정보 저장
            'num_channels': self.model.num_channels,
        }
        if self.scaler is not None:
            save_dict['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(save_dict, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 저장된 모델 구조 정보가 있으면 사용
        if 'num_res_blocks' in checkpoint and 'num_channels' in checkpoint:
            num_res_blocks = checkpoint['num_res_blocks']
            num_channels = checkpoint['num_channels']
        else:
            # 기존 모델: state_dict에서 구조 추론
            state_dict = checkpoint['model_state_dict']
            
            # res_blocks의 최대 인덱스 찾기
            max_block_idx = -1
            for key in state_dict.keys():
                if key.startswith('res_blocks.'):
                    block_idx = int(key.split('.')[1])
                    max_block_idx = max(max_block_idx, block_idx)
            
            num_res_blocks = max_block_idx + 1  # 0-indexed이므로 +1
            
            # num_channels 추론 (input layer에서)
            num_channels = state_dict['input.0.weight'].shape[0]
            
            print(f"Inferred model structure from checkpoint: {num_res_blocks} blocks, {num_channels} channels")
        
        # 현재 모델 구조와 다르면 새로 생성
        if len(self.model.res_blocks) != num_res_blocks or self.model.num_channels != num_channels:
            print(f"Recreating model with {num_res_blocks} blocks and {num_channels} channels")
            self.model = Model(num_res_blocks=num_res_blocks, num_channels=num_channels).to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                               lr=self.optimizer.param_groups[0]['lr'],
                                               weight_decay=self.optimizer.param_groups[0]['weight_decay'])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])


if __name__ == "__main__":
    net = AlphaZeroNet()
    
    dummy_input = torch.randn(1, 6, 9, 9).to(net.device)
    policy_logits, value = net.model(dummy_input)
    
    summary(net.model, dummy_input)
