import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import threading
from torchsummaryX import summary
from torch.amp import autocast, GradScaler


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block - Channel Attention"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC → ReLU → FC → Sigmoid
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: Channel-wise multiplication
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual Block with SE (Squeeze-and-Excitation)"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction=16)  # SE Block 추가
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # SE Block으로 channel attention 적용
        out = self.se(out)
        
        # Residual connection
        out += residual
        out = F.relu(out)
        
        return out


class Model(nn.Module):
    def __init__(self, num_res_blocks=10, num_channels=256):
        super(Model, self).__init__()
        
        self.num_channels = num_channels
        
        # Step 5: Updated input to 7 channels (from 6)
        # Channels: my_pieces, opponent_pieces, my_completed, opponent_completed, 
        #           draw_completed, last_move, valid_board_mask
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
        
        # Step 1: Mask completed boards (completed small boards' cells are meaningless)
        if hasattr(board_state, 'completed_boards'):
            for br in range(3):
                for bc in range(3):
                    if board_state.completed_boards[br][bc] != 0:
                        # 완료된 보드의 개별 칸 정보 제거 (규칙상 무의미)
                        start_r, start_c = br * 3, bc * 3
                        boards[start_r:start_r+3, start_c:start_c+3] = 0
        
        # Step 2: Perspective normalization (my pieces vs opponent pieces)
        # Always represent from current player's perspective for better learning
        current_player = board_state.current_player if hasattr(board_state, 'current_player') else 1
        
        if current_player == 1:
            my_plane = (boards == 1).astype(np.float32)
            opponent_plane = (boards == 2).astype(np.float32)
        else:  # current_player == 2
            my_plane = (boards == 2).astype(np.float32)
            opponent_plane = (boards == 1).astype(np.float32)
        
        # Completed boards (also perspective normalized)
        my_completed_plane = np.zeros((9, 9), dtype=np.float32)
        opponent_completed_plane = np.zeros((9, 9), dtype=np.float32)
        draw_completed_plane = np.zeros((9, 9), dtype=np.float32)
        
        if hasattr(board_state, 'completed_boards'):
            for br in range(3):
                for bc in range(3):
                    status = board_state.completed_boards[br][bc]
                    start_r, start_c = br * 3, bc * 3
                    
                    if status == current_player:
                        # 내가 완료한 보드
                        my_completed_plane[start_r:start_r+3, start_c:start_c+3] = 1
                    elif status == (3 - current_player):
                        # 상대가 완료한 보드
                        opponent_completed_plane[start_r:start_r+3, start_c:start_c+3] = 1
                    elif status == 3:
                        # 무승부
                        draw_completed_plane[start_r:start_r+3, start_c:start_c+3] = 1
        
        # Step 3: Last move plane (critical for Ultimate Tic-Tac-Toe rules)
        last_move_plane = np.zeros((9, 9), dtype=np.float32)
        if hasattr(board_state, 'last_move') and board_state.last_move is not None:
            last_r, last_c = board_state.last_move
            last_move_plane[last_r, last_c] = 1.0
        
        # Step 4: Valid board mask (shows which small boards can be played)
        valid_board_mask = np.zeros((9, 9), dtype=np.float32)
        if hasattr(board_state, 'last_move') and board_state.last_move is not None:
            last_r, last_c = board_state.last_move
            target_board_r = last_r % 3
            target_board_c = last_c % 3
            
            # Check if target board is completed
            if hasattr(board_state, 'completed_boards'):
                if board_state.completed_boards[target_board_r][target_board_c] == 0:
                    # Target board is available - mark only that board
                    start_r = target_board_r * 3
                    start_c = target_board_c * 3
                    valid_board_mask[start_r:start_r+3, start_c:start_c+3] = 1.0
                else:
                    # Target board completed - can play any non-completed board
                    for br in range(3):
                        for bc in range(3):
                            if board_state.completed_boards[br][bc] == 0:
                                start_r, start_c = br * 3, bc * 3
                                valid_board_mask[start_r:start_r+3, start_c:start_c+3] = 1.0
            else:
                # No completed_boards info - assume target board is valid
                start_r = target_board_r * 3
                start_c = target_board_c * 3
                valid_board_mask[start_r:start_r+3, start_c:start_c+3] = 1.0
        else:
            # First move - all boards are valid
            valid_board_mask[:] = 1.0
        
        board_tensor = np.stack([
            my_plane, opponent_plane,
            my_completed_plane, opponent_completed_plane, draw_completed_plane,
            last_move_plane, valid_board_mask
        ], axis=0)
        board_tensor = torch.FloatTensor(board_tensor).unsqueeze(0)
        
        device = next(self.parameters()).device
        board_tensor = board_tensor.to(device)
        
        return board_tensor


class AlphaZeroNet:
    def __init__(self, lr=0.001, weight_decay=1e-4, device=None, model=None, use_amp=True, total_iterations=300):
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
        
        # Optimizer with explicit hyperparameters
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        # Cosine Annealing LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_iterations,
            eta_min=lr * 0.01  # Minimum LR = 1% of initial
        )
        
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
            
            # Gradient clipping (unscale first for AMP)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_logits, value_preds = self.model(boards_tensor)
            policy_loss = F.cross_entropy(policy_logits, policies_tensor)
            value_loss = F.mse_loss(value_preds.squeeze(), values_tensor)
            total_loss = policy_loss + value_loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def step_scheduler(self):
        """Call after each iteration to update learning rate"""
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        return current_lr
    
    def get_current_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def save(self, filepath):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),  # Scheduler 상태 저장
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
        
        # Scheduler 상태 복원
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])


if __name__ == "__main__":
    net = AlphaZeroNet()
    
    dummy_input = torch.randn(1, 6, 9, 9).to(net.device)
    policy_logits, value = net.model(dummy_input)
    
    summary(net.model, dummy_input)
