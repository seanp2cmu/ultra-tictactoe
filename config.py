from dataclasses import dataclass

@dataclass
class NetworkConfig:
    # RTX 5090 (32GB VRAM)은 매우 강력 → 더 큰 네트워크 사용 가능
    num_res_blocks: int = 30  # 20→30: VRAM 충분하므로 깊이 증가 (정확도 향상)
    num_channels: int = 512   # 384→512: 채널 증가로 표현력 향상 (VRAM 여유)
    
@dataclass
class TrainingConfig:
    num_iterations: int = 300
    num_self_play_games: int = 250    # 200→250: 12 vCPU로 충분히 처리 가능
    num_train_epochs: int = 40
    num_simulations: int = 800        # 400→800: RTX 5090으로 빠른 inference 가능
    batch_size: int = 4096            # 2048→4096: 32GB VRAM이면 2배 증가 가능 (학습 안정성 향상)
    lr: float = 0.002
    weight_decay: float = 1e-4
    
    replay_buffer_size: int = 1000000  # 500k→1M: 92GB RAM으로 2배 증가 (더 다양한 경험)
    
    save_dir: str = "./model"
    save_interval: int = 10
    
    use_amp: bool = True
    num_parallel_games: int = 12       # 32→12: 12 vCPU에 맞춤 (CPU 100% 활용)
    
@dataclass
class GPUConfig:
    device: str = "cuda"
    num_workers: int = 12  # 12 vCPU에 정확히 맞춤 (최적)
    pin_memory: bool = True
    
@dataclass
class MCTSConfig:
    c_puct: float = 1.0
    temperature_start: float = 1.0
    temperature_end: float = 0.3

@dataclass
class DTWConfig:
    endgame_threshold: int = 15       # 완전 탐색 (15칸 이하)
    midgame_threshold: int = 45       # 얕은 탐색 (16-45칸)
    shallow_depth: int = 8            # 중반 얕은 탐색 depth 제한
    hot_cache_size: int = 5000000     # 500만: 92GB RAM으로 2.5배 증가 (캐시 hit rate 향상)
    cold_cache_size: int = 20000000   # 2000만: Disk 80GB로 제한적 (압축 저장)
    # 보드 대칭 정규화 항상 사용 (8배 메모리 절약)

@dataclass
class PredictionConfig:
    """실제 게임/예측 시 사용할 설정"""
    num_simulations: int = 1600       # 400→1600: RTX 5090으로 4배 증가 (더 정확한 예측)
    temperature: float = 0.1          # 낮은 temperature (더 결정적) - 적절함
    # DTW는 항상 사용됨 (엔드게임 완벽 해결)
    
@dataclass
class Config:
    network: NetworkConfig = None
    training: TrainingConfig = None
    gpu: GPUConfig = None
    mcts: MCTSConfig = None
    dtw: DTWConfig = None
    
    def __post_init__(self):
        if self.network is None:
            self.network = NetworkConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.gpu is None:
            self.gpu = GPUConfig()
        if self.mcts is None:
            self.mcts = MCTSConfig()
        if self.dtw is None:
            self.dtw = DTWConfig()