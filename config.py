from dataclasses import dataclass

@dataclass
class NetworkConfig:
    num_res_blocks: int = 20
    num_channels: int = 384
    
@dataclass
class TrainingConfig:
    num_iterations: int = 300
    num_self_play_games: int = 200
    num_train_epochs: int = 40
    num_simulations: int = 400
    batch_size: int = 2048
    lr: float = 0.002
    weight_decay: float = 1e-4
    
    replay_buffer_size: int = 500000
    
    save_dir: str = "./model"
    save_interval: int = 10
    
    use_amp: bool = True
    num_parallel_games: int = 32
    
@dataclass
class GPUConfig:
    device: str = "cuda"
    num_workers: int = 12
    pin_memory: bool = True
    
@dataclass
class MCTSConfig:
    c_puct: float = 1.0
    temperature_start: float = 1.0
    temperature_end: float = 0.3

@dataclass
class DTWConfig:
    use_dtw: bool = True
    max_depth: int = 18               # DTW 탐색 깊이 (18수 앞까지 계산)
    endgame_threshold: int = 25       # 빈 칸 25개 이하면 엔드게임 (Tablebase 영역)
    hot_cache_size: int = 2000000     # 200만 (빠른 접근)
    cold_cache_size: int = 20000000   # 2000만 (압축 저장)
    use_symmetry: bool = True         # 보드 대칭 정규화 (8배 메모리 절약)
    use_tablebase: bool = True        # Retrograde Tablebase 사용 (25칸 이하 완벽)

@dataclass
class PredictionConfig:
    """실제 게임/예측 시 사용할 설정"""
    num_simulations: int = 400        # 실전에서는 더 많은 시뮬레이션
    temperature: float = 0.1          # 낮은 temperature (더 결정적)
    use_dtw: bool = True              # DTW 사용 (엔드게임 완벽)
    dtw_max_depth: int = 18           # DTW 깊이
    
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