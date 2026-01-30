from dataclasses import dataclass

@dataclass
class NetworkConfig:
    num_res_blocks: int = 10
    num_channels: int = 256
    
@dataclass
class TrainingConfig:
    num_iterations: int = 100
    num_self_play_games: int = 20
    num_train_epochs: int = 50
    num_simulations: int = 200
    batch_size: int = 512
    lr: float = 0.001
    weight_decay: float = 1e-4
    
    replay_buffer_size: int = 50000
    
    save_dir: str = "./model"
    save_interval: int = 10
    
    use_amp: bool = True
    num_parallel_games: int = 1
    
@dataclass
class GPUConfig:
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
@dataclass
class MCTSConfig:
    c_puct: float = 1.0
    temperature_start: float = 1.0
    temperature_end: float = 0.5
    
@dataclass
class Config:
    network: NetworkConfig = None
    training: TrainingConfig = None
    gpu: GPUConfig = None
    mcts: MCTSConfig = None
    
    def __post_init__(self):
        if self.network is None:
            self.network = NetworkConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.gpu is None:
            self.gpu = GPUConfig()
        if self.mcts is None:
            self.mcts = MCTSConfig()

def get_default_config():
    return Config()

def get_gpu_optimized_config():
    """RTX 5090 같은 고성능 GPU를 위한 설정"""
    config = Config()
    config.network = NetworkConfig(
        num_res_blocks=15,
        num_channels=256
    )
    config.training = TrainingConfig(
        num_iterations=200,  # 100 → 200 (2배 증가)
        num_self_play_games=100,  # 30 → 100 (더 많은 데이터)
        num_train_epochs=30,  # 50 → 30 (오버피팅 방지, 대신 데이터 증가)
        num_simulations=150,  # 100 → 150 (더 정확한 MCTS)
        batch_size=1024,
        lr=0.002,
        weight_decay=1e-4,
        replay_buffer_size=200000,  # 100k → 200k (더 많은 다양성)
        save_interval=10,  # 5 → 10 (저장 빈도 감소)
        use_amp=True,
        num_parallel_games=32
    )
    config.gpu = GPUConfig(
        device="cuda",
        num_workers=8,
        pin_memory=True
    )
    return config

def get_cpu_config():
    """CPU 또는 저성능 환경을 위한 설정"""
    config = Config()
    config.network = NetworkConfig(
        num_res_blocks=5,
        num_channels=64
    )
    config.training = TrainingConfig(
        num_iterations=20,
        num_self_play_games=5,
        num_train_epochs=10,
        num_simulations=50,
        batch_size=16,
        lr=0.001,
        use_amp=False,
        num_parallel_games=1
    )
    config.gpu = GPUConfig(
        device="cpu",
        num_workers=2,
        pin_memory=False
    )
    return config
