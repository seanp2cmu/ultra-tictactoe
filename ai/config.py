from dataclasses import dataclass

@dataclass
class NetworkConfig:
    num_res_blocks: int = 20
    num_channels: int = 256
    
@dataclass
class TrainingConfig:
    num_iterations: int = 300
    num_self_play_games: int = 8192
    num_train_epochs: int = 20
    num_simulations: int = 800
    batch_size: int = 2048         
    lr: float = 0.001
    weight_decay: float = 1e-4
    
    replay_buffer_size: int = 750000
    
    save_dir: str = "./model"
    
    use_amp: bool = True
    
    # Phase 2: auto-switch when convergence detected
    phase2_num_simulations: int = 800
    phase2_num_train_epochs: int = 10
    convergence_window: int = 20       # check last N iterations
    convergence_threshold: float = 0.0  # disabled (phase1 == phase2 == 800 sims)
    
@dataclass
class GPUConfig:
    device: str = "cuda"
    num_workers: int = 16  
    pin_memory: bool = True
    parallel_games: int = 8192
    inference_batch_size: int = 8192

@dataclass
class DTWConfig:
    endgame_threshold: int = 15       
    hot_cache_size: int = 60000000 
    cold_cache_size: int = 240000000    
    
@dataclass
class Config:
    network: NetworkConfig = None
    training: TrainingConfig = None
    gpu: GPUConfig = None
    dtw: DTWConfig = None
    
    def __post_init__(self):
        if self.network is None:
            self.network = NetworkConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.gpu is None:
            self.gpu = GPUConfig()
        if self.dtw is None:
            self.dtw = DTWConfig()