from dataclasses import dataclass

@dataclass
class NetworkConfig:
    num_res_blocks: int = 15
    num_channels: int = 256
    
@dataclass
class TrainingConfig:
    num_iterations: int = 100
    num_self_play_games: int = 2048
    num_train_epochs: int = 40
    num_simulations: int = 200
    batch_size: int = 1024         
    lr: float = 0.002
    weight_decay: float = 1e-4
    
    replay_buffer_size: int = 2000000  
    
    save_dir: str = "./model"
    save_interval: int = 5
    
    use_amp: bool = True
    
@dataclass
class GPUConfig:
    device: str = "cuda"
    num_workers: int = 12  
    pin_memory: bool = True
    parallel_games: int = 2048

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