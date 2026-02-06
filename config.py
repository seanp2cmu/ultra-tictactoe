from dataclasses import dataclass

@dataclass
class NetworkConfig:
    num_res_blocks: int = 30
    num_channels: int = 512 
    
@dataclass
class TrainingConfig:
    num_iterations: int = 300
    num_self_play_games: int = 250    
    num_train_epochs: int = 40
    num_simulations: int = 800        
    batch_size: int = 1024            
    lr: float = 0.002
    weight_decay: float = 1e-4
    
    replay_buffer_size: int = 1000000  
    
    save_dir: str = "./model"
    save_interval: int = 5  # 5 iteration마다 저장 (자동 복구 지원)
    
    use_amp: bool = True
    
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
    endgame_threshold: int = 15       
    midgame_threshold: int = 45       
    shallow_depth: int = 8            
    hot_cache_size: int = 60000000 
    cold_cache_size: int = 240000000
@dataclass
class PredictionConfig:
    """실제 게임/예측 시 사용할 설정"""
    num_simulations: int = 1600       
    temperature: float = 0.1          
    
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