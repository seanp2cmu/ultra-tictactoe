"""Main trainer class for AlphaZero with DTW."""
from typing import Dict, Optional
from tqdm import tqdm

from .replay_buffer import SelfPlayData
from .self_play import SelfPlayWorker, reset_parallel_timing


class Trainer:
    """AlphaZero trainer with DTW endgame support."""
    
    def __init__(
        self,
        network=None,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        num_simulations: int = 100,
        replay_buffer_size: int = 10000,
        device: Optional[str] = None,
        use_amp: bool = True,
        num_res_blocks: int = 10,
        num_channels: int = 256,
        endgame_threshold: int = 15,
        hot_cache_size: int = 50000,
        cold_cache_size: int = 500000,
        total_iterations: int = 300
    ) -> None:
        if network is None:
            from ..core import Model, AlphaZeroNet
            model = Model(num_res_blocks=num_res_blocks, num_channels=num_channels)
            self.network = AlphaZeroNet(
                model=model, lr=lr, weight_decay=weight_decay,
                device=device, use_amp=use_amp, total_iterations=total_iterations
            )
        else:
            self.network = network
        
        self.batch_size = batch_size
        self.num_simulations = num_simulations
        self.replay_buffer = SelfPlayData(max_size=replay_buffer_size)
        
        # DTW always enabled
        from ..endgame import DTWCalculator
        self.dtw_calculator = DTWCalculator(
            use_cache=True,
            hot_size=hot_cache_size,
            cold_size=cold_cache_size,
            endgame_threshold=endgame_threshold
        )
    
    def generate_self_play_data(
        self,
        num_games: int = 10,
        temperature: float = 1.0,
        verbose: bool = False,
        disable_tqdm: bool = False,
        num_simulations: Optional[int] = None,
        parallel_games: int = 128
    ) -> int:
        """Generate self-play data using parallel self-play."""
        sims = num_simulations if num_simulations is not None else self.num_simulations
        
        # Reset timing stats
        reset_parallel_timing()
        
        # Parallel self-play
        worker = SelfPlayWorker(
            network=self.network,
            dtw_calculator=self.dtw_calculator,
            num_simulations=sims,
            temperature=temperature,
            parallel_games=parallel_games
        )
        
        # Get starting game_id from replay buffer
        game_id_start = self.replay_buffer._next_game_id
        all_data = worker.play_games(num_games, disable_tqdm=disable_tqdm, 
                                      game_id_start=game_id_start)
        
        # Update replay buffer's game_id counter
        max_game_id = max((d[3] for d in all_data), default=game_id_start)
        self.replay_buffer._next_game_id = max_game_id + 1
        
        # Batch insert into replay buffer (avoids per-sample Python overhead)
        if all_data:
            import numpy as np
            states = np.array([d[0] for d in all_data])
            policies = np.array([d[1] for d in all_data])
            values = np.array([d[2] for d in all_data], dtype=np.float32)
            game_ids_arr = np.array([d[3] for d in all_data], dtype=np.int64)
            self.replay_buffer.add_batch(states, policies, values, game_ids_arr)
        
        if verbose:
            print(f"\nTotal samples in replay buffer: {len(self.replay_buffer)}")
            if self.dtw_calculator:
                cache_stats = self.dtw_calculator.get_stats()
                print(f"DTW Cache: {cache_stats}")
        
        return len(all_data)
    
    def train(self, num_epochs: int = 10, verbose: bool = False, disable_tqdm: bool = False) -> Dict:
        """Train network on replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0}
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        epoch_pbar = tqdm(range(num_epochs), desc="Training", leave=False, disable=disable_tqdm, ncols=100, position=1)
        
        for _ in epoch_pbar:
            boards, policies, values, _ = self.replay_buffer.sample(self.batch_size)
            
            t_loss, p_loss, v_loss = self.network.train_step(boards, policies, values)
            
            total_loss += t_loss
            total_policy_loss += p_loss
            total_value_loss += v_loss
            num_batches += 1
            
            epoch_pbar.set_postfix({
                "loss": f"{t_loss:.4f}",
                "p_loss": f"{p_loss:.4f}",
                "v_loss": f"{v_loss:.4f}"
            })
        
        return {
            'total_loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches
        }
    
    def train_iteration(
        self,
        num_self_play_games: int = 10,
        num_train_epochs: int = 10,
        temperature: float = 1.0,
        verbose: bool = False,
        disable_tqdm: bool = False,
        num_simulations: Optional[int] = None,
        parallel_games: int = 1,
        iteration: int = 0
    ) -> Dict:
        """Single training iteration."""
        # Set current iteration for age-based weighting
        self.replay_buffer.set_iteration(iteration)
        
        if verbose:
            print("=" * 60)
            print("Generating self-play data...")
            print("DTW enabled for endgame improvement")
            print("=" * 60)
        
        num_samples = self.generate_self_play_data(
            num_games=num_self_play_games,
            temperature=temperature,
            verbose=verbose,
            disable_tqdm=disable_tqdm,
            num_simulations=num_simulations,
            parallel_games=parallel_games
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("Training network...")
            print("=" * 60)
        
        avg_loss = self.train(num_epochs=num_train_epochs, verbose=verbose, disable_tqdm=disable_tqdm)
        
        current_lr = self.network.step_scheduler()
        
        # Sync TRT engine with updated weights for next iteration's self-play
        self.network.sync_trt_weights()
        
        result = {
            'num_samples': num_samples,
            'avg_loss': avg_loss,
            'learning_rate': current_lr
        }
        
        if self.dtw_calculator:
            result['dtw_stats'] = self.dtw_calculator.get_stats()
        
        if verbose:
            print(f"Learning rate: {current_lr:.6f}")
        
        return result
    
    def save(self, filepath: str, iteration: int = None) -> None:
        """Save network with optional iteration info."""
        self.network.save(filepath, iteration=iteration)
    
    def load(self, filepath: str) -> int:
        """Load network and return iteration number."""
        return self.network.load(filepath)
