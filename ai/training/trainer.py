"""Main trainer class for AlphaZero with DTW."""
from typing import Dict, Optional
from tqdm import tqdm

from .replay_buffer import SelfPlayData
from .parallel_mcts import reset_parallel_timing, get_parallel_timing
from .self_play import run_multiprocess_self_play


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
        total_iterations: int = 300,
        inference_batch_size: int = 8192,
    ) -> None:
        if network is None:
            from ..core import Model, AlphaZeroNet
            model = Model(num_res_blocks=num_res_blocks, num_channels=num_channels)
            self.network = AlphaZeroNet(
                model=model, lr=lr, weight_decay=weight_decay,
                device=device, use_amp=use_amp, total_iterations=total_iterations,
                inference_batch_size=inference_batch_size,
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
        num_simulations: Optional[int] = None,
        parallel_games: int = 2048,
        num_workers: int = 4,
    ) -> int:
        """Generate self-play data using multi-process workers + centralised GPU inference."""
        import numpy as np
        sims = num_simulations if num_simulations is not None else self.num_simulations
        game_id_start = self.replay_buffer._next_game_id
        
        (states, policies, values, game_ids), self._mp_timing = run_multiprocess_self_play(
            network=self.network,
            num_games=num_games,
            num_simulations=sims,
            parallel_games=parallel_games,
            temperature=temperature,
            game_id_start=game_id_start,
            num_workers=num_workers,
            endgame_threshold=self.dtw_calculator.endgame_threshold if self.dtw_calculator else 15,
        )
        
        if len(states) > 0:
            self.replay_buffer._next_game_id = int(game_ids.max()) + 1
            self.replay_buffer.add_batch(states, policies, values, game_ids)
        
        return len(states)
    
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
        parallel_games: int = 2048,
        iteration: int = 0,
        num_workers: int = 4,
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
            num_simulations=num_simulations,
            parallel_games=parallel_games,
            num_workers=num_workers,
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
        
        if hasattr(self, '_mp_timing') and self._mp_timing.get('dtw_stats'):
            result['dtw_stats'] = self._mp_timing['dtw_stats']
        elif self.dtw_calculator:
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
