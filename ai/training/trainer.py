"""Main trainer class for AlphaZero with DTW."""
from typing import Dict, Optional
from tqdm import tqdm
import time
from functools import wraps

from .replay_buffer import SelfPlayData
from .self_play import SelfPlayWorker, get_timing_stats, reset_timing_stats, set_slow_log_file
from ai.mcts.agent import get_mcts_timing, reset_mcts_timing

def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        return result, te-ts
    return wrap

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
        
        self.hot_cache_size = hot_cache_size
        self.cold_cache_size = cold_cache_size
        
        # DTW always enabled (symmetry always used)
        from ..endgame import DTWCalculator
        self.dtw_calculator = DTWCalculator(
            use_cache=True,
            hot_size=hot_cache_size,
            cold_size=cold_cache_size,
            endgame_threshold=endgame_threshold
        )
        
        self.total_dtw_positions = 0
        self.total_dtw_wins = 0
    
    def _play_single_game(
        self,
        temperature: float,
        verbose: bool,
        num_simulations: Optional[int] = None
    ):
        """Play single self-play game."""
        sims = num_simulations if num_simulations is not None else self.num_simulations
        
        worker = SelfPlayWorker(
            self.network,
            dtw_calculator=self.dtw_calculator,
            num_simulations=sims,
            temperature=temperature
        )
        
        return worker.play_game(verbose=verbose)
    
    def generate_self_play_data(
        self,
        num_games: int = 10,
        temperature: float = 1.0,
        verbose: bool = False,
        disable_tqdm: bool = False,
        num_simulations: Optional[int] = None
    ) -> int:
        """Generate self-play data."""
        all_data = []
        
        # Reset timing stats
        reset_timing_stats()
        reset_mcts_timing()
        
        game_pbar = tqdm(range(num_games), desc="Self-play",
                       leave=False, disable=disable_tqdm, ncols=100)
        for i in game_pbar:
            game_data = self._play_single_game(temperature, verbose, num_simulations)
            all_data.extend(game_data)
            
            dtw_count = sum(1 for _, _, _, dtw in game_data if dtw is not None)
            dtw_rate = dtw_count / len(game_data) if game_data else 0
            avg_length = len(all_data) / (i + 1)
            
            game_pbar.set_postfix({
                "samples": len(all_data),
                "avg_len": f"{avg_length:.1f}",
                "dtw%": f"{dtw_rate:.1%}"
            })
        
        for state, policy, value, dtw in all_data:
            self.replay_buffer.add(state, policy, value, dtw)
            
            if dtw is not None:
                self.total_dtw_positions += 1
                if dtw < float('inf'):
                    self.total_dtw_wins += 1
        
        if verbose:
            print(f"\nTotal samples in replay buffer: {len(self.replay_buffer)}")
            if self.total_dtw_positions > 0:
                win_rate = self.total_dtw_wins / self.total_dtw_positions
                print(f"DTW positions: {self.total_dtw_positions}, Win rate: {win_rate:.2%}")
                if self.dtw_calculator:
                    cache_stats = self.dtw_calculator.get_stats()
                    print(f"DTW Cache: {cache_stats}")
        
        # Print timing stats
        self._print_timing_stats()
        
        return len(all_data)
    
    def _print_timing_stats(self):
        """Print detailed timing breakdown."""
        sp_stats = get_timing_stats()
        mcts_stats = get_mcts_timing()
        
        print("\n" + "="*60)
        print("⏱️  TIMING BREAKDOWN")
        print("="*60)
        
        # Self-play level
        print("\n[Self-Play Level]")
        if sp_stats['mcts_count'] > 0:
            print(f"  MCTS Search: {sp_stats['mcts_search']:.2f}s ({sp_stats['mcts_count']} calls, avg {sp_stats['mcts_search']/sp_stats['mcts_count']:.3f}s)")
        if sp_stats['dtw_endgame_count'] > 0:
            print(f"  DTW Endgame: {sp_stats['dtw_endgame']:.2f}s ({sp_stats['dtw_endgame_count']} calls, avg {sp_stats['dtw_endgame']/sp_stats['dtw_endgame_count']:.3f}s)")
        if sp_stats['dtw_midgame_count'] > 0:
            print(f"  DTW Midgame: {sp_stats['dtw_midgame']:.2f}s ({sp_stats['dtw_midgame_count']} calls, avg {sp_stats['dtw_midgame']/sp_stats['dtw_midgame_count']:.3f}s)")
        print(f"  Board to Input: {sp_stats['board_to_input']:.2f}s")
        print(f"  Total Steps: {sp_stats['total_steps']}")
        
        # MCTS level
        print("\n[MCTS Level]")
        print(f"  Network Predict (root): {mcts_stats['network_predict']:.2f}s")
        print(f"  Network Predict Batch: {mcts_stats['network_predict_batch']:.2f}s")
        if mcts_stats['dtw_in_mcts_count'] > 0:
            print(f"  DTW in MCTS: {mcts_stats['dtw_in_mcts']:.2f}s ({mcts_stats['dtw_in_mcts_count']} calls)")
        print(f"  Select: {mcts_stats['select']:.2f}s")
        print(f"  Expand: {mcts_stats['expand']:.2f}s")
        print(f"  Backprop: {mcts_stats['backprop']:.2f}s")
        
        # Slow steps summary
        if sp_stats['slow_steps']:
            print(f"\n[⚠️ SLOW STEPS (>5s)] - {len(sp_stats['slow_steps'])} found (see slow_steps.txt)")
        else:
            print(f"\n[✅ No slow steps (>5s)]")
        
        print("="*60)
    
    def train(self, num_epochs: int = 10, verbose: bool = False, disable_tqdm: bool = False) -> Dict:
        """Train network on replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0}
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        epoch_pbar = tqdm(range(num_epochs), desc="Training", leave=False, disable=disable_tqdm, ncols=100)
        
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
        num_simulations: Optional[int] = None
    ) -> Dict:
        """Single training iteration."""
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
            num_simulations=num_simulations
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("Training network...")
            print("=" * 60)
        
        avg_loss = self.train(num_epochs=num_train_epochs, verbose=verbose, disable_tqdm=disable_tqdm)
        
        current_lr = self.network.step_scheduler()
        
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
    
    def clear_dtw_cache(self) -> None:
        """Clear DTW cache to free memory."""
        if self.dtw_calculator:
            self.dtw_calculator.clear_cache()
