"""Main trainer class for AlphaZero with DTW."""
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .replay_buffer import SelfPlayData
from .self_play import SelfPlayWorker
from ai.endgame import DTWCalculator

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
        num_parallel_games: int = 1,
        endgame_threshold: int = 15,
        hot_cache_size: int = 50000,
        cold_cache_size: int = 500000,
        use_symmetry: bool = True,
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
        self.num_parallel_games = num_parallel_games
        
        self.hot_cache_size = hot_cache_size
        self.cold_cache_size = cold_cache_size
        self.use_symmetry = use_symmetry
        
        # DTW always enabled
        from ..endgame import DTWCalculator
        self.dtw_calculator = DTWCalculator(
            use_cache=True,
            hot_size=hot_cache_size,
            cold_size=cold_cache_size,
            use_symmetry=use_symmetry,
            endgame_threshold=endgame_threshold
        )
        
        self.total_dtw_positions = 0
        self.total_dtw_wins = 0
    
    def _play_single_game(
        self,
        temperature: float,
        verbose: bool,
        batch_predictor=None,
        num_simulations: Optional[int] = None
    ):
        """Play single self-play game."""
        sims = num_simulations if num_simulations is not None else self.num_simulations
        
        worker = SelfPlayWorker(
            self.network,
            dtw_calculator=self.dtw_calculator,  # 공유 캐시 사용!
            batch_predictor=batch_predictor,
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
        
        if self.num_parallel_games > 1:
            from ..utils import BatchPredictor
            with BatchPredictor(self.network, batch_size=self.num_parallel_games,
                              wait_time=0.005, verbose=verbose) as batch_predictor:
                with ThreadPoolExecutor(max_workers=self.num_parallel_games) as executor:
                    futures = [
                        executor.submit(self._play_single_game,
                                      temperature, verbose, batch_predictor, num_simulations)
                        for _ in range(num_games)
                    ]
                    
                    game_pbar = tqdm(total=num_games, desc="Self-play",
                                   leave=False, disable=disable_tqdm, ncols=100)
                    
                    completed_games = 0
                    total_positions = 0
                    dtw_positions = 0
                    
                    for future in as_completed(futures):
                        game_data = future.result()
                        all_data.extend(game_data)
                        completed_games += 1
                        
                        for _, _, _, dtw in game_data:
                            total_positions += 1
                            if dtw is not None:
                                dtw_positions += 1
                        
                        avg_length = len(all_data) / completed_games if completed_games > 0 else 0
                        dtw_rate = dtw_positions / total_positions if total_positions > 0 else 0
                        
                        game_pbar.update(1)
                        game_pbar.set_postfix({
                            "samples": len(all_data),
                            "avg_len": f"{avg_length:.1f}",
                            "dtw%": f"{dtw_rate:.1%}"
                        })
                    
                    game_pbar.close()
        else:
            game_pbar = tqdm(range(num_games), desc="Self-play",
                           leave=False, disable=disable_tqdm, ncols=100)
            for i in game_pbar:
                game_data = self._play_single_game(temperature, verbose, None, num_simulations)
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
        
        return len(all_data)
    
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
    
    def save(self, filepath: str) -> None:
        """Save network."""
        self.network.save(filepath)
    
    def load(self, filepath: str) -> None:
        """Load network."""
        self.network.load(filepath)
    
    def clear_dtw_cache(self) -> None:
        """Clear DTW cache to free memory."""
        if self.dtw_calculator:
            self.dtw_calculator.clear_cache()
