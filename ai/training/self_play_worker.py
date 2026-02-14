"""Self-play worker for generating training data."""
from typing import List, Tuple, Optional
import time

from game import Board
from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator
from utils import BoardEncoder

from .parallel_mcts import ParallelMCTS, _parallel_timing


class SelfPlayWorker:
    """Self-play worker that runs multiple games in parallel."""
    
    def __init__(
        self,
        network: AlphaZeroNet,
        dtw_calculator: Optional[DTWCalculator] = None,
        num_simulations: int = 100,
        temperature: float = 1.0,
        parallel_games: int = 8
    ):
        self.network = network
        self.dtw_calculator = dtw_calculator
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.parallel_games = parallel_games
        
        self.mcts = ParallelMCTS(
            network=network,
            num_simulations=num_simulations,
            dtw_calculator=dtw_calculator
        )
    
    def _new_game(self, game_id):
        """Create a fresh game dict."""
        return {
            'board': Board(),
            'history': [],
            'done': False,
            'game_id': game_id,
        }
    
    def _finalize_game(self, game, all_data):
        """Collect training data from a finished game."""
        board = game['board']
        if board.winner in (None, -1, 3):
            result = 0.0  # draw
        else:
            result = 1.0 if board.winner == 1 else -1.0
        
        game_id = game['game_id']
        for step in game['history']:
            value = result if step['player'] == 1 else -result
            all_data.append((step['state'], step['policy'], value, game_id))
    
    def _finalize_game_dtw(self, game, dtw_result_code, all_data):
        """Collect training data from a game resolved by DTW."""
        board = game['board']
        if dtw_result_code == 1:
            final_value = 1.0
        elif dtw_result_code == -1:
            final_value = -1.0
        else:
            final_value = 0.0  # draw
        
        game_id = game['game_id']
        for step in game['history']:
            value = final_value if step['player'] == board.current_player else -final_value
            all_data.append((step['state'], step['policy'], value, game_id))
    
    def play_games(self, num_games: int, disable_tqdm: bool = False, 
                    game_id_start: int = 0) -> List[Tuple]:
        """
        Play multiple games with continuous batching.
        
        Finished games are immediately replaced with new ones to keep the
        batch at full capacity, maximizing GPU utilization throughout.
        
        Returns:
            List of (state, policy, value, game_id) tuples
        """
        from tqdm import tqdm
        global _parallel_timing
        
        all_data = []
        games_completed = 0
        games_started = 0
        current_game_id = game_id_start
        
        start_time = time.perf_counter()
        
        pbar = tqdm(total=num_games, desc="Self-play", 
                    disable=disable_tqdm, ncols=100, leave=False, position=1)
        
        # Initialize game pool at full capacity
        pool_size = min(self.parallel_games, num_games)
        pool = []
        for _ in range(pool_size):
            pool.append(self._new_game(current_game_id))
            current_game_id += 1
            games_started += 1
        
        # Main loop: each iteration does one MCTS move for all active games
        while pool:
            # Separate endgame vs MCTS games
            endgame_games = []
            mcts_games = []
            for g in pool:
                if self.dtw_calculator and self.dtw_calculator.is_endgame(g['board']):
                    endgame_games.append(g)
                else:
                    mcts_games.append(g)
            
            # DTW search for endgame positions
            finished_indices = set()
            for game in endgame_games:
                dtw_result = self.dtw_calculator.calculate_dtw(game['board'], need_best_move=False)
                if dtw_result is not None:
                    result_code, _, _ = dtw_result
                    game['done'] = True
                    self._finalize_game_dtw(game, result_code, all_data)
                    finished_indices.add(id(game))
                else:
                    mcts_games.append(game)
            
            # MCTS for non-endgame games
            if mcts_games:
                game_temps = []
                for g in mcts_games:
                    move_count = len(g['history'])
                    game_temps.append(self.temperature if move_count < 8 else 0)
                
                batch_temp = sum(game_temps) / len(game_temps) if game_temps else 0
                
                results = self.mcts.search_parallel(
                    mcts_games,
                    temperature=batch_temp,
                    add_noise=(batch_temp > 0)
                )
                
                # Apply moves
                for game, (policy, action) in zip(mcts_games, results):
                    board = game['board']
                    
                    canonical_tensor, canonical_policy = BoardEncoder.to_training_tensor(board, policy)
                    game['history'].append({
                        'state': canonical_tensor,
                        'policy': canonical_policy,
                        'player': board.current_player
                    })
                    
                    row, col = action // 9, action % 9
                    try:
                        board.make_move(row, col)
                    except Exception:
                        legal = board.get_legal_moves()
                        if legal:
                            board.make_move(*legal[0])
                    
                    if board.is_game_over():
                        game['done'] = True
                        self._finalize_game(game, all_data)
                        finished_indices.add(id(game))
            
            # Replace finished games with new ones (continuous batching)
            if finished_indices:
                new_pool = []
                newly_completed = 0
                for g in pool:
                    if id(g) in finished_indices:
                        newly_completed += 1
                        # Replace with new game if we still need more
                        if games_started < num_games:
                            new_pool.append(self._new_game(current_game_id))
                            current_game_id += 1
                            games_started += 1
                    else:
                        new_pool.append(g)
                pool = new_pool
                games_completed += newly_completed
                pbar.update(newly_completed)
                pbar.set_postfix({"samples": len(all_data), "active": len(pool)})
        
        pbar.close()
        
        _parallel_timing['total_time'] += time.perf_counter() - start_time
        _parallel_timing['games'] += num_games
        _parallel_timing['moves'] += len(all_data)
        _parallel_timing['mcts_overhead'] = _parallel_timing['total_time'] - _parallel_timing['network_time']
        
        return all_data
