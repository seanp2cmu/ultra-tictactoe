"""Self-play worker for generating training data."""
from typing import List, Tuple, Optional
import numpy as np
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
    
    def play_games(self, num_games: int, disable_tqdm: bool = False, 
                    game_id_start: int = 0) -> List[Tuple]:
        """
        Play multiple games and return training data.
        
        Returns:
            List of (state, policy, value, game_id) tuples
        """
        from tqdm import tqdm
        global _parallel_timing
        
        all_data = []
        games_completed = 0
        current_game_id = game_id_start
        
        start_time = time.perf_counter()
        
        pbar = tqdm(total=num_games, desc="Self-play", 
                    disable=disable_tqdm, ncols=100, leave=False, position=1)
        
        while games_completed < num_games:
            batch_size = min(self.parallel_games, num_games - games_completed)
            games = []
            for _ in range(batch_size):
                games.append({
                    'board': Board(), 
                    'history': [], 
                    'done': False,
                    'game_id': current_game_id
                })
                current_game_id += 1
            
            while any(not g['done'] for g in games):
                active_games = [g for g in games if not g['done']]
                
                if not active_games:
                    break
                
                endgame_games = []
                mcts_games = []
                for g in active_games:
                    if self.dtw_calculator and self.dtw_calculator.is_endgame(g['board']):
                        endgame_games.append(g)
                    else:
                        mcts_games.append(g)
                
                # DTW search for endgame
                for game in endgame_games:
                    board = game['board']
                    dtw_result = self.dtw_calculator.calculate_dtw(board, need_best_move=False)
                    
                    if dtw_result is not None:
                        result, dtw, _ = dtw_result
                        game['done'] = True
                        
                        if result == 1:
                            final_value = 1.0
                        elif result == -1:
                            final_value = 0.0
                        else:
                            final_value = 0.5
                        
                        for step in game['history']:
                            if step['player'] == board.current_player:
                                value = final_value
                            else:
                                value = 1.0 - final_value
                            
                            all_data.append((
                                step['state'],
                                step['policy'],
                                value,
                                game['game_id']
                            ))
                    else:
                        mcts_games.append(game)
                
                # MCTS for non-endgame games
                mcts_results = []
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
                    mcts_results = list(zip(mcts_games, results))
                
                # Apply moves
                for game, (policy, action) in mcts_results:
                    board = game['board']
                    
                    canonical_tensor, canonical_policy = BoardEncoder.to_training_tensor(board, policy)
                    
                    game['history'].append({
                        'state': canonical_tensor,
                        'policy': canonical_policy,
                        'player': board.current_player
                    })
                    
                    row, col = action // 9, action % 9
                    if (row, col) in board.get_legal_moves():
                        board.make_move(row, col)
                    else:
                        legal = board.get_legal_moves()
                        if legal:
                            board.make_move(*legal[0])
                    
                    if board.winner not in (None, -1) or not board.get_legal_moves():
                        game['done'] = True
                        
                        if board.winner in (None, -1, 3):
                            result = 0.5
                        else:
                            result = 1.0 if board.winner == 1 else 0.0
                        
                        for step in game['history']:
                            if step['player'] == 1:
                                value = result
                            else:
                                value = 1.0 - result
                            
                            all_data.append((
                                step['state'],
                                step['policy'],
                                value,
                                game['game_id']
                            ))
            
            games_completed += batch_size
            _parallel_timing['batches'] += 1
            pbar.update(batch_size)
            pbar.set_postfix({"samples": len(all_data)})
        
        pbar.close()
        
        _parallel_timing['total_time'] += time.perf_counter() - start_time
        _parallel_timing['games'] += num_games
        _parallel_timing['moves'] += len(all_data)
        _parallel_timing['mcts_overhead'] = _parallel_timing['total_time'] - _parallel_timing['network_time']
        
        return all_data
