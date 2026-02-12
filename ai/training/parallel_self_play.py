"""Parallel self-play: multiple games with shared GPU inference."""
from typing import List, Tuple, Optional, Dict
import numpy as np
import time

from game import Board
from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator
from ai.mcts.node import Node
from utils import BoardSymmetry


# Timing stats for parallel self-play
_parallel_timing = {
    'total_time': 0.0,
    'network_time': 0.0,
    'mcts_overhead': 0.0,
    'batches': 0,
    'games': 0,
    'moves': 0,
}

def reset_parallel_timing():
    global _parallel_timing
    _parallel_timing = {
        'total_time': 0.0,
        'network_time': 0.0,
        'mcts_overhead': 0.0,
        'batches': 0,
        'games': 0,
        'moves': 0,
    }

def get_parallel_timing():
    return _parallel_timing.copy()


class ParallelMCTS:
    """MCTS that can collect leaves from multiple games for batch inference."""
    
    def __init__(
        self,
        network: AlphaZeroNet,
        num_simulations: int = 100,
        c_puct: float = 1.0,
        dtw_calculator: Optional[DTWCalculator] = None
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dtw_calculator = dtw_calculator
    
    def search_parallel(
        self,
        games: List[Dict],
        temperature: float = 1.0,
        add_noise: bool = True
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Run MCTS on multiple games in parallel, sharing GPU inference.
        
        Args:
            games: List of game dicts with 'board' key
            temperature: Temperature for move selection
            add_noise: Add Dirichlet noise at root
            
        Returns:
            List of (policy, action) for each game
        """
        # Initialize roots for each game
        roots = []
        for game in games:
            root = Node(game['board'])
            roots.append(root)
        
        global _parallel_timing
        
        # Initial expansion of all roots
        boards = [g['board'] for g in games]
        t0 = time.perf_counter()
        policies, _ = self.network.predict_batch(boards)
        _parallel_timing['network_time'] += time.perf_counter() - t0
        
        for i, root in enumerate(roots):
            policy = policies[i]
            if add_noise:
                noise = np.random.dirichlet([0.3] * 81)
                policy = 0.75 * policy + 0.25 * noise
            root.expand(dict(enumerate(policy)))
        
        # Run simulations
        sims_per_batch = max(1, self.num_simulations // 10)  # 10 batches
        
        for _ in range(0, self.num_simulations, sims_per_batch):
            batch_size = min(sims_per_batch, self.num_simulations)
            
            # Collect leaves from all games
            all_leaves = []  # (game_idx, node, search_path)
            all_boards = []
            
            for game_idx, root in enumerate(roots):
                for _ in range(batch_size // len(games) + 1):
                    node = root
                    search_path = [node]
                    
                    while node.is_expanded() and not node.is_terminal():
                        _, node = node.select_child(self.c_puct)
                        search_path.append(node)
                    
                    if not node.is_terminal() and not node.is_expanded():
                        all_leaves.append((game_idx, node, search_path))
                        all_boards.append(node.board)
            
            if not all_boards:
                continue
            
            # Batch inference for all leaves
            t0 = time.perf_counter()
            policies_batch, values_batch = self.network.predict_batch(all_boards)
            _parallel_timing['network_time'] += time.perf_counter() - t0
            
            # Expand and backup
            for i, (game_idx, node, search_path) in enumerate(all_leaves):
                policy = policies_batch[i]
                value = 2.0 * values_batch[i].item() - 1.0  # 0~1 -> -1~1
                
                # DTW check
                if self.dtw_calculator and self.dtw_calculator.is_endgame(node.board):
                    cached = self.dtw_calculator.lookup_cache(node.board)
                    if cached is not None:
                        result, _, _ = cached
                        value = float(result)
                
                node.expand(dict(enumerate(policy)))
                
                # Backup
                for path_node in reversed(search_path):
                    path_node.update(value)
                    value = -value
        
        # Select moves for each game
        results = []
        for root in roots:
            visits = np.array([
                root.children[a].visits if a in root.children else 0
                for a in range(81)
            ])
            
            if visits.sum() == 0:
                # No visits - use uniform over legal moves
                legal = root.board.get_legal_moves()
                policy = np.zeros(81)
                for r, c in legal:
                    policy[r * 9 + c] = 1.0 / len(legal)
                action = legal[np.random.randint(len(legal))]
                action = action[0] * 9 + action[1]
            elif temperature == 0:
                action = int(np.argmax(visits))
                policy = np.zeros(81)
                policy[action] = 1.0
            else:
                visits_temp = visits ** (1.0 / temperature)
                total = visits_temp.sum()
                if total == 0:
                    policy = np.ones(81) / 81
                else:
                    policy = visits_temp / total
                # Ensure sum to 1
                policy = policy / policy.sum()
                action = np.random.choice(81, p=policy)
            
            results.append((policy, action))
        
        return results


class ParallelSelfPlayWorker:
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
    
    def play_games(self, num_games: int, disable_tqdm: bool = False) -> List[Tuple]:
        """
        Play multiple games and return training data.
        
        Returns:
            List of (state, policy, value, dtw) tuples
        """
        from tqdm import tqdm
        global _parallel_timing
        
        all_data = []
        games_completed = 0
        total_moves = 0
        
        start_time = time.perf_counter()
        
        pbar = tqdm(total=num_games, desc="Parallel self-play", 
                    disable=disable_tqdm, ncols=100, leave=False)
        
        while games_completed < num_games:
            # Start batch of parallel games
            batch_size = min(self.parallel_games, num_games - games_completed)
            games = [{'board': Board(), 'history': [], 'done': False} for _ in range(batch_size)]
            
            # Play until all games done
            while any(not g['done'] for g in games):
                active_games = [g for g in games if not g['done']]
                
                if not active_games:
                    break
                
                # MCTS for all active games
                results = self.mcts.search_parallel(
                    active_games,
                    temperature=self.temperature,
                    add_noise=True
                )
                
                # Apply moves
                for game, (policy, action) in zip(active_games, results):
                    board = game['board']
                    
                    # Get canonical form for training
                    boards_arr, completed_arr, transform_idx = BoardSymmetry.get_canonical_with_transform(board)
                    canonical_policy = BoardSymmetry.transform_policy(policy, transform_idx)
                    
                    # Store training data
                    game['history'].append({
                        'state': self._board_to_input(board),
                        'policy': canonical_policy,
                        'player': board.current_player
                    })
                    
                    # Make move
                    row, col = action // 9, action % 9
                    if (row, col) in board.get_legal_moves():
                        board.make_move(row, col)
                    else:
                        # Invalid move - pick random legal
                        legal = board.get_legal_moves()
                        if legal:
                            board.make_move(*legal[0])
                    
                    # Check game end
                    if board.winner not in (None, -1) or not board.get_legal_moves():
                        game['done'] = True
                        
                        # Assign values based on game result
                        if board.winner in (None, -1, 3):
                            result = 0.5  # draw
                        else:
                            result = 1.0 if board.winner == 1 else 0.0
                        
                        # Create training samples
                        for i, step in enumerate(game['history']):
                            # Value from perspective of player at that step
                            if step['player'] == 1:
                                value = result
                            else:
                                value = 1.0 - result
                            
                            all_data.append((
                                step['state'],
                                step['policy'],
                                value,
                                None  # DTW placeholder
                            ))
            
            games_completed += batch_size
            _parallel_timing['batches'] += 1
            pbar.update(batch_size)
            pbar.set_postfix({"samples": len(all_data)})
        
        pbar.close()
        
        # Record final timing stats
        _parallel_timing['total_time'] += time.perf_counter() - start_time
        _parallel_timing['games'] += num_games
        _parallel_timing['moves'] += len(all_data)
        _parallel_timing['mcts_overhead'] = _parallel_timing['total_time'] - _parallel_timing['network_time']
        
        return all_data
    
    def _board_to_input(self, board: Board) -> np.ndarray:
        """Convert board to network input tensor."""
        tensor = np.zeros((7, 9, 9), dtype=np.float32)
        boards = np.array(board.to_array(), dtype=np.float32)
        
        current_player = board.current_player
        opponent_player = 3 - current_player
        
        # Channel 0: current player pieces
        tensor[0] = (boards == current_player).astype(np.float32)
        # Channel 1: opponent pieces
        tensor[1] = (boards == opponent_player).astype(np.float32)
        # Channel 2: empty cells
        tensor[2] = (boards == 0).astype(np.float32)
        
        # Channel 3-4: completed boards
        if hasattr(board, 'get_completed_boards_2d'):
            completed = board.get_completed_boards_2d()
        else:
            completed = board.completed_boards
        
        for br in range(3):
            for bc in range(3):
                status = completed[br][bc]
                sr, sc = br * 3, bc * 3
                if status == current_player:
                    tensor[3, sr:sr+3, sc:sc+3] = 1.0
                elif status == opponent_player:
                    tensor[4, sr:sr+3, sc:sc+3] = 1.0
        
        # Channel 5: legal moves (active board)
        for r, c in board.get_legal_moves():
            tensor[5, r, c] = 1.0
        
        # Channel 6: last move
        if board.last_move:
            tensor[6, board.last_move[0], board.last_move[1]] = 1.0
        
        return tensor
