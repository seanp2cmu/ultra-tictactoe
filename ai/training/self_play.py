"""Self-play worker with DTW endgame support."""
from typing import List, Tuple, Optional
import numpy as np
import time

from game import Board
from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator
from ai.mcts import AlphaZeroAgent
from utils import BoardSymmetry

# 디버깅용 타이밍 변수
DEBUG_TIMING = True
SLOW_THRESHOLD = 5.0  # 5초 이상만 기록
_timing_stats = {
    'mcts_search': 0.0,
    'mcts_count': 0,
    'dtw_endgame': 0.0,
    'dtw_endgame_count': 0,
    'board_to_input': 0.0,
    'total_steps': 0,
    'slow_steps': [],  # (step, section, elapsed) for steps > 5s
}
_slow_log_file = None

def set_slow_log_file(path: str):
    global _slow_log_file
    _slow_log_file = path
    # Clear file
    with open(path, 'w') as f:
        f.write("=== Slow Steps Log (>5s) ===\n")

def _log_slow_step(step: int, section: str, elapsed: float):
    if _slow_log_file:
        with open(_slow_log_file, 'a') as f:
            f.write(f"Step {step}: {section} = {elapsed:.2f}s\n")

def reset_timing_stats():
    global _timing_stats
    _timing_stats = {
        'mcts_search': 0.0,
        'mcts_count': 0,
        'dtw_endgame': 0.0,
        'dtw_endgame_count': 0,
        'board_to_input': 0.0,
        'total_steps': 0,
        'slow_steps': [],
    }
    if _slow_log_file:
        with open(_slow_log_file, 'a') as f:
            f.write("\n--- New Iteration ---\n")

def get_timing_stats():
    return _timing_stats.copy()

class SelfPlayWorker:
    """Self-play worker for generating training data."""
    
    def __init__(
        self,
        network: AlphaZeroNet,
        dtw_calculator: Optional[DTWCalculator] = None,
        num_simulations: int = 100,
        temperature: float = 1.0,
        endgame_threshold: int = 15,
        hot_cache_size: int = 50000,
        cold_cache_size: int = 500000
    ) -> None:
        self.network: AlphaZeroNet = network
        
        if dtw_calculator is None:
            self.dtw_calculator = DTWCalculator(
                use_cache=True,
                hot_size=hot_cache_size,
                cold_size=cold_cache_size,
                endgame_threshold=endgame_threshold
            )
        else:
            self.dtw_calculator = dtw_calculator
        
        self.agent = AlphaZeroAgent(
            network, 
            num_simulations=num_simulations, 
            temperature=temperature,
            batch_size=32,  # GPU에서 더 효율적
            dtw_calculator=self.dtw_calculator
        )
    
    def play_game(self, verbose: bool = False) -> List[Tuple]:
        """Play one self-play game."""
        global _timing_stats
        board = Board()
        game_data = []
        step = 0
        
        while board.winner in (None, -1):  # BoardCy uses -1 for no winner
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            current_player = board.current_player
            dtw = None
            empty_cells = board.count_playable_empty_cells()
            
            # === DTW Endgame Check ===
            if self.dtw_calculator.is_endgame(board):
                    t0 = time.time()
                    result_data = self.dtw_calculator.calculate_dtw(board)
                    elapsed = time.time() - t0
                    _timing_stats['dtw_endgame'] += elapsed
                    _timing_stats['dtw_endgame_count'] += 1
                    if elapsed > SLOW_THRESHOLD:
                        _timing_stats['slow_steps'].append((step, f'dtw_endgame(empty={empty_cells})', elapsed))
                        _log_slow_step(step, f'dtw_endgame(empty={empty_cells})', elapsed)
                    
                    if result_data is not None:
                        result, dtw, best_move = result_data
                        is_winning = (result == 1)
                        
                        if is_winning and dtw <= 5 and best_move:
                            action_probs = np.zeros(81, dtype=np.float32)
                            action = best_move[0] * 9 + best_move[1]
                            action_probs[action] = 1.0
                            
                            # Use canonical form for training
                            state, action_probs = self._board_to_canonical_input(board, action_probs)
                            game_data.append((state, action_probs, current_player, dtw))
                            
                            board.make_move(best_move[0], best_move[1])
                            step += 1
                            _timing_stats['total_steps'] += 1
                            continue
            
            # === MCTS Search ===
            t0 = time.time()
            root = self.agent.search(board, add_noise=True)  # Dirichlet noise for exploration
            elapsed = time.time() - t0
            _timing_stats['mcts_search'] += elapsed
            _timing_stats['mcts_count'] += 1
            if elapsed > SLOW_THRESHOLD:
                _timing_stats['slow_steps'].append((step, f'mcts_search(empty={empty_cells})', elapsed))
                _log_slow_step(step, f'mcts_search(empty={empty_cells})', elapsed)
            
            action_probs = np.zeros(81, dtype=np.float32)
            for action, child in root.children.items():
                action_probs[action] = child.visits
            
            if np.sum(action_probs) == 0:
                break
            action_probs = action_probs / np.sum(action_probs)
            
            t0 = time.time()
            # Use canonical form for training (8x data efficiency)
            state, canonical_probs = self._board_to_canonical_input(board, action_probs)
            _timing_stats['board_to_input'] += time.time() - t0
            
            game_data.append((state, canonical_probs, current_player, dtw))
            _timing_stats['total_steps'] += 1
            
            # Temperature only for first 8 moves (AlphaZero style)
            TEMP_MOVES = 8
            if step < TEMP_MOVES and self.agent.temperature > 0:
                # Apply temperature for exploration in opening
                action = int(np.random.choice(81, p=action_probs))
            else:
                # Greedy selection after opening
                action = int(np.argmax(action_probs))
            row, col = action // 9, action % 9
            
            if (row, col) not in legal_moves:
                break
            
            board.make_move(row, col)
            step += 1
        
        if board.winner in (None, -1, 3):  # BoardCy uses -1 for no winner
            winner = None
        else:
            winner = board.winner
        
        training_data = []
        for state, policy, player, dtw in game_data:
            # Value range: 0~1 (loss=0, draw=0.5, win=1) - AlphaZero style
            if winner is None:
                value = 0.5  # draw
            elif winner == player:
                value = 1.0  # win
            else:
                value = 0.0  # loss
            
            training_data.append((state, policy, value, dtw))
        
        return training_data
    
    def _board_to_input(self, board: Board) -> np.ndarray:
        """Convert board to network input (7 channels)."""
        tensor = self.network.model._board_to_tensor(board)
        state = tensor.squeeze(0).cpu().numpy()
        return state
    
    def _board_to_canonical_input(self, board: Board, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert board and policy to canonical form for training.
        
        This normalizes symmetric positions to the same representation,
        effectively increasing training data efficiency by 8x.
        
        Args:
            board: Board object
            policy: Policy probabilities (81,)
            
        Returns:
            (canonical_state, canonical_policy)
        """
        boards_arr, completed_arr, trans_policy, player = BoardSymmetry.canonicalize_training_sample(board, policy)
        
        # Build canonical board object for tensor conversion
        canonical_board = Board()
        for r in range(9):
            for c in range(9):
                if boards_arr[r, c] != 0:
                    canonical_board.set_cell(r, c, int(boards_arr[r, c]))
        # Handle both Cython BoardCy and Python Board
        if hasattr(canonical_board, 'set_completed_boards_2d'):
            canonical_board.set_completed_boards_2d(completed_arr.tolist())
        else:
            canonical_board.completed_boards = completed_arr.tolist()
        canonical_board.current_player = player
        canonical_board.last_move = board.last_move  # Keep last_move for valid moves channel
        
        # Apply same transform to last_move if present
        if board.last_move is not None:
            _, _, transform_idx = BoardSymmetry.get_canonical_with_transform(board)
            if transform_idx != 0:
                transforms = BoardSymmetry._build_transforms()
                old_idx = board.last_move[0] * 9 + board.last_move[1]
                new_idx = np.where(transforms[transform_idx] == old_idx)[0]
                if len(new_idx) > 0:
                    canonical_board.last_move = (new_idx[0] // 9, new_idx[0] % 9)
        
        state = self._board_to_input(canonical_board)
        return state, trans_policy
