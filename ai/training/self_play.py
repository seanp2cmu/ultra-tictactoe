"""Self-play worker with DTW endgame support."""
from typing import List, Tuple, Optional
import numpy as np
import time

from game import Board
from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator
from ai.mcts import AlphaZeroAgent

# 디버깅용 타이밍 변수
DEBUG_TIMING = True
SLOW_THRESHOLD = 5.0  # 5초 이상만 기록
_timing_stats = {
    'mcts_search': 0.0,
    'mcts_count': 0,
    'dtw_endgame': 0.0,
    'dtw_endgame_count': 0,
    'dtw_midgame': 0.0,
    'dtw_midgame_count': 0,
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
        'dtw_midgame': 0.0,
        'dtw_midgame_count': 0,
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
            dtw_calculator=self.dtw_calculator
        )
    
    def play_game(self, verbose: bool = False) -> List[Tuple]:
        """Play one self-play game."""
        global _timing_stats
        board = Board()
        game_data = []
        step = 0
        
        while board.winner is None:
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
                            
                            state = self._board_to_input(board)
                            game_data.append((state, action_probs, current_player, dtw))
                            
                            board.make_move(best_move[0], best_move[1])
                            step += 1
                            _timing_stats['total_steps'] += 1
                            continue
            
            # === MCTS Search ===
            t0 = time.time()
            root = self.agent.search(board)
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
            
            # === DTW Midgame Check ===
            selected_move = None
            if self.dtw_calculator.is_midgame(board):
                top_k = 5
                top_actions = np.argsort(action_probs)[-top_k:][::-1]
                candidate_moves = [(a // 9, a % 9) for a in top_actions if action_probs[a] > 0]
                
                if candidate_moves:
                    t0 = time.time()
                    check_result = self.dtw_calculator.check_candidate_moves(board, candidate_moves)
                    elapsed = time.time() - t0
                    _timing_stats['dtw_midgame'] += elapsed
                    _timing_stats['dtw_midgame_count'] += 1
                    if elapsed > SLOW_THRESHOLD:
                        _timing_stats['slow_steps'].append((step, f'dtw_midgame(empty={empty_cells})', elapsed))
                        _log_slow_step(step, f'dtw_midgame(empty={empty_cells})', elapsed)
                    
                    if check_result['winning_move']:
                        selected_move = check_result['winning_move']
                        action = selected_move[0] * 9 + selected_move[1]
                        action_probs = np.zeros(81, dtype=np.float32)
                        action_probs[action] = 1.0
                    elif check_result['safe_moves']:
                        losing_actions = set(m[0] * 9 + m[1] for m in check_result['losing_moves'])
                        for la in losing_actions:
                            action_probs[la] = 0.0
                        if np.sum(action_probs) > 0:
                            action_probs = action_probs / np.sum(action_probs)
            
            t0 = time.time()
            state = self._board_to_input(board)
            _timing_stats['board_to_input'] += time.time() - t0
            
            game_data.append((state, action_probs, current_player, dtw))
            _timing_stats['total_steps'] += 1
            
            if selected_move:
                row, col = selected_move
            elif self.agent.temperature == 0:
                action = int(np.argmax(action_probs))
                row, col = action // 9, action % 9
            else:
                action = int(np.random.choice(81, p=action_probs))
                row, col = action // 9, action % 9
            
            if (row, col) not in legal_moves:
                break
            
            board.make_move(row, col)
            step += 1
        
        if board.winner is None or board.winner == 3:
            winner = None
        else:
            winner = board.winner
        
        training_data = []
        for state, policy, player, dtw in game_data:
            if winner is None:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            
            training_data.append((state, policy, value, dtw))
        
        return training_data
    
    def _board_to_input(self, board: Board) -> np.ndarray:
        """Convert board to network input (7 channels)."""
        tensor = self.network.model._board_to_tensor(board)
        state = tensor.squeeze(0).cpu().numpy()
        return state
