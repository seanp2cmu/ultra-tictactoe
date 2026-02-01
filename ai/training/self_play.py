"""Self-play worker with DTW endgame support."""
from typing import List, Tuple, Optional
import numpy as np

from game import Board
from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator
from ai.utils import BatchPredictor
from ai.mcts import AlphaZeroAgent

class SelfPlayWorker:
    """Self-play worker for generating training data."""
    
    def __init__(
        self,
        network: AlphaZeroNet,
        dtw_calculator: Optional[DTWCalculator] = None,
        batch_predictor: Optional[BatchPredictor] = None,
        num_simulations: int = 100,
        temperature: float = 1.0,
        use_dtw_endgame: bool = True,
        dtw_max_depth: int = 12,
        hot_cache_size: int = 50000,
        cold_cache_size: int = 500000,
        use_symmetry: bool = True
    ) -> None:
        self.agent = AlphaZeroAgent(network, num_simulations=num_simulations, temperature=temperature)
        self.network: AlphaZeroNet = network
        self.use_dtw_endgame = use_dtw_endgame
        
        if use_dtw_endgame:
            if dtw_calculator is None:
                self.dtw_calculator = DTWCalculator(
                    max_depth=dtw_max_depth,
                    use_cache=True,
                    hot_size=hot_cache_size,
                    cold_size=cold_cache_size,
                    use_symmetry=use_symmetry
                )
            else:
                self.dtw_calculator = dtw_calculator
        else:
            self.dtw_calculator = None
    
    def play_game(self) -> List[Tuple]:
        """Play one self-play game."""
        board = Board()
        game_data = []
        step = 0
        
        while board.winner is None:
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            current_player = board.current_player
            dtw = None
            
            if self.use_dtw_endgame and self.dtw_calculator:
                if self.dtw_calculator.is_endgame(board):
                    result_data = self.dtw_calculator.calculate_dtw(board)
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
                            continue
            
            root = self.agent.search(board)
            
            action_probs = np.zeros(81, dtype=np.float32)
            for action, child in root.children.items():
                action_probs[action] = child.visits
            
            action_probs = action_probs / np.sum(action_probs)
            
            state = self._board_to_input(board)
            game_data.append((state, action_probs, current_player, dtw))
            
            action = self.agent.select_action(board, temperature=self.agent.temperature)
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
