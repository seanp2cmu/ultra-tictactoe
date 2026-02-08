"""
NNUE Training Data Generator

Generates training data for NNUE by using the AlphaZero model as a teacher.
The AlphaZero model provides high-quality position evaluations through MCTS.
"""

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional, List, Tuple
from game import Board
from ai.core import AlphaZeroNet
from ai.mcts import AlphaZeroAgent
from ai.endgame import DTWCalculator
from .features import NNUEFeatureExtractor


class NNUEDataGenerator:
    """
    Generate NNUE training data using AlphaZero as teacher.
    
    Data format:
    - features: (288,) float32 - NNUE input features
    - value: float32 - Target value from AlphaZero/MCTS
    - game_result: float32 - Actual game outcome (for validation)
    """
    
    def __init__(
        self,
        model_path: str,
        num_simulations: int = 200,
        device: str = 'cpu'
    ):
        """
        Args:
            model_path: Path to trained AlphaZero model
            num_simulations: MCTS simulations per position
            device: torch device
        """
        self.device = device
        self.num_simulations = num_simulations
        
        # Load AlphaZero model
        self.network = AlphaZeroNet(device=device)
        self.network.load(model_path)
        print(f"✓ Loaded AlphaZero model from {model_path}")
        
        # DTW calculator for endgame
        self.dtw_calculator = DTWCalculator(
            use_cache=True,
            endgame_threshold=15
        )
        
        # Feature extractor
        self.feature_extractor = NNUEFeatureExtractor()
    
    def generate_from_self_play(
        self,
        num_games: int,
        temperature: float = 0.5,
        save_path: Optional[str] = None
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Generate training data from self-play games.
        
        Args:
            num_games: Number of games to play
            temperature: Move selection temperature
            save_path: Optional path to save data
            
        Returns:
            List of (features, mcts_value, game_result) tuples
        """
        agent = AlphaZeroAgent(
            network=self.network,
            num_simulations=self.num_simulations,
            c_puct=1.0,
            temperature=temperature,
            batch_size=8,
            dtw_calculator=self.dtw_calculator
        )
        
        all_data = []
        
        for game_idx in tqdm(range(num_games), desc="Generating games"):
            game_data = self._play_game(agent, temperature)
            all_data.extend(game_data)
        
        print(f"Generated {len(all_data)} positions from {num_games} games")
        
        if save_path:
            self._save_data(all_data, save_path)
        
        return all_data
    
    def _play_game(
        self,
        agent: AlphaZeroAgent,
        temperature: float
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Play a single game and collect training data.
        
        Returns:
            List of (features, mcts_value, placeholder_result) for each position
        """
        board = Board()
        positions = []
        
        step = 0
        while board.winner is None and step < 81:
            # Extract features BEFORE the move
            features = self.feature_extractor.extract(board)
            current_player = board.current_player
            
            # Get MCTS evaluation
            root = agent.search(board)
            mcts_value = root.value()  # -1 to 1 range
            
            # Convert to 0-1 range
            mcts_value_01 = (mcts_value + 1) / 2
            
            # Store position data
            positions.append({
                'features': features,
                'mcts_value': mcts_value_01,
                'player': current_player
            })
            
            # Select move
            if step < 8:
                # Temperature for first 8 moves
                action = agent.select_action(board, temperature=temperature)
            else:
                # Greedy after
                action = agent.select_action(board, temperature=0)
            
            row, col = action // 9, action % 9
            board.make_move(row, col)
            step += 1
        
        # Determine game result
        if board.winner is None or board.winner == 3:
            result = 0.5  # Draw
        else:
            result = 1.0 if board.winner == 1 else 0.0
        
        # Assign game results to positions
        game_data = []
        for pos in positions:
            # Result from this player's perspective
            if pos['player'] == 1:
                pos_result = result
            else:
                pos_result = 1.0 - result
            
            game_data.append((
                pos['features'],
                pos['mcts_value'],
                pos_result
            ))
        
        return game_data
    
    def generate_from_positions(
        self,
        boards: List[Board],
        save_path: Optional[str] = None
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Generate training data from existing positions.
        
        Args:
            boards: List of Board objects to evaluate
            save_path: Optional path to save data
            
        Returns:
            List of (features, value) tuples
        """
        agent = AlphaZeroAgent(
            network=self.network,
            num_simulations=self.num_simulations,
            c_puct=1.0,
            temperature=0,
            batch_size=8,
            dtw_calculator=self.dtw_calculator
        )
        
        data = []
        
        for board in tqdm(boards, desc="Evaluating positions"):
            if board.winner is not None:
                continue
            
            features = self.feature_extractor.extract(board)
            
            # Get MCTS value
            root = agent.search(board)
            mcts_value = (root.value() + 1) / 2  # Convert to 0-1
            
            data.append((features, mcts_value))
        
        if save_path:
            self._save_data(data, save_path)
        
        return data
    
    def _save_data(self, data: list, path: str):
        """Save generated data to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Saved {len(data)} samples to {path}")
    
    @staticmethod
    def load_data(path: str) -> list:
        """Load generated data from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded {len(data)} samples from {path}")
        return data


def main():
    """Generate NNUE training data from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate NNUE training data')
    parser.add_argument('--model', type=str, required=True, help='Path to AlphaZero model')
    parser.add_argument('--games', type=int, default=1000, help='Number of games')
    parser.add_argument('--sims', type=int, default=200, help='MCTS simulations')
    parser.add_argument('--output', type=str, default='nnue_data.pkl', help='Output path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    generator = NNUEDataGenerator(
        model_path=args.model,
        num_simulations=args.sims,
        device=args.device
    )
    
    generator.generate_from_self_play(
        num_games=args.games,
        save_path=args.output
    )


if __name__ == '__main__':
    main()
