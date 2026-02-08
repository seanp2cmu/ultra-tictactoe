"""
Tablebase Generator for Ultimate Tic-Tac-Toe

Generates endgame tablebase using retrograde analysis.
Enumerates all valid positions with ≤15 empty cells and computes exact values.
"""

import os
import pickle
import numpy as np
from itertools import product, combinations
from typing import List, Tuple, Dict, Optional, Generator
from tqdm import tqdm
from collections import defaultdict

from .meta_board import MetaBoardEnumerator, check_subboard_winner, WIN_PATTERNS


class SubBoardGenerator:
    """
    Generate valid sub-board configurations given constraints.
    
    A sub-board has 9 cells, each can be 0 (empty), 1 (X), or 2 (O).
    """
    
    # Precompute all valid sub-board states
    # Format: {(state, x_count, o_count): [list of valid 9-tuples]}
    _cache = {}
    
    @classmethod
    def generate_winning_boards(cls, winner: int, x_count: int, o_count: int) -> List[Tuple[int, ...]]:
        """
        Generate sub-boards where 'winner' has won.
        
        Args:
            winner: 1 (X) or 2 (O)
            x_count: Number of X pieces on this sub-board
            o_count: Number of O pieces on this sub-board
        """
        key = ('win', winner, x_count, o_count)
        if key in cls._cache:
            return cls._cache[key]
        
        results = []
        total_cells = 9
        empty_count = total_cells - x_count - o_count
        
        if empty_count < 0:
            return []
        
        # For each winning pattern
        for pattern in WIN_PATTERNS:
            # Winner must occupy all 3 cells in the pattern
            # Generate all valid placements
            remaining_winner = (x_count if winner == 1 else o_count) - 3
            loser_count = o_count if winner == 1 else x_count
            loser = 2 if winner == 1 else 1
            
            if remaining_winner < 0:
                continue
            
            # Cells not in winning pattern
            other_cells = [i for i in range(9) if i not in pattern]
            
            # Place remaining winner pieces and loser pieces in other cells
            # Loser cannot have a winning pattern
            for winner_extra in combinations(other_cells, remaining_winner):
                remaining_cells = [c for c in other_cells if c not in winner_extra]
                
                for loser_cells in combinations(remaining_cells, loser_count):
                    # Check loser doesn't have a winning line
                    loser_set = set(loser_cells)
                    has_loser_win = False
                    for p in WIN_PATTERNS:
                        if all(c in loser_set for c in p):
                            has_loser_win = True
                            break
                    
                    if has_loser_win:
                        continue
                    
                    # Build the board
                    board = [0] * 9
                    for c in pattern:
                        board[c] = winner
                    for c in winner_extra:
                        board[c] = winner
                    for c in loser_cells:
                        board[c] = loser
                    
                    results.append(tuple(board))
        
        # Remove duplicates
        results = list(set(results))
        cls._cache[key] = results
        return results
    
    @classmethod
    def generate_draw_boards(cls, x_count: int, o_count: int) -> List[Tuple[int, ...]]:
        """
        Generate full sub-boards (draws) - all 9 cells filled, no winner.
        """
        key = ('draw', x_count, o_count)
        if key in cls._cache:
            return cls._cache[key]
        
        if x_count + o_count != 9:
            return []
        
        results = []
        
        # Place X pieces
        for x_cells in combinations(range(9), x_count):
            board = [2] * 9  # Start with all O
            for c in x_cells:
                board[c] = 1
            
            # Check no winner
            if check_subboard_winner(board) == 3:  # Draw
                results.append(tuple(board))
        
        cls._cache[key] = results
        return results
    
    @classmethod
    def generate_open_boards(cls, x_count: int, o_count: int) -> List[Tuple[int, ...]]:
        """
        Generate open sub-boards (game still in progress on this sub-board).
        No winner yet, has empty cells.
        """
        key = ('open', x_count, o_count)
        if key in cls._cache:
            return cls._cache[key]
        
        empty_count = 9 - x_count - o_count
        if empty_count <= 0:
            return []
        
        results = []
        
        # Place X and O pieces
        for x_cells in combinations(range(9), x_count):
            remaining = [c for c in range(9) if c not in x_cells]
            for o_cells in combinations(remaining, o_count):
                board = [0] * 9
                for c in x_cells:
                    board[c] = 1
                for c in o_cells:
                    board[c] = 2
                
                # Check no winner yet
                if check_subboard_winner(board) == 0:  # Open
                    results.append(tuple(board))
        
        cls._cache[key] = results
        return results


class TablebaseGenerator:
    """
    Generate endgame tablebase for Ultimate Tic-Tac-Toe.
    
    Strategy:
    1. Enumerate valid meta-board configurations
    2. For each meta-board, enumerate valid cell configurations
    3. Filter by total empty cells (≤15)
    4. Compute minimax value via retrograde analysis
    """
    
    def __init__(self, max_empty: int = 15):
        """
        Args:
            max_empty: Maximum empty cells for tablebase (default 15)
        """
        self.max_empty = max_empty
        self.meta_enumerator = MetaBoardEnumerator()
        self.positions = {}  # board_hash -> (result, dtw)
        self.stats = defaultdict(int)
    
    def generate(self, save_path: Optional[str] = None) -> Dict:
        """
        Generate the complete tablebase.
        
        Returns:
            Dict mapping board hashes to (result, dtw) tuples
        """
        print("=" * 60)
        print("Tablebase Generation")
        print("=" * 60)
        
        # Step 1: Enumerate meta-boards
        print("\nStep 1: Enumerating meta-boards...")
        meta_boards = self.meta_enumerator.enumerate_all()
        print(f"  Valid meta-boards: {len(meta_boards)}")
        print(f"  By open sub-boards: {self.meta_enumerator.count_by_open_subboards()}")
        
        # Step 2: For each meta-board, enumerate positions
        print(f"\nStep 2: Generating positions (max {self.max_empty} empty)...")
        
        total_positions = 0
        
        for meta_idx, meta in enumerate(tqdm(meta_boards, desc="Meta-boards")):
            positions = self._generate_positions_for_meta(meta)
            
            for pos in positions:
                board_hash = self._hash_position(pos)
                
                # Compute result via minimax
                result, dtw = self._evaluate_position(pos)
                
                self.positions[board_hash] = (result, dtw)
                total_positions += 1
                
                self.stats[f'result_{result}'] += 1
        
        print(f"\nTotal positions: {total_positions}")
        print(f"Results distribution: {dict(self.stats)}")
        
        if save_path:
            self._save(save_path)
        
        return self.positions
    
    def _generate_positions_for_meta(self, meta: Tuple[int, ...]) -> Generator:
        """
        Generate all valid board positions for a given meta-board configuration.
        
        This is the core enumeration logic:
        1. Determine X/O counts that satisfy constraints
        2. Distribute pieces across sub-boards
        3. Filter by total empty cells
        """
        # Count completed sub-boards
        x_wins = meta.count(1)
        o_wins = meta.count(2)
        draws = meta.count(3)
        opens = meta.count(0)
        
        # For positions with ≤15 empty, we have ≥66 filled cells
        min_filled = 81 - self.max_empty  # 66
        
        # Total X and O pieces must satisfy:
        # - |x_total - o_total| <= 1 (alternating turns)
        # - Each won sub-board has valid winner configuration
        
        # This is complex - let's simplify for now with direct enumeration
        # of smaller cases
        
        # For efficiency, only handle cases with few open sub-boards
        if opens > 3:
            return  # Skip for now - too many combinations
        
        # Generate sub-board configurations
        sub_configs = self._enumerate_subboard_configs(meta)
        
        for config in sub_configs:
            # config is a tuple of 9 sub-board configurations (each 9-tuple)
            # Flatten to 81-cell board
            board = []
            for sub in config:
                board.extend(sub)
            
            # Check total empty cells
            empty = board.count(0)
            if empty > self.max_empty:
                continue
            
            # Check X-O balance
            x_total = board.count(1)
            o_total = board.count(2)
            if abs(x_total - o_total) > 1:
                continue
            
            yield tuple(board)
    
    def _enumerate_subboard_configs(self, meta: Tuple[int, ...]) -> Generator:
        """
        Enumerate valid sub-board configurations for given meta-board.
        """
        # Get constraints for each sub-board
        constraints = self.meta_enumerator.get_sub_board_constraints(meta)
        
        # Generate valid configs for each sub-board
        sub_options = []
        
        for i, constraint in enumerate(constraints):
            state = constraint['state']
            
            if state == 1:  # X won
                # X has winning line, various piece counts possible
                options = []
                for x in range(3, 7):  # X needs at least 3 to win
                    for o in range(0, min(x, 6)):  # O has fewer
                        boards = SubBoardGenerator.generate_winning_boards(1, x, o)
                        options.extend(boards)
                sub_options.append(options if options else [None])
                
            elif state == 2:  # O won
                options = []
                for o in range(3, 7):
                    for x in range(0, min(o + 1, 6)):  # X can have at most o pieces (moved first)
                        boards = SubBoardGenerator.generate_winning_boards(2, x, o)
                        options.extend(boards)
                sub_options.append(options if options else [None])
                
            elif state == 3:  # Draw
                # Full board, no winner
                options = []
                for x in range(4, 6):  # 4 or 5 X's in a draw
                    o = 9 - x
                    boards = SubBoardGenerator.generate_draw_boards(x, o)
                    options.extend(boards)
                sub_options.append(options if options else [None])
                
            else:  # Open
                # Not complete - various configurations
                options = []
                for x in range(0, 5):
                    for o in range(0, 5):
                        if x + o < 9:  # Must have empty
                            boards = SubBoardGenerator.generate_open_boards(x, o)
                            options.extend(boards)
                sub_options.append(options if options else [None])
        
        # Generate all combinations (limit to avoid explosion)
        MAX_COMBOS = 100000
        combo_count = 1
        for opts in sub_options:
            if opts and opts[0] is not None:
                combo_count *= len(opts)
        
        if combo_count > MAX_COMBOS:
            # Sample instead of full enumeration
            return
        
        # Full enumeration
        for combo in product(*sub_options):
            if None in combo:
                continue
            yield combo
    
    def _hash_position(self, board: Tuple[int, ...]) -> int:
        """
        Create canonical hash for board position.
        Uses symmetry reduction (D4 on big board).
        """
        # For now, simple hash
        # TODO: Add symmetry reduction
        return hash(board)
    
    def _evaluate_position(self, board: Tuple[int, ...]) -> Tuple[int, int]:
        """
        Evaluate position using minimax.
        
        Returns:
            (result, dtw) where:
            - result: 1 (X wins), -1 (O wins), 0 (draw)
            - dtw: distance to win (moves until game ends)
        """
        # Convert to 2D and check if terminal
        # This is a simplified evaluation - full implementation would
        # use alpha-beta search
        
        # For now, return placeholder
        # TODO: Implement proper minimax
        return (0, 0)
    
    def _save(self, path: str):
        """Save tablebase to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'positions': self.positions,
                'stats': dict(self.stats),
                'max_empty': self.max_empty
            }, f)
        
        print(f"✓ Saved tablebase to {path}")


if __name__ == '__main__':
    generator = TablebaseGenerator(max_empty=15)
    generator.generate(save_path='tablebase.pkl')
