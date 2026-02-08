"""
Practical Tablebase Builder

Instead of enumerating all positions theoretically, this builder:
1. Plays random games to reach endgame positions
2. Solves each endgame position exactly
3. Stores results with canonical hashing (symmetry reduction)

This is more practical and guarantees reachable positions only.
"""

import os
import pickle
import random
import time
from typing import Dict, Tuple, Optional, Set
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from game import Board
from .solver import TablebaseSolver, count_empty
from .enumerator import PositionEnumerator
from .tablebase import Tablebase, CompactTablebase


class EndgameTablebaseBuilder:
    """
    Build endgame tablebase by playing games and solving endgame positions.
    
    Strategy:
    1. Play random games until reaching endgame threshold
    2. Solve each unique endgame position
    3. Use symmetry reduction to minimize storage
    4. Incrementally save progress
    """
    
    def __init__(
        self,
        endgame_threshold: int = 15,
        save_interval: int = 10000,
        save_path: str = 'tablebase/endgame.pkl'
    ):
        """
        Args:
            endgame_threshold: Max empty playable cells for tablebase
            save_interval: Save progress every N positions
            save_path: Path to save tablebase
        """
        self.endgame_threshold = endgame_threshold
        self.save_interval = save_interval
        self.save_path = save_path
        
        self.solver = TablebaseSolver(max_depth=25)
        self.reachability_checker = ReachabilityChecker(max_depth=4)
        
        # Storage
        self.positions: Dict[int, Tuple[int, int]] = {}  # hash -> (result, dtw)
        self.seen_hashes: Set[int] = set()
        
        # Stats
        self.stats = defaultdict(int)
        
        # Load existing if available
        self._load_existing()
    
    def _load_existing(self):
        """Load existing tablebase if available."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)
                self.positions = data.get('positions', {})
                self.seen_hashes = set(self.positions.keys())
                self.stats = defaultdict(int, data.get('stats', {}))
                print(f"✓ Loaded existing tablebase: {len(self.positions)} positions")
            except Exception as e:
                print(f"⚠ Failed to load existing: {e}")
    
    def build(
        self,
        num_games: int = 100000,
        target_positions: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[int, Tuple[int, int]]:
        """
        Build tablebase by playing random games.
        
        Args:
            num_games: Number of random games to play
            target_positions: Stop when reaching this many positions
            verbose: Show progress
            
        Returns:
            Dict mapping canonical hashes to (result, dtw)
        """
        start_time = time.time()
        start_positions = len(self.positions)
        
        print("=" * 60)
        print(f"Building Endgame Tablebase (≤{self.endgame_threshold} empty cells)")
        print("=" * 60)
        print(f"Starting with {start_positions} existing positions")
        
        games_played = 0
        positions_added = 0
        
        pbar = tqdm(total=num_games, desc="Games") if verbose else None
        
        try:
            while games_played < num_games:
                if target_positions and len(self.positions) >= target_positions:
                    print(f"\n✓ Reached target: {target_positions} positions")
                    break
                
                # Play one game and collect endgame positions
                new_positions = self._play_game_and_solve()
                positions_added += new_positions
                games_played += 1
                
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'positions': len(self.positions),
                        'new': new_positions
                    })
                
                # Periodic save
                if games_played % self.save_interval == 0:
                    self._save()
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        
        finally:
            if pbar:
                pbar.close()
            self._save()
        
        elapsed = time.time() - start_time
        new_total = len(self.positions) - start_positions
        
        print(f"\n{'=' * 60}")
        print(f"Build Complete!")
        print(f"  Games played: {games_played}")
        print(f"  New positions: {new_total}")
        print(f"  Total positions: {len(self.positions)}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Rate: {games_played/elapsed:.1f} games/s")
        print(f"{'=' * 60}")
        
        return self.positions
    
    def _play_game_and_solve(self) -> int:
        """
        Play a random game and solve all endgame positions encountered.
        
        Returns:
            Number of new positions added
        """
        board = Board()
        new_count = 0
        
        # Play until game ends
        while board.winner is None:
            empty = count_empty(board)
            
            # Once in endgame range, solve and store
            if empty <= self.endgame_threshold:
                if self._solve_and_store(board.clone()):
                    new_count += 1
            
            # Make random move
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            move = random.choice(legal_moves)
            board.make_move(move[0], move[1])
        
        self.stats['games_played'] += 1
        
        return new_count
    
    def _solve_and_store(self, board: Board) -> bool:
        """
        Solve position and store in tablebase.
        
        Returns:
            True if new position was added
        """
        # Get canonical hash
        board_hash = self.solver._hash_board(board)
        
        # Skip if already seen
        if board_hash in self.seen_hashes:
            self.stats['cache_hits'] += 1
            return False
        
        # Check reachability (4-move backward DFS)
        if not self.reachability_checker.is_reachable(board):
            self.stats['unreachable'] += 1
            return False
        
        self.seen_hashes.add(board_hash)
        
        # Solve position with BFS + memoization
        result, dtw, best_move = self.solver.solve(board)
        
        # Store result (result, dtw, best_move)
        self.positions[board_hash] = (result, dtw, best_move)
        
        # Update stats
        self.stats['positions_solved'] += 1
        if result == 1:
            self.stats['wins'] += 1
        elif result == -1:
            self.stats['losses'] += 1
        else:
            self.stats['draws'] += 1
        
        return True
    
    def _save(self):
        """Save current tablebase."""
        os.makedirs(os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else '.', exist_ok=True)
        
        with open(self.save_path, 'wb') as f:
            pickle.dump({
                'positions': self.positions,
                'stats': dict(self.stats),
                'endgame_threshold': self.endgame_threshold
            }, f)
    
    def get_stats(self) -> dict:
        """Get builder statistics."""
        return {
            'total_positions': len(self.positions),
            'endgame_threshold': self.endgame_threshold,
            **dict(self.stats)
        }
    
    def export_compact(self, path: str):
        """Export to compact numpy format."""
        compact = CompactTablebase()
        compact.build_from_dict(self.positions)
        compact.save(path)
        print(f"✓ Exported compact tablebase: {compact.get_size_mb():.2f} MB")


class TargetedTablebaseBuilder(EndgameTablebaseBuilder):
    """
    Build tablebase targeting specific game phases.
    
    Uses smarter game generation to find diverse endgame positions.
    """
    
    def _play_game_and_solve(self) -> int:
        """Play a game with varied opening to find diverse endgames."""
        board = Board()
        new_count = 0
        
        # Random opening moves
        opening_length = random.randint(10, 50)
        
        for _ in range(opening_length):
            if board.winner is not None:
                break
            
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            # Mix of random and "interesting" moves
            if random.random() < 0.3:
                # Prefer center-ish moves
                scored_moves = []
                for move in legal_moves:
                    r, c = move
                    # Prefer center of sub-boards
                    sub_r, sub_c = r % 3, c % 3
                    center_dist = abs(sub_r - 1) + abs(sub_c - 1)
                    score = random.random() - center_dist * 0.1
                    scored_moves.append((score, move))
                scored_moves.sort(reverse=True)
                move = scored_moves[0][1]
            else:
                move = random.choice(legal_moves)
            
            board.make_move(move[0], move[1])
        
        # Continue to endgame, solving along the way
        while board.winner is None:
            empty = count_empty(board)
            
            if empty <= self.endgame_threshold:
                if self._solve_and_store(board.clone()):
                    new_count += 1
            
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            move = random.choice(legal_moves)
            board.make_move(move[0], move[1])
        
        self.stats['games_played'] += 1
        
        return new_count


def main():
    """Build tablebase from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build endgame tablebase')
    parser.add_argument('--games', type=int, default=10000, help='Number of games')
    parser.add_argument('--threshold', type=int, default=15, help='Endgame threshold')
    parser.add_argument('--output', type=str, default='tablebase/endgame.pkl', help='Output path')
    parser.add_argument('--target', type=int, default=None, help='Target number of positions')
    
    args = parser.parse_args()
    
    builder = TargetedTablebaseBuilder(
        endgame_threshold=args.threshold,
        save_path=args.output
    )
    
    builder.build(
        num_games=args.games,
        target_positions=args.target
    )
    
    # Export compact version
    compact_path = args.output.replace('.pkl', '.npz')
    builder.export_compact(compact_path)


if __name__ == '__main__':
    main()
