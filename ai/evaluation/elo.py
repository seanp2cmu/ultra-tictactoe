"""
Elo rating system for checkpoint-vs-checkpoint evaluation.

Maintains a pool of checkpoints and computes Elo ratings via
round-robin matches between the newest checkpoint and recent opponents.
Supports both AlphaZero (GPU/MCTS) and NNUE (C++ search) agents.

Usage:
    tracker = EloTracker(save_path="model/run_id/elo.json")
    tracker.register_checkpoint("iter_010", path="model/run_id/ckpt_010.pt")
    metrics = tracker.evaluate_latest(make_agent_fn, num_games=50)
    # metrics = {'elo/current': 1523, 'elo/delta': +23, ...}
"""
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple


# ─── Elo math ────────────────────────────────────────────────────

K_FACTOR = 32
INITIAL_ELO = 1500
DRAW_SCORE = 0.5

# Anchor: Minimax-4 = 1800, iter_050 (0 sims) vs Minimax-4 winrate = 4.4%
# Elo diff = 400 * log10(0.956 / 0.044) ≈ +386 → iter_050 ≈ 1414
ANCHOR_MINIMAX4_ELO = 1800
ANCHOR_ITER050_ELO = 1414  # 1800 - 386


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400))


def update_elo(rating: float, expected: float, actual: float, k: float = K_FACTOR) -> float:
    """Update Elo rating given expected and actual score."""
    return rating + k * (actual - expected)


def winrate_to_elo_diff(winrate: float) -> float:
    """Convert winrate (0-1) to Elo difference. winrate > 0.5 means positive diff."""
    if winrate <= 0.0:
        return -800.0
    if winrate >= 1.0:
        return 800.0
    return 400.0 * math.log10(winrate / (1.0 - winrate))


# ─── Match result ────────────────────────────────────────────────

@dataclass
class MatchResult:
    player_a: str
    player_b: str
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    games: int = 0


# ─── Checkpoint entry ────────────────────────────────────────────

@dataclass
class CheckpointEntry:
    name: str
    path: str
    elo: float = INITIAL_ELO
    games_played: int = 0
    iteration: int = 0


# ─── Elo Tracker ─────────────────────────────────────────────────

class EloTracker:
    """Tracks Elo ratings across training checkpoints.
    
    Args:
        save_path: Path to persist Elo state as JSON.
        max_opponents: Max number of recent checkpoints to play against.
        num_games: Default games per matchup.
    """
    
    def __init__(self, save_path: str, max_opponents: int = 5, num_games: int = 50):
        self.save_path = save_path
        self.max_opponents = max_opponents
        self.num_games = num_games
        self.checkpoints: Dict[str, CheckpointEntry] = {}
        self.match_history: List[dict] = []
        self._load()
    
    # ─── Checkpoint management ───────────────────────────────
    
    def register_checkpoint(self, name: str, path: str, iteration: int = 0):
        """Register a new checkpoint for Elo tracking."""
        if name not in self.checkpoints:
            self.checkpoints[name] = CheckpointEntry(
                name=name, path=path, iteration=iteration
            )
            self._save()
    
    def get_opponents(self, exclude: str) -> List[CheckpointEntry]:
        """Get recent checkpoints to play against (excluding the given one)."""
        others = [c for c in self.checkpoints.values() if c.name != exclude]
        # Sort by iteration descending, take most recent
        others.sort(key=lambda c: c.iteration, reverse=True)
        return others[:self.max_opponents]
    
    @property
    def latest_checkpoint(self) -> Optional[CheckpointEntry]:
        """Get the most recently registered checkpoint."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints.values(), key=lambda c: c.iteration)
    
    # ─── Match playing ───────────────────────────────────────
    
    @staticmethod
    def play_match(agent_a, agent_b, num_games: int,
                   board_cls=None, random_opening_plies: int = 6) -> MatchResult:
        """Play a match between two agents.
        
        Args:
            agent_a: Agent with select_action(board) -> int
            agent_b: Agent with select_action(board) -> int
            num_games: Number of games to play
            board_cls: Board class to use (auto-detected if None)
            random_opening_plies: Random moves at start for diversity
            
        Returns:
            MatchResult with win/draw counts
        """
        if board_cls is None:
            try:
                import uttt_cpp
                board_cls = uttt_cpp.Board
            except ImportError:
                from game import Board as board_cls
        
        result = MatchResult(
            player_a=getattr(agent_a, 'name', 'A'),
            player_b=getattr(agent_b, 'name', 'B'),
        )
        
        for game_num in range(num_games):
            board = board_cls()
            a_is_p1 = (game_num % 2 == 0)
            
            # Random opening for diversity
            for _ in range(random_opening_plies):
                legal = board.get_legal_moves()
                if not legal or board.winner not in (None, -1):
                    break
                r, c = random.choice(legal)
                board.make_move(r, c)
            
            # Play game
            while board.winner in (None, -1):
                legal = board.get_legal_moves()
                if not legal:
                    break
                
                is_a_turn = (board.current_player == 1) == a_is_p1
                if is_a_turn:
                    action = agent_a.select_action(board)
                else:
                    action = agent_b.select_action(board)
                
                board.make_move(action // 9, action % 9)
            
            # Record result
            result.games += 1
            if board.winner == 3 or board.winner in (None, -1):
                result.draws += 1
            elif (board.winner == 1 and a_is_p1) or (board.winner == 2 and not a_is_p1):
                result.wins_a += 1
            else:
                result.wins_b += 1
        
        return result
    
    @staticmethod
    def play_match_parallel(network_a, network_b, num_games: int,
                            num_simulations: int = 0,
                            random_opening_plies: int = 6) -> MatchResult:
        """Play a match using ParallelMCTS batch inference (GPU-accelerated).
        
        All games run simultaneously — much faster than sequential play_match.
        
        Args:
            network_a: AlphaZeroNet instance for player A
            network_b: AlphaZeroNet instance for player B
            num_games: Number of games to play
            num_simulations: MCTS simulations per move (0 = raw policy)
            random_opening_plies: Random moves at start for diversity
        """
        from game import Board
        from ai.training.parallel_mcts import ParallelMCTS
        
        mcts_a = ParallelMCTS(network=network_a, num_simulations=num_simulations, c_puct=1.0)
        mcts_b = ParallelMCTS(network=network_b, num_simulations=num_simulations, c_puct=1.0)
        
        result = MatchResult(player_a='A', player_b='B')
        
        # Init all games
        active_games = []
        for i in range(num_games):
            board = Board()
            a_is_p1 = (i % 2 == 0)
            active_games.append({
                'board': board,
                'a_is_p1': a_is_p1,
                'move_count': 0,
                'done': False,
            })
        
        # Random openings
        for g in active_games:
            for _ in range(random_opening_plies):
                legal = g['board'].get_legal_moves()
                if not legal or g['board'].winner not in (None, -1):
                    break
                r, c = random.choice(legal)
                g['board'].make_move(r, c)
                g['move_count'] += 1
        
        # Play until all done
        while any(not g['done'] for g in active_games):
            a_turn_games = []
            b_turn_games = []
            
            for g in active_games:
                if g['done']:
                    continue
                is_a_turn = (g['board'].current_player == 1) == g['a_is_p1']
                if is_a_turn:
                    a_turn_games.append(g)
                else:
                    b_turn_games.append(g)
            
            # Batch moves for A
            if a_turn_games:
                results_a = mcts_a.search_parallel(
                    a_turn_games, temperature=0.0, add_noise=False
                )
                for g, (_, action) in zip(a_turn_games, results_a):
                    g['board'].make_move(action // 9, action % 9)
                    g['move_count'] += 1
                    if g['board'].winner not in (None, -1):
                        g['done'] = True
            
            # Batch moves for B
            if b_turn_games:
                results_b = mcts_b.search_parallel(
                    b_turn_games, temperature=0.0, add_noise=False
                )
                for g, (_, action) in zip(b_turn_games, results_b):
                    g['board'].make_move(action // 9, action % 9)
                    g['move_count'] += 1
                    if g['board'].winner not in (None, -1):
                        g['done'] = True
        
        # Tally results
        for g in active_games:
            result.games += 1
            w = g['board'].winner
            if w == 3 or w in (None, -1):
                result.draws += 1
            elif (w == 1 and g['a_is_p1']) or (w == 2 and not g['a_is_p1']):
                result.wins_a += 1
            else:
                result.wins_b += 1
        
        return result
    
    # ─── Elo evaluation ──────────────────────────────────────
    
    def evaluate_latest(self, make_agent_fn: Callable[[str], object],
                        num_games: int = None) -> Dict[str, float]:
        """Evaluate the latest checkpoint against recent opponents.
        
        Args:
            make_agent_fn: Function that takes a checkpoint path and returns
                          an agent with select_action(board) -> int.
            num_games: Games per matchup (uses self.num_games if None).
            
        Returns:
            Dict of metrics for wandb logging:
                elo/current, elo/delta, elo/games_played,
                elo/vs_<opponent>_score, ...
        """
        num_games = num_games or self.num_games
        latest = self.latest_checkpoint
        if latest is None:
            return {}
        
        opponents = self.get_opponents(exclude=latest.name)
        if not opponents:
            return {'elo/current': latest.elo, 'elo/delta': 0.0}
        
        old_elo = latest.elo
        latest_agent = make_agent_fn(latest.path)
        metrics = {}
        
        for opp in opponents:
            opp_agent = make_agent_fn(opp.path)
            
            match = self.play_match(latest_agent, opp_agent, num_games)
            
            # Compute actual scores (1 for win, 0.5 for draw, 0 for loss)
            if match.games == 0:
                continue
            
            score_a = (match.wins_a + DRAW_SCORE * match.draws) / match.games
            score_b = (match.wins_b + DRAW_SCORE * match.draws) / match.games
            
            # Update both ratings
            exp_a = expected_score(latest.elo, opp.elo)
            exp_b = expected_score(opp.elo, latest.elo)
            
            latest.elo = update_elo(latest.elo, exp_a, score_a)
            opp.elo = update_elo(opp.elo, exp_b, score_b)
            
            latest.games_played += match.games
            opp.games_played += match.games
            
            # Record match
            self.match_history.append({
                'a': latest.name, 'b': opp.name,
                'wins_a': match.wins_a, 'wins_b': match.wins_b,
                'draws': match.draws, 'games': match.games,
                'elo_a': latest.elo, 'elo_b': opp.elo,
            })
            
            metrics[f'elo/vs_{opp.name}_score'] = score_a
        
        metrics['elo/current'] = latest.elo
        metrics['elo/delta'] = latest.elo - old_elo
        metrics['elo/games_played'] = latest.games_played
        metrics['elo/num_checkpoints'] = len(self.checkpoints)
        
        # wandb tables
        metrics.update(self._wandb_tables())
        
        # Clean up agents if they have a clear method
        if hasattr(latest_agent, 'clear'):
            latest_agent.clear()
        
        self._save()
        return metrics
    
    # ─── Full tournament ─────────────────────────────────────
    
    def run_tournament(self, make_agent_fn: Callable[[str], object],
                       num_games: int = None) -> Dict[str, float]:
        """Run round-robin tournament between ALL checkpoints.
        
        Useful for recalibrating Elo ratings from scratch.
        Resets all ratings to INITIAL_ELO before starting.
        
        Returns:
            Dict mapping checkpoint name -> final Elo rating
        """
        num_games = num_games or self.num_games
        names = sorted(self.checkpoints.keys(),
                       key=lambda n: self.checkpoints[n].iteration)
        
        if len(names) < 2:
            return {n: INITIAL_ELO for n in names}
        
        # Reset ratings
        for c in self.checkpoints.values():
            c.elo = INITIAL_ELO
            c.games_played = 0
        self.match_history.clear()
        
        # Build agents
        agents = {n: make_agent_fn(self.checkpoints[n].path) for n in names}
        
        # Round-robin
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                match = self.play_match(agents[name_a], agents[name_b], num_games)
                
                if match.games == 0:
                    continue
                
                ca, cb = self.checkpoints[name_a], self.checkpoints[name_b]
                score_a = (match.wins_a + DRAW_SCORE * match.draws) / match.games
                score_b = (match.wins_b + DRAW_SCORE * match.draws) / match.games
                
                exp_a = expected_score(ca.elo, cb.elo)
                exp_b = expected_score(cb.elo, ca.elo)
                
                ca.elo = update_elo(ca.elo, exp_a, score_a)
                cb.elo = update_elo(cb.elo, exp_b, score_b)
                ca.games_played += match.games
                cb.games_played += match.games
                
                self.match_history.append({
                    'a': name_a, 'b': name_b,
                    'wins_a': match.wins_a, 'wins_b': match.wins_b,
                    'draws': match.draws, 'games': match.games,
                    'elo_a': ca.elo, 'elo_b': cb.elo,
                })
        
        # Clean up
        for a in agents.values():
            if hasattr(a, 'clear'):
                a.clear()
        
        self._save()
        return {n: self.checkpoints[n].elo for n in names}
    
    # ─── wandb logging ────────────────────────────────────────
    
    def _wandb_tables(self) -> Dict:
        """Build wandb Table objects for leaderboard and recent matches."""
        try:
            import wandb
        except ImportError:
            return {}
        
        metrics = {}
        
        # Leaderboard table
        ranked = sorted(self.checkpoints.values(), key=lambda c: c.elo, reverse=True)
        leaderboard = wandb.Table(
            columns=['Rank', 'Checkpoint', 'Elo', 'Games', 'Iteration'],
        )
        for i, c in enumerate(ranked, 1):
            leaderboard.add_data(i, c.name, round(c.elo, 1), c.games_played, c.iteration)
        metrics['elo/leaderboard'] = leaderboard
        
        # Recent matches table (last 20)
        if self.match_history:
            matches = wandb.Table(
                columns=['Player A', 'Player B', 'Wins A', 'Wins B', 'Draws', 'Elo A', 'Elo B'],
            )
            for m in self.match_history[-20:]:
                matches.add_data(
                    m['a'], m['b'], m['wins_a'], m['wins_b'], m['draws'],
                    round(m['elo_a'], 1), round(m['elo_b'], 1),
                )
            metrics['elo/matches'] = matches
        
        return metrics
    
    # ─── Persistence ─────────────────────────────────────────
    
    def _save(self):
        """Save state to JSON."""
        os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)
        state = {
            'checkpoints': {
                name: asdict(entry)
                for name, entry in self.checkpoints.items()
            },
            'match_history': self.match_history[-200:],  # Keep last 200 matches
        }
        with open(self.save_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load(self):
        """Load state from JSON if it exists."""
        if not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path) as f:
                state = json.load(f)
            for name, data in state.get('checkpoints', {}).items():
                self.checkpoints[name] = CheckpointEntry(**data)
            self.match_history = state.get('match_history', [])
        except (json.JSONDecodeError, TypeError, KeyError):
            pass  # Corrupted file, start fresh
    
    # ─── Summary ─────────────────────────────────────────────
    
    def summary(self) -> str:
        """Return a formatted Elo leaderboard string."""
        if not self.checkpoints:
            return "No checkpoints registered."
        
        ranked = sorted(self.checkpoints.values(), key=lambda c: c.elo, reverse=True)
        lines = ["╔══════════════════════════════════════════════╗",
                 "║           Elo Rating Leaderboard             ║",
                 "╠══════════════════════════════════════════════╣"]
        for i, c in enumerate(ranked, 1):
            lines.append(f"║  {i:>2}. {c.name:<20} Elo: {c.elo:>7.1f}  ({c.games_played}g) ║")
        lines.append("╚══════════════════════════════════════════════╝")
        return "\n".join(lines)
