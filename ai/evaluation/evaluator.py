"""
Unified evaluation engine for Ultra Tic-Tac-Toe.

All match-playing logic lives here. Other modules import from this file:
  - training eval (run_evaluation_suite)
  - Gradio baseline test (baseline.py)
  - Elo checkpoint matches (elo.py)
  - sim strength comparison (sim_strength_test.py)
"""
import random
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm
from game import Board
from ai.baselines import RandomAgent, HeuristicAgent, MinimaxAgent
from ai.training.parallel_mcts import ParallelMCTS


# ─── Default opening settings ────────────────────────────────────
OPENING_MOVES = 8
OPENING_TEMP = 1.0


# ─── Match result ────────────────────────────────────────────────

@dataclass
class MatchResult:
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    games: int = 0

    @property
    def win_rate_a(self):
        return self.wins_a / self.games if self.games else 0.0

    @property
    def win_rate_b(self):
        return self.wins_b / self.games if self.games else 0.0

    @property
    def draw_rate(self):
        return self.draws / self.games if self.games else 0.0

    @property
    def score_a(self):
        """Score for player A (win=1, draw=0.5, loss=0)."""
        return (self.wins_a + 0.5 * self.draws) / self.games if self.games else 0.5


# ─── Core parallel match engine ──────────────────────────────────

def play_parallel(
    mcts_a: ParallelMCTS,
    mcts_b: Optional[ParallelMCTS],
    num_games: int,
    baseline_b=None,
    opening_moves: int = OPENING_MOVES,
    opening_temp: float = OPENING_TEMP,
    random_opening_plies: int = 0,
    on_batch_done=None,
) -> MatchResult:
    """Core parallel match engine. Plays num_games between two players.

    Player A is always mcts_a (network + MCTS).
    Player B is either mcts_b (network + MCTS) or baseline_b (CPU agent).
    Exactly one of mcts_b or baseline_b must be provided.

    Args:
        mcts_a: ParallelMCTS for player A.
        mcts_b: ParallelMCTS for player B (network vs network).
        baseline_b: Agent with select_action(board) for player B (model vs baseline).
        num_games: Total games to play (alternating first player).
        opening_moves: Number of opening moves with opening_temp.
        opening_temp: Temperature for opening moves (0 after).
        random_opening_plies: Random moves at start for diversity (Elo matches).
        on_batch_done: Optional callback(result_so_far, batch_games) for streaming.

    Returns:
        MatchResult with win/draw counts from player A's perspective.
    """
    assert (mcts_b is not None) ^ (baseline_b is not None), \
        "Provide exactly one of mcts_b or baseline_b"

    result = MatchResult()

    # Init all games
    active_games = []
    for i in range(num_games):
        active_games.append({
            'board': Board(),
            'a_is_p1': (i % 2 == 0),
            'move_count': 0,
            'done': False,
        })

    # Random opening plies (for Elo diversity)
    if random_opening_plies > 0:
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
        a_turn = []
        b_turn = []

        for g in active_games:
            if g['done']:
                continue
            is_a_turn = (g['board'].current_player == 1) == g['a_is_p1']
            if is_a_turn:
                a_turn.append(g)
            else:
                b_turn.append(g)

        # Player A moves (always MCTS)
        if a_turn:
            temp = opening_temp if a_turn[0]['move_count'] < opening_moves else 0.0
            results_a = mcts_a.search_parallel(a_turn, temperature=temp, add_noise=False)
            for g, (_, action) in zip(a_turn, results_a):
                g['board'].make_move(action // 9, action % 9)
                g['move_count'] += 1
                if g['board'].winner not in (None, -1):
                    g['done'] = True

        # Player B moves (MCTS or baseline)
        if b_turn:
            if mcts_b is not None:
                temp = opening_temp if b_turn[0]['move_count'] < opening_moves else 0.0
                results_b = mcts_b.search_parallel(b_turn, temperature=temp, add_noise=False)
                for g, (_, action) in zip(b_turn, results_b):
                    g['board'].make_move(action // 9, action % 9)
                    g['move_count'] += 1
                    if g['board'].winner not in (None, -1):
                        g['done'] = True
            else:
                for g in b_turn:
                    action = baseline_b.select_action(g['board'])
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

    if on_batch_done:
        on_batch_done(result, active_games)

    return result


# ─── Convenience wrappers ────────────────────────────────────────

def evaluate_vs_baseline(network, baseline, num_games=500, num_simulations=0,
                         parallel_games=None, dtw_calculator=None,
                         opening_moves=OPENING_MOVES, opening_temp=OPENING_TEMP):
    """Model (MCTS) vs baseline agent. Returns dict with win_rate, etc."""
    mcts = ParallelMCTS(
        network=network, num_simulations=num_simulations,
        c_puct=1.0, dtw_calculator=dtw_calculator
    )
    r = play_parallel(
        mcts_a=mcts, mcts_b=None, baseline_b=baseline,
        num_games=num_games,
        opening_moves=opening_moves, opening_temp=opening_temp,
    )
    return {
        'model_wins': r.wins_a, 'baseline_wins': r.wins_b,
        'draws': r.draws, 'games': r.games,
        'win_rate': r.win_rate_a,
    }


def play_agents(agent_a, agent_b, num_games, random_opening_plies=6):
    """Sequential match between two agents with select_action(board) -> int.
    Used by Elo evaluate_latest / run_tournament for non-GPU agents (e.g. NNUE).
    """
    result = MatchResult()

    for i in range(num_games):
        board = Board()
        a_is_p1 = (i % 2 == 0)

        for _ in range(random_opening_plies):
            legal = board.get_legal_moves()
            if not legal or board.winner not in (None, -1):
                break
            r, c = random.choice(legal)
            board.make_move(r, c)

        while board.winner in (None, -1):
            legal = board.get_legal_moves()
            if not legal:
                break
            is_a_turn = (board.current_player == 1) == a_is_p1
            action = agent_a.select_action(board) if is_a_turn else agent_b.select_action(board)
            board.make_move(action // 9, action % 9)

        result.games += 1
        w = board.winner
        if w == 3 or w in (None, -1):
            result.draws += 1
        elif (w == 1 and a_is_p1) or (w == 2 and not a_is_p1):
            result.wins_a += 1
        else:
            result.wins_b += 1

    return result


def play_networks_parallel(network_a, network_b, num_games,
                           num_simulations=0, random_opening_plies=6,
                           opening_moves=OPENING_MOVES, opening_temp=OPENING_TEMP):
    """Network vs network (GPU-accelerated parallel). Returns MatchResult."""
    mcts_a = ParallelMCTS(network=network_a, num_simulations=num_simulations, c_puct=1.0)
    mcts_b = ParallelMCTS(network=network_b, num_simulations=num_simulations, c_puct=1.0)
    return play_parallel(
        mcts_a=mcts_a, mcts_b=mcts_b, baseline_b=None,
        num_games=num_games,
        random_opening_plies=random_opening_plies,
        opening_moves=opening_moves, opening_temp=opening_temp,
    )


def play_sims_parallel(network, sims_a, sims_b, num_games,
                       opening_moves=OPENING_MOVES, opening_temp=OPENING_TEMP):
    """Same network, different sim counts. Returns MatchResult."""
    mcts_a = ParallelMCTS(network=network, num_simulations=sims_a, c_puct=1.0)
    mcts_b = ParallelMCTS(network=network, num_simulations=sims_b, c_puct=1.0)
    return play_parallel(
        mcts_a=mcts_a, mcts_b=mcts_b, baseline_b=None,
        num_games=num_games,
        opening_moves=opening_moves, opening_temp=opening_temp,
    )


# ─── Training evaluation suite ───────────────────────────────────

def run_evaluation_suite(network, num_games_random=1000, num_games_sims=200,
                         eval_sims=50, dtw_calculator=None):
    """
    Lean evaluation suite (~30s total):
      1. Sanity check: 0 sims vs random (fast, ~3s)
      2. Primary metric: eval_sims vs minimax2 (actual strength, ~25s)

    Returns:
        dict of metrics ready for wandb logging
    """
    metrics = {}

    # --- Sanity: raw policy vs random ---
    r_rand = evaluate_vs_baseline(
        network, RandomAgent(), num_games=num_games_random, num_simulations=0
    )
    metrics['eval/vs_random_winrate_new'] = r_rand['win_rate'] * 100

    # --- Primary: MCTS vs minimax2 ---
    mm2 = MinimaxAgent(depth=2)
    r_sims = evaluate_vs_baseline(
        network, mm2, num_games=num_games_sims,
        num_simulations=eval_sims, dtw_calculator=dtw_calculator
    )
    metrics[f'eval/vs_minimax2_{eval_sims}sim_winrate_new'] = r_sims['win_rate'] * 100
    metrics[f'eval/vs_minimax2_{eval_sims}sim_drawrate_new'] = r_sims['draws'] / num_games_sims * 100

    # Elo from sims eval
    from ai.evaluation.elo import winrate_to_elo_diff
    MINIMAX2_ELO = 1620

    sims_score = r_sims['win_rate'] + 0.5 * (r_sims['draws'] / num_games_sims)
    if 0 < sims_score < 1:
        metrics['elo/current_new'] = MINIMAX2_ELO + winrate_to_elo_diff(sims_score)

    return metrics
