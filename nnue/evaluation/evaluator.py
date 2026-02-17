"""
NNUE evaluation against baseline opponents.
Plays games using C++ NNUE engine vs random/heuristic/minimax.
"""
import random
import time
from tqdm import tqdm
import uttt_cpp
from ai.baselines import RandomAgent, HeuristicAgent, MinimaxAgent
from nnue.agent import NNUEAgent


RANDOM_OPENING_PLIES = 6


def _play_random_opening(board, num_plies):
    """Play random opening moves to diversify game starts."""
    for _ in range(num_plies):
        legal = board.get_legal_moves()
        if not legal or board.winner not in (None, -1):
            break
        r, c = random.choice(legal)
        board.make_move(r, c)


def evaluate_vs_baseline(nnue_agent, baseline, num_games=200):
    """
    Evaluate NNUE agent vs a baseline agent.
    Random opening moves are played first to diversify positions.
    
    Args:
        nnue_agent: NNUEAgent instance
        baseline: Baseline agent with select_action(board) -> action
        num_games: Number of games to play
        
    Returns:
        dict with win_rate, model_wins, baseline_wins, draws
    """
    results = {'model_wins': 0, 'baseline_wins': 0, 'draws': 0}
    
    for game_num in range(num_games):
        board = uttt_cpp.Board()
        model_is_p1 = (game_num % 2 == 0)
        
        # Random opening for diversity
        _play_random_opening(board, RANDOM_OPENING_PLIES)
        
        while board.winner in (None, -1):
            legal = board.get_legal_moves()
            if not legal:
                break
            
            is_model_turn = (board.current_player == 1) == model_is_p1
            
            if is_model_turn:
                action = nnue_agent.select_action(board)
            else:
                action = baseline.select_action(board)
            
            board.make_move(action // 9, action % 9)
        
        if board.winner == 3 or board.winner in (None, -1):
            results['draws'] += 1
        elif (board.winner == 1 and model_is_p1) or \
             (board.winner == 2 and not model_is_p1):
            results['model_wins'] += 1
        else:
            results['baseline_wins'] += 1
    
    results['win_rate'] = results['model_wins'] / max(1, num_games)
    results['draw_rate'] = results['draws'] / max(1, num_games)
    results['games'] = num_games
    return results


def run_evaluation_suite(nnue_agent, num_games=200, num_games_minimax=100):
    """
    Run evaluation suite against all baselines.
    
    Args:
        nnue_agent: NNUEAgent instance
        num_games: Games per baseline (random, heuristic)
        num_games_minimax: Games for minimax (slower)
        
    Returns:
        dict of metrics for wandb logging
    """
    metrics = {}
    
    baselines = [
        ('random', RandomAgent(), num_games),
        ('heuristic', HeuristicAgent(), num_games),
        ('minimax2', MinimaxAgent(depth=2), num_games_minimax),
    ]
    
    pbar = tqdm(baselines, desc="Eval", ncols=80, leave=False)
    for name, agent, n in pbar:
        pbar.set_postfix_str(f"vs {name}")
        t0 = time.time()
        r = evaluate_vs_baseline(nnue_agent, agent, num_games=n)
        elapsed = time.time() - t0
        
        metrics[f'eval/vs_{name}_winrate'] = r['win_rate'] * 100
        metrics[f'eval/vs_{name}_drawrate'] = r['draw_rate'] * 100
        metrics[f'eval/vs_{name}_time_s'] = elapsed
    
    return metrics
