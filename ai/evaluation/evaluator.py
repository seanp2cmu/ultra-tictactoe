"""
Fast baseline evaluation for training loop.
Runs model vs baseline agents and returns win rates.
"""
import numpy as np
from tqdm import tqdm
from game import Board
from ai.baselines import RandomAgent, HeuristicAgent, MinimaxAgent
from ai.training.parallel_mcts import ParallelMCTS


OPENING_MOVES = 16
OPENING_TEMP = 1.0


def evaluate_vs_baseline(network, baseline, num_games=500, num_simulations=0,
                         parallel_games=None, dtw_calculator=None):
    """
    Evaluate model vs a baseline agent.
    Uses temperature=0.2 for first 8 moves (opening diversity),
    then greedy (temperature=0) for the rest.
    
    Returns:
        dict with win_rate, model_wins, baseline_wins, draws
    """
    if parallel_games is None:
        parallel_games = num_games
    
    mcts = ParallelMCTS(
        network=network,
        num_simulations=num_simulations,
        c_puct=1.0,
        dtw_calculator=dtw_calculator
    )
    
    results = {'model_wins': 0, 'baseline_wins': 0, 'draws': 0}
    games_completed = 0
    
    while games_completed < num_games:
        batch_size = min(parallel_games, num_games - games_completed)
        active_games = []
        for i in range(batch_size):
            game_num = games_completed + i + 1
            model_is_p1 = (game_num % 2 == 1)
            active_games.append({
                'board': Board(),
                'model_is_p1': model_is_p1,
                'move_count': 0,
                'done': False
            })
        
        while any(not g['done'] for g in active_games):
            model_turn_games = []
            baseline_turn_games = []
            
            for g in active_games:
                if g['done']:
                    continue
                is_model_turn = (g['board'].current_player == 1) == g['model_is_p1']
                if is_model_turn:
                    model_turn_games.append(g)
                else:
                    baseline_turn_games.append(g)
            
            if model_turn_games:
                temp = OPENING_TEMP if model_turn_games[0]['move_count'] < OPENING_MOVES else 0.0
                mcts_results = mcts.search_parallel(
                    model_turn_games, temperature=temp, add_noise=False
                )
                for g, (_, action) in zip(model_turn_games, mcts_results):
                    g['board'].make_move(action // 9, action % 9)
                    g['move_count'] += 1
                    if g['board'].winner not in (None, -1):
                        g['done'] = True
            
            for g in baseline_turn_games:
                action = baseline.select_action(g['board'])
                g['board'].make_move(action // 9, action % 9)
                g['move_count'] += 1
                if g['board'].winner not in (None, -1):
                    g['done'] = True
        
        for g in active_games:
            board = g['board']
            if board.winner == 3 or board.winner in (None, -1):
                results['draws'] += 1
            elif (board.winner == 1 and g['model_is_p1']) or \
                 (board.winner == 2 and not g['model_is_p1']):
                results['model_wins'] += 1
            else:
                results['baseline_wins'] += 1
        
        games_completed += batch_size
    
    results['win_rate'] = results['model_wins'] / num_games
    results['games'] = num_games
    return results


def run_evaluation_suite(network, num_games=4000, num_games_minimax=2000, dtw_calculator=None):
    """
    Run fast evaluation suite using raw policy only (0 sims).
    
    Returns:
        dict of metrics ready for wandb logging
    """
    metrics = {}
    
    baselines = [
        ('random', RandomAgent(), num_games),
        ('heuristic', HeuristicAgent(), num_games),
        ('minimax2', MinimaxAgent(depth=2), num_games_minimax),
    ]
    
    pbar = tqdm(baselines, desc="Eval", ncols=80, leave=False, position=1)
    for name, agent, n in pbar:
        pbar.set_postfix_str(f"vs {name} ({n}g)")
        r = evaluate_vs_baseline(network, agent, num_games=n, num_simulations=0)
        metrics[f'eval/vs_{name}_winrate'] = r['win_rate'] * 100
        metrics[f'eval/vs_{name}_drawrate'] = r['draws'] / n * 100
    
    # Elo estimation from minimax2 results (anchor: Minimax-2 = 1620, Minimax-4 = 1800)
    mm2_wr = metrics.get('eval/vs_minimax2_winrate', 0) / 100
    mm2_dr = metrics.get('eval/vs_minimax2_drawrate', 0) / 100
    score = mm2_wr + 0.5 * mm2_dr
    if 0 < score < 1:
        from ai.evaluation.elo import winrate_to_elo_diff
        MINIMAX2_ELO = 1620
        metrics['elo/current'] = MINIMAX2_ELO + winrate_to_elo_diff(score)
        metrics['elo/anchor_minimax4'] = 1800
    
    return metrics
