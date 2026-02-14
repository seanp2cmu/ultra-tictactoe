"""
Fast baseline evaluation for training loop.
Runs model vs baseline agents and returns win rates.
"""
import numpy as np
from game import Board
from ai.baselines import RandomAgent, HeuristicAgent, MinimaxAgent
from ai.training.self_play import ParallelMCTS


def evaluate_vs_baseline(network, baseline, num_games=500, num_simulations=0,
                         parallel_games=None, dtw_calculator=None):
    """
    Evaluate model vs a baseline agent.
    
    Args:
        network: AlphaZeroNet instance (on GPU)
        baseline: Baseline agent with select_action(board) method
        num_games: Number of games to play
        num_simulations: MCTS simulations per move (0 = raw policy)
        parallel_games: Max parallel games (default: num_games)
        dtw_calculator: Optional DTW calculator
    
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
                mcts_results = mcts.search_parallel(
                    model_turn_games, temperature=0.0, add_noise=False
                )
                for g, (_, action) in zip(model_turn_games, mcts_results):
                    g['board'].make_move(action // 9, action % 9)
                    if g['board'].winner not in (None, -1):
                        g['done'] = True
            
            for g in baseline_turn_games:
                action = baseline.select_action(g['board'])
                g['board'].make_move(action // 9, action % 9)
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


def run_evaluation_suite(network, num_games=500, dtw_calculator=None):
    """
    Run fast evaluation suite using raw policy only (0 sims).
    
    Returns:
        dict of metrics ready for wandb logging
    """
    metrics = {}
    
    random_agent = RandomAgent()
    heuristic_agent = HeuristicAgent()
    minimax2_agent = MinimaxAgent(depth=2)
    
    # vs Random - raw policy (0 sims) ~7s
    r = evaluate_vs_baseline(network, random_agent, num_games=num_games, num_simulations=0)
    metrics['eval/vs_random_winrate'] = r['win_rate'] * 100
    metrics['eval/vs_random_drawrate'] = r['draws'] / num_games * 100
    
    # vs Heuristic - raw policy (0 sims) ~7s
    r = evaluate_vs_baseline(network, heuristic_agent, num_games=num_games, num_simulations=0)
    metrics['eval/vs_heuristic_winrate'] = r['win_rate'] * 100
    metrics['eval/vs_heuristic_drawrate'] = r['draws'] / num_games * 100
    
    # vs Minimax-2 - raw policy (0 sims) ~15s
    r = evaluate_vs_baseline(network, minimax2_agent, num_games=num_games, num_simulations=0)
    metrics['eval/vs_minimax2_winrate'] = r['win_rate'] * 100
    metrics['eval/vs_minimax2_drawrate'] = r['draws'] / num_games * 100
    
    return metrics
