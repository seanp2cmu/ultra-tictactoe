"""Baseline Test tab - Test model against baseline opponents"""
import json
import time
import gradio as gr

from ai.baselines import RandomAgent, HeuristicAgent, MinimaxAgent
from ai.training.self_play import ParallelMCTS
from game import Board
from app import config


# Baseline agents
baseline_agents = {
    'Random': RandomAgent(),
    'Heuristic': HeuristicAgent(),
    'Minimax-2': MinimaxAgent(depth=2),
    'Minimax-3': MinimaxAgent(depth=3),
    'Minimax-4': MinimaxAgent(depth=4),
}


def stop_baseline_test():
    """Stop the running baseline test."""
    config.baseline_test_stop_flag = True
    return "⏹️ Stop requested. Test will stop after current game."


def reset_baseline_stop():
    """Reset stop flag before starting new test."""
    config.baseline_test_stop_flag = False


def get_baseline_test_duration(model_name, baseline_name, num_games, num_simulations):
    """Calculate dynamic GPU duration based on test parameters."""
    # Estimate ~2 seconds per game for simpler baselines, ~5 for harder ones (parallel reduces time)
    base_time_per_game = 2 if baseline_name in ['Random', 'Heuristic'] else 5
    # More simulations = more time
    sim_factor = int(num_simulations) / 200  # baseline is 200 sims
    # Parallel games (8) reduces effective time by ~4x
    parallel_factor = 0.25
    estimated_time = int(num_games) * base_time_per_game * sim_factor * parallel_factor
    # Add 30% buffer, minimum 60s, cap at 1800 seconds (30 min)
    return min(max(int(estimated_time * 1.3) + 60, 60), 1800)


@config.spaces.GPU(duration=get_baseline_test_duration)
def test_vs_baseline(model_name, baseline_name, num_games, num_simulations):
    """Test AlphaZero model against baseline opponent with parallel game playing."""
    reset_baseline_stop()
    
    num_games = int(num_games)
    num_simulations = int(num_simulations)
    parallel_games = num_games  # H100 handles full parallelization
    
    try:
        if model_name not in config.models:
            yield json.dumps({
                'error': f'Model {model_name} not found',
                'available_models': list(config.models.keys())
            }, indent=2)
            return
        
        if baseline_name not in baseline_agents:
            yield json.dumps({
                'error': f'Baseline {baseline_name} not found',
                'available_baselines': list(baseline_agents.keys())
            }, indent=2)
            return
        
        # Move model to GPU inside @spaces.GPU context
        network = config.ensure_model_on_gpu(model_name)
        
        parallel_mcts = ParallelMCTS(
            network=network,
            num_simulations=num_simulations,
            c_puct=1.0,
            dtw_calculator=config.dtw_calculator
        )
        baseline = baseline_agents[baseline_name]
        
        results = {
            'model': model_name,
            'baseline': baseline_name,
            'num_simulations': num_simulations,
            'parallel_games': parallel_games,
            'games': [],
            'summary': {
                'model_wins': 0,
                'baseline_wins': 0,
                'draws': 0,
                'model_as_p1_wins': 0,
                'model_as_p2_wins': 0,
                'total_time': 0
            }
        }
        
        games_completed = 0
        batch_start_time = time.time()
        
        while games_completed < num_games:
            # Check stop flag
            if config.baseline_test_stop_flag:
                results['status'] = 'stopped'
                results['summary']['completed_games'] = games_completed
                if games_completed > 0:
                    results['summary']['win_rate'] = f"{results['summary']['model_wins'] / games_completed * 100:.1f}%"
                yield json.dumps({
                    'status': 'stopped',
                    'message': f'⏹️ Test stopped after {games_completed} games',
                    'summary': results['summary']
                }, indent=2)
                return
            
            # Start batch of parallel games
            batch_size = min(parallel_games, num_games - games_completed)
            active_games = []
            for i in range(batch_size):
                game_num = games_completed + i + 1
                model_is_p1 = (game_num % 2 == 1)
                active_games.append({
                    'board': Board(),
                    'game_num': game_num,
                    'model_is_p1': model_is_p1,
                    'moves': [],
                    'done': False
                })
            
            # Play until all games in batch are done
            while any(not g['done'] for g in active_games):
                # Separate games by whose turn it is
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
                
                # Batch MCTS for model turn games
                if model_turn_games:
                    mcts_results = parallel_mcts.search_parallel(
                        model_turn_games, temperature=0.0, add_noise=False
                    )
                    for g, (_, action) in zip(model_turn_games, mcts_results):
                        g['moves'].append(('model', int(action)))
                        g['board'].make_move(action // 9, action % 9)
                        if g['board'].winner not in (None, -1):
                            g['done'] = True
                
                # Sequential baseline moves (CPU-bound, no batching benefit)
                for g in baseline_turn_games:
                    action = baseline.select_action(g['board'])
                    g['moves'].append(('baseline', int(action)))
                    g['board'].make_move(action // 9, action % 9)
                    if g['board'].winner not in (None, -1):
                        g['done'] = True
            
            # Record results for completed batch
            for g in active_games:
                game_num = g['game_num']
                model_is_p1 = g['model_is_p1']
                board = g['board']
                
                if board.winner == 3 or board.winner in (None, -1):
                    results['summary']['draws'] += 1
                    result = 'draw'
                elif (board.winner == 1 and model_is_p1) or (board.winner == 2 and not model_is_p1):
                    results['summary']['model_wins'] += 1
                    if model_is_p1:
                        results['summary']['model_as_p1_wins'] += 1
                    else:
                        results['summary']['model_as_p2_wins'] += 1
                    result = 'model_win'
                else:
                    results['summary']['baseline_wins'] += 1
                    result = 'baseline_win'
                
                results['games'].append({
                    'game_number': game_num,
                    'model_plays_as': 1 if model_is_p1 else 2,
                    'result': result,
                    'total_moves': len(g['moves'])
                })
            
            games_completed += batch_size
            results['summary']['total_time'] = round(time.time() - batch_start_time, 2)
            results['summary']['completed_games'] = games_completed
            config.baseline_test_results['current'] = results.copy()
            
            win_rate = results['summary']['model_wins'] / games_completed * 100
            
            progress = {
                'status': 'in_progress',
                'completed_games': games_completed,
                'total_games': num_games,
                'parallel_games': parallel_games,
                'win_rate': f"{win_rate:.1f}%",
                'batch_completed': batch_size,
                'current_summary': results['summary'].copy()
            }
            
            yield json.dumps(progress, indent=2)
        
        results['status'] = 'completed'
        results['summary']['win_rate'] = f"{results['summary']['model_wins'] / num_games * 100:.1f}%"
        results['summary']['avg_time_per_game'] = round(
            results['summary']['total_time'] / num_games, 2
        )
        
        # Create formatted summary table
        s = results['summary']
        table = f"""
╔══════════════════════════════════════════════════════════════╗
║                    🎯 BASELINE TEST RESULTS                   ║
╠══════════════════════════════════════════════════════════════╣
║  Model:     {model_name:<20}  vs  {baseline_name:<18} ║
║  Games:     {num_games:<20}  Simulations: {num_simulations:<10} ║
╠══════════════════════════════════════════════════════════════╣
║                         RESULTS                               ║
╠═══════════════════════╦══════════════════════════════════════╣
║  Model Wins           ║  {s['model_wins']:>6}  ({s['win_rate']})                      ║
║    - As Player 1 (X)  ║  {s['model_as_p1_wins']:>6}                                ║
║    - As Player 2 (O)  ║  {s['model_as_p2_wins']:>6}                                ║
║  Baseline Wins        ║  {s['baseline_wins']:>6}                                ║
║  Draws                ║  {s['draws']:>6}                                ║
╠═══════════════════════╩══════════════════════════════════════╣
║  Total Time: {s['total_time']:.1f}s    Avg/Game: {s['avg_time_per_game']:.2f}s                   ║
╚══════════════════════════════════════════════════════════════╝
"""
        yield table
        
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        
        # Check if we have partial results to return
        if config.baseline_test_results['current'] and config.baseline_test_results['current'].get('summary', {}).get('completed_games', 0) > 0:
            partial = config.baseline_test_results['current']
            completed = partial['summary']['completed_games']
            if completed > 0:
                partial['summary']['win_rate'] = f"{partial['summary']['model_wins'] / completed * 100:.1f}%"
                partial['summary']['avg_time_per_game'] = round(partial['summary']['total_time'] / completed, 2)
            
            s = partial['summary']
            table = f"""
⚠️ GPU TIMEOUT - Returning partial results

╔══════════════════════════════════════════════════════════════╗
║              🎯 PARTIAL BASELINE TEST RESULTS                 ║
╠══════════════════════════════════════════════════════════════╣
║  Completed: {completed}/{num_games} games (GPU timeout)                      ║
╠══════════════════════════════════════════════════════════════╣
║  Model Wins           ║  {s['model_wins']:>6}  ({s.get('win_rate', 'N/A')})                      ║
║    - As Player 1 (X)  ║  {s['model_as_p1_wins']:>6}                                ║
║    - As Player 2 (O)  ║  {s['model_as_p2_wins']:>6}                                ║
║  Baseline Wins        ║  {s['baseline_wins']:>6}                                ║
║  Draws                ║  {s['draws']:>6}                                ║
╠═══════════════════════╩══════════════════════════════════════╣
║  Total Time: {s['total_time']:.1f}s    Avg/Game: {s.get('avg_time_per_game', 0):.2f}s                   ║
╚══════════════════════════════════════════════════════════════╝

💡 Tip: Try fewer games or lower simulations to avoid timeout.
"""
            yield table
        else:
            traceback.print_exc()
            yield json.dumps({
                'error': str(e),
                'type': error_type,
                'message': 'Test failed. Try fewer games or lower simulations.'
            }, indent=2)


def create_baseline_tab(available_models):
    """Create the Baseline Test tab UI"""
    gr.Markdown("### 🎯 Baseline Test - Sanity Check")
    gr.Markdown("Test your AlphaZero model against baseline opponents to verify training progress.")
    
    with gr.Row():
        baseline_model_dropdown = gr.Dropdown(
            choices=available_models,
            label="AlphaZero Model",
            value=available_models[0] if available_models else None
        )
        baseline_dropdown = gr.Dropdown(
            choices=list(baseline_agents.keys()),
            label="Baseline Opponent",
            value="Random"
        )
    
    with gr.Row():
        baseline_games_slider = gr.Slider(
            minimum=10,
            maximum=500,
            value=50,
            step=10,
            label="Number of Games (alternating first player)"
        )
        baseline_sims_slider = gr.Slider(
            minimum=0,
            maximum=800,
            value=200,
            step=50,
            label="MCTS Simulations"
        )
    
    with gr.Row():
        baseline_btn = gr.Button("🧪 Run Baseline Test", variant="primary", size="lg")
        baseline_stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
    
    baseline_output = gr.Textbox(
        label="Results (Updates in Real-Time)",
        lines=20,
        interactive=False
    )
    
    baseline_btn.click(
        fn=test_vs_baseline,
        inputs=[baseline_model_dropdown, baseline_dropdown, baseline_games_slider, baseline_sims_slider],
        outputs=baseline_output,
        api_name="baseline_test"
    )
    
    baseline_stop_btn.click(
        fn=stop_baseline_test,
        inputs=[],
        outputs=baseline_output
    )
    
    gr.Markdown("""
    ### 📊 Baseline Opponents
    
    | Opponent | Description | Expected Win Rate |
    |----------|-------------|-------------------|
    | **Random** | Random legal moves | >95% (if training works) |
    | **Heuristic** | Greedy one-step lookahead | >80% |
    | **Minimax-2** | Alpha-beta depth 2 | >70% |
    | **Minimax-3** | Alpha-beta depth 3 | >60% |
    | **Minimax-4** | Alpha-beta depth 4 | >50% |
    
    ⚠️ If your model can't beat Random, something is wrong!
    """)
    
    return baseline_model_dropdown
