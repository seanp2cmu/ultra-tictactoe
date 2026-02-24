"""Baseline Test tab - Test model against baseline opponents"""
import json
import time
import gradio as gr

from ai.baselines import RandomAgent, HeuristicAgent, MinimaxAgent
from ai.evaluation.evaluator import evaluate_vs_baseline
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
    base_time_per_game = 2 if baseline_name in ['Random', 'Heuristic'] else 5
    sim_factor = int(num_simulations) / 200
    parallel_factor = 0.25
    estimated_time = int(num_games) * base_time_per_game * sim_factor * parallel_factor
    return min(max(int(estimated_time * 1.3) + 60, 60), 1800)


@config.spaces.GPU(duration=get_baseline_test_duration)
def test_vs_baseline(model_name, baseline_name, num_games, num_simulations):
    """Test AlphaZero model against baseline opponent using unified evaluator."""
    reset_baseline_stop()
    
    num_games = int(num_games)
    num_simulations = int(num_simulations)
    
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
        
        network = config.ensure_model_on_gpu(model_name)
        baseline = baseline_agents[baseline_name]
        
        batch_start_time = time.time()
        
        yield json.dumps({
            'status': 'in_progress',
            'completed_games': 0,
            'total_games': num_games,
            'message': f'Running {num_games} games with {num_simulations} sims...'
        }, indent=2)
        
        r = evaluate_vs_baseline(
            network, baseline, num_games=num_games,
            num_simulations=num_simulations,
            dtw_calculator=config.dtw_calculator,
        )
        
        total_time = round(time.time() - batch_start_time, 2)
        avg_time = round(total_time / num_games, 2) if num_games > 0 else 0
        win_rate_str = f"{r['win_rate'] * 100:.1f}%"
        
        config.baseline_test_results['current'] = {
            'summary': {
                'model_wins': r['model_wins'],
                'baseline_wins': r['baseline_wins'],
                'draws': r['draws'],
                'win_rate': win_rate_str,
                'total_time': total_time,
                'completed_games': num_games,
            }
        }
        
        s = config.baseline_test_results['current']['summary']
        table = f"""
╔══════════════════════════════════════════════════════════════╗
║                    🎯 BASELINE TEST RESULTS                   ║
╠══════════════════════════════════════════════════════════════╣
║  Model:     {model_name:<20}  vs  {baseline_name:<18} ║
║  Games:     {num_games:<20}  Simulations: {num_simulations:<10} ║
╠══════════════════════════════════════════════════════════════╣
║                         RESULTS                               ║
╠═══════════════════════╦══════════════════════════════════════╣
║  Model Wins           ║  {s['model_wins']:>6}  ({win_rate_str})                      ║
║  Baseline Wins        ║  {s['baseline_wins']:>6}                                ║
║  Draws                ║  {s['draws']:>6}                                ║
╠═══════════════════════╩══════════════════════════════════════╣
║  Total Time: {total_time:.1f}s    Avg/Game: {avg_time:.2f}s                   ║
╚══════════════════════════════════════════════════════════════╝
"""
        yield table
        
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        
        if config.baseline_test_results.get('current') and config.baseline_test_results['current'].get('summary', {}).get('completed_games', 0) > 0:
            partial = config.baseline_test_results['current']
            s = partial['summary']
            table = f"""
⚠️ GPU TIMEOUT - Returning partial results

╔══════════════════════════════════════════════════════════════╗
║              🎯 PARTIAL BASELINE TEST RESULTS                 ║
╠══════════════════════════════════════════════════════════════╣
║  Model Wins           ║  {s['model_wins']:>6}  ({s.get('win_rate', 'N/A')})                      ║
║  Baseline Wins        ║  {s['baseline_wins']:>6}                                ║
║  Draws                ║  {s['draws']:>6}                                ║
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
