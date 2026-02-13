"""Arena tab - Flexible agent vs agent matches"""
import time
import gradio as gr

from ai.mcts import AlphaZeroAgent
from ai.baselines import RandomAgent, HeuristicAgent, MinimaxAgent
from ai.training.self_play import ParallelMCTS
from game import Board
from app import config

# Arena state
arena_stop_flag = False
arena_results = {'current': None}


def create_agent(agent_type: str, model_name: str, num_sims: int, temperature: float):
    """Create agent based on type. Returns (agent, display_name)."""
    if agent_type == "Random":
        return RandomAgent(), "Random"
    elif agent_type == "Heuristic":
        return HeuristicAgent(), "Heuristic"
    elif agent_type == "Minimax-2":
        return MinimaxAgent(depth=2), "Minimax-2"
    elif agent_type == "Minimax-3":
        return MinimaxAgent(depth=3), "Minimax-3"
    elif agent_type == "Model":
        if model_name not in config.models:
            return None, f"Model '{model_name}' not found"
        network = config.ensure_model_on_gpu(model_name)
        agent = AlphaZeroAgent(
            network=network,
            num_simulations=num_sims,
            c_puct=1.0,
            temperature=temperature,
            batch_size=8,
            dtw_calculator=config.dtw_calculator
        )
        return agent, f"{model_name}({num_sims}sims)"
    return None, "Unknown agent type"


def stop_arena():
    """Stop the running arena match."""
    global arena_stop_flag
    arena_stop_flag = True
    return "⏹️ Stop requested. Match will stop after current batch."


def reset_arena_stop():
    """Reset stop flag before starting new match."""
    global arena_stop_flag
    arena_stop_flag = False


def get_arena_duration(p1_type, p1_model, p1_sims, p2_type, p2_model, p2_sims, num_games, *args):
    """Calculate dynamic GPU duration based on match parameters."""
    num_games = int(num_games)
    p1_sims = int(p1_sims)
    p2_sims = int(p2_sims)
    
    # Estimate time per game based on agent types
    if p1_type == "Model" and p2_type == "Model":
        # Both models - parallel processing helps
        base_time = 3
        sim_factor = max(p1_sims, p2_sims) / 200
        parallel_factor = 0.25  # 8 games parallel
    elif p1_type == "Model" or p2_type == "Model":
        # One model - some parallelization
        base_time = 2
        sim_factor = max(p1_sims, p2_sims) / 200
        parallel_factor = 0.4
    else:
        # No models - CPU only, fast
        return 60
    
    estimated_time = num_games * base_time * sim_factor * parallel_factor
    # Add buffer, min 60s, max 1800s
    return min(max(int(estimated_time * 1.3) + 60, 60), 1800)


@config.spaces.GPU(duration=get_arena_duration)
def arena_match(p1_type, p1_model, p1_sims, p2_type, p2_model, p2_sims, num_games, temperature, alternate_colors, seed):
    """Flexible arena with parallel game playing for Model vs Model."""
    global arena_stop_flag, arena_results
    import random as rand_module
    
    reset_arena_stop()
    
    num_games = int(num_games)
    p1_sims = int(p1_sims)
    p2_sims = int(p2_sims)
    seed = int(seed) if seed else None
    parallel_games = num_games  # H100 handles full parallelization
    
    if seed is not None:
        rand_module.seed(seed)
    
    try:
        # Check if we can use parallel MCTS (both are Models)
        use_parallel = (p1_type == "Model" and p2_type == "Model")
        
        if use_parallel:
            # Setup parallel MCTS for both players
            if p1_model not in config.models:
                yield f"Error: Model '{p1_model}' not found"
                return
            if p2_model not in config.models:
                yield f"Error: Model '{p2_model}' not found"
                return
            
            network1 = config.ensure_model_on_gpu(p1_model)
            network2 = config.ensure_model_on_gpu(p2_model)
            
            parallel_mcts1 = ParallelMCTS(
                network=network1, num_simulations=p1_sims,
                c_puct=1.0, dtw_calculator=config.dtw_calculator
            )
            parallel_mcts2 = ParallelMCTS(
                network=network2, num_simulations=p2_sims,
                c_puct=1.0, dtw_calculator=config.dtw_calculator
            )
            name1 = f"{p1_model}({p1_sims})"
            name2 = f"{p2_model}({p2_sims})"
        else:
            # Create regular agents
            agent1, name1 = create_agent(p1_type, p1_model, p1_sims, temperature)
            if agent1 is None:
                yield f"Error: {name1}"
                return
            
            agent2, name2 = create_agent(p2_type, p2_model, p2_sims, temperature)
            if agent2 is None:
                yield f"Error: {name2}"
                return
        
        results = {
            'p1': name1, 'p2': name2,
            'p1_wins': 0, 'p2_wins': 0, 'draws': 0,
            'p1_as_X_wins': 0, 'p1_as_O_wins': 0,
            'total_time': 0
        }
        
        games_completed = 0
        batch_start_time = time.time()
        
        while games_completed < num_games:
            # Check stop flag
            if arena_stop_flag:
                results['total_time'] = time.time() - batch_start_time
                if games_completed > 0:
                    p1_score = (results['p1_wins'] + results['draws'] * 0.5) / games_completed
                    yield f"""⏹️ Stopped after {games_completed} games

{name1}: {results['p1_wins']} wins | {name2}: {results['p2_wins']} wins | Draws: {results['draws']}
P1 Score: {p1_score:.3f}"""
                return
            
            batch_size = min(parallel_games, num_games - games_completed)
            
            if use_parallel:
                # Parallel Model vs Model
                active_games = []
                for i in range(batch_size):
                    game_num = games_completed + i + 1
                    p1_is_X = (game_num % 2 == 1) if alternate_colors else True
                    active_games.append({
                        'board': Board(),
                        'game_num': game_num,
                        'p1_is_X': p1_is_X,
                        'moves': 0,
                        'done': False
                    })
                
                # Play until all games in batch are done
                while any(not g['done'] for g in active_games):
                    # Separate by whose turn
                    p1_turn_games = []
                    p2_turn_games = []
                    
                    for g in active_games:
                        if g['done']:
                            continue
                        is_p1_turn = (g['board'].current_player == 1) == g['p1_is_X']
                        if is_p1_turn:
                            p1_turn_games.append(g)
                        else:
                            p2_turn_games.append(g)
                    
                    # Batch MCTS for P1
                    if p1_turn_games:
                        mcts_results = parallel_mcts1.search_parallel(
                            p1_turn_games, temperature=temperature, add_noise=False
                        )
                        for g, (_, action) in zip(p1_turn_games, mcts_results):
                            g['board'].make_move(action // 9, action % 9)
                            g['moves'] += 1
                            if g['board'].winner not in (None, -1):
                                g['done'] = True
                    
                    # Batch MCTS for P2
                    if p2_turn_games:
                        mcts_results = parallel_mcts2.search_parallel(
                            p2_turn_games, temperature=temperature, add_noise=False
                        )
                        for g, (_, action) in zip(p2_turn_games, mcts_results):
                            g['board'].make_move(action // 9, action % 9)
                            g['moves'] += 1
                            if g['board'].winner not in (None, -1):
                                g['done'] = True
            else:
                # Sequential play for non-parallel cases
                active_games = []
                for i in range(batch_size):
                    game_num = games_completed + i + 1
                    p1_is_X = (game_num % 2 == 1) if alternate_colors else True
                    board = Board()
                    move_count = 0
                    
                    while board.winner in (None, -1) and move_count < 81:
                        is_p1_turn = (board.current_player == 1) == p1_is_X
                        current_agent = agent1 if is_p1_turn else agent2
                        
                        if isinstance(current_agent, AlphaZeroAgent):
                            action = current_agent.select_action(board, temperature=temperature)
                        else:
                            action = current_agent.select_action(board)
                        
                        board.make_move(action // 9, action % 9)
                        move_count += 1
                    
                    active_games.append({
                        'board': board,
                        'game_num': game_num,
                        'p1_is_X': p1_is_X,
                        'moves': move_count,
                        'done': True
                    })
            
            # Record results
            for g in active_games:
                board = g['board']
                p1_is_X = g['p1_is_X']
                
                if board.winner == 3 or board.winner in (None, -1):
                    results['draws'] += 1
                elif (board.winner == 1 and p1_is_X) or (board.winner == 2 and not p1_is_X):
                    results['p1_wins'] += 1
                    if p1_is_X:
                        results['p1_as_X_wins'] += 1
                    else:
                        results['p1_as_O_wins'] += 1
                else:
                    results['p2_wins'] += 1
            
            games_completed += batch_size
            results['total_time'] = time.time() - batch_start_time
            arena_results['current'] = results.copy()
            
            # Progress update
            p1_rate = results['p1_wins'] / games_completed
            p2_rate = results['p2_wins'] / games_completed
            draw_rate = results['draws'] / games_completed
            p1_score = (results['p1_wins'] + results['draws'] * 0.5) / games_completed
            
            progress = f"""
╔══════════════════════════════════════════════════════════════╗
║  🎮 ARENA: {name1} vs {name2} {'[PARALLEL]' if use_parallel else ''}
╠══════════════════════════════════════════════════════════════╣
║  Progress: {games_completed}/{num_games} games (batch: {batch_size})
╠══════════════════════════════════════════════════════════════╣
║  {name1}: {results['p1_wins']} wins ({p1_rate*100:.1f}%)
║    - As X: {results['p1_as_X_wins']}  As O: {results['p1_as_O_wins']}
║  {name2}: {results['p2_wins']} wins ({p2_rate*100:.1f}%)
║  Draws: {results['draws']} ({draw_rate*100:.1f}%)
╠══════════════════════════════════════════════════════════════╣
║  📊 P1 Score: {p1_score:.3f}  (1.0=all wins, 0.5=even, 0.0=all losses)
╚══════════════════════════════════════════════════════════════╝
"""
            yield progress
        
        # Final summary
        total = num_games
        p1_score = (results['p1_wins'] + results['draws'] * 0.5) / total
        avg_time = results['total_time'] / total
        
        final = f"""
╔══════════════════════════════════════════════════════════════╗
║                    🏆 FINAL RESULTS                          ║
╠══════════════════════════════════════════════════════════════╣
║  {name1} vs {name2}
║  Games: {num_games}  |  Seed: {seed if seed else 'None'}  |  {'Parallel' if use_parallel else 'Sequential'}
╠══════════════════════════════════════════════════════════════╣
║  {name1}: {results['p1_wins']} wins ({results['p1_wins']/total*100:.1f}%)
║    - As X (first): {results['p1_as_X_wins']}
║    - As O (second): {results['p1_as_O_wins']}
║  {name2}: {results['p2_wins']} wins ({results['p2_wins']/total*100:.1f}%)
║  Draws: {results['draws']} ({results['draws']/total*100:.1f}%)
╠══════════════════════════════════════════════════════════════╣
║  📊 P1 Score: {p1_score:.3f}
║  ⏱️  Total: {results['total_time']:.1f}s  Avg/Game: {avg_time:.2f}s
╚══════════════════════════════════════════════════════════════╝
"""
        yield final
        
    except Exception as e:
        import traceback
        # Return partial results on timeout
        if arena_results['current'] and arena_results['current'].get('p1_wins', 0) + arena_results['current'].get('p2_wins', 0) + arena_results['current'].get('draws', 0) > 0:
            r = arena_results['current']
            completed = r['p1_wins'] + r['p2_wins'] + r['draws']
            p1_score = (r['p1_wins'] + r['draws'] * 0.5) / completed if completed > 0 else 0
            yield f"""⚠️ GPU TIMEOUT - Partial results ({completed} games)

{r['p1']}: {r['p1_wins']} wins | {r['p2']}: {r['p2_wins']} wins | Draws: {r['draws']}
P1 Score: {p1_score:.3f}

💡 Try fewer games or lower simulations."""
        else:
            traceback.print_exc()
            yield f"Error: {e}\n{traceback.format_exc()}"


def create_arena_tab(available_models):
    """Create the Arena tab UI"""
    gr.Markdown("### 🎮 Flexible Arena - Any Agent vs Any Agent")
    gr.Markdown("Test any combination: Random vs Random, Model vs Heuristic, Model(200) vs Model(0), etc.")
    
    agent_types = ["Random", "Heuristic", "Minimax-2", "Minimax-3", "Model"]
    
    gr.Markdown("#### Player 1 (P1)")
    with gr.Row():
        p1_type = gr.Dropdown(choices=agent_types, label="P1 Type", value="Model")
        p1_model = gr.Dropdown(choices=available_models, label="P1 Model (if Model type)", 
                               value=available_models[0] if available_models else None)
        p1_sims = gr.Slider(minimum=0, maximum=800, value=200, step=50, label="P1 Simulations")
    
    gr.Markdown("#### Player 2 (P2)")
    with gr.Row():
        p2_type = gr.Dropdown(choices=agent_types, label="P2 Type", value="Random")
        p2_model = gr.Dropdown(choices=available_models, label="P2 Model (if Model type)",
                               value=available_models[0] if available_models else None)
        p2_sims = gr.Slider(minimum=0, maximum=800, value=200, step=50, label="P2 Simulations")
    
    gr.Markdown("#### Match Settings")
    with gr.Row():
        arena_games = gr.Slider(minimum=2, maximum=500, value=50, step=10, label="Number of Games")
        arena_temp = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature")
    
    with gr.Row():
        alternate_checkbox = gr.Checkbox(label="Alternate Colors (recommended)", value=True)
        seed_input = gr.Number(label="Seed (optional, for reproducibility)", value=None, precision=0)
    
    with gr.Row():
        arena_btn = gr.Button("⚔️ Start Arena Match", variant="primary", size="lg")
        arena_stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
    
    arena_output = gr.Textbox(label="Results", lines=20, interactive=False)
    
    arena_btn.click(
        fn=arena_match,
        inputs=[p1_type, p1_model, p1_sims, p2_type, p2_model, p2_sims, 
                arena_games, arena_temp, alternate_checkbox, seed_input],
        outputs=arena_output,
        api_name="arena"
    )
    
    arena_stop_btn.click(
        fn=stop_arena,
        inputs=[],
        outputs=arena_output
    )
    
    gr.Markdown("""
    ### 🧪 Test Cases
    | Test | P1 | P2 | Expected Score |
    |------|----|----|----------------|
    | Random vs Random | Random | Random | ~0.5 |
    | Heuristic vs Random | Heuristic | Random | >0.5 |
    | Model vs Model | Model(200) | Model(200) | ~0.5 |
    | MCTS benefit | Model(200) | Model(0) | >0.5 |
    | Reproducibility | Same seed twice | | Same results |
    """)
    
    return p1_model, p2_model
