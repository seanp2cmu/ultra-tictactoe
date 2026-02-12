"""
Ultra Tic-Tac-Toe AI - Gradio Interface
AlphaZero-style engine analysis for Ultimate Tic-Tac-Toe
"""
import os
import subprocess
import sys

# Build Cython/C++ extensions BEFORE any imports that need them
def build_extensions():
    """Build Cython and C++ extensions on HF Spaces startup"""
    if not os.environ.get('SPACE_ID'):
        return  # Skip on local
    
    try:
        # Install build dependencies first
        print("Installing build dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "Cython", "pybind11", "numpy"], 
                      capture_output=True, check=True)
        print("✓ Build dependencies installed")
        
        print("Building Cython extensions...")
        result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], 
                      capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Cython extensions built")
        else:
            print(f"⚠ Cython build error: {result.stderr}")
        
        print("Building C++ extensions...")
        result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"],
                      cwd="cpp", capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ C++ extensions built")
        else:
            print(f"⚠ C++ build error: {result.stderr}")
    except Exception as e:
        print(f"⚠ Build error: {e}")

build_extensions()

import json
import time
import gradio as gr
import torch

# Mock spaces.GPU decorator for local runs
# Check if we're on HF Spaces by looking for SPACE_ID env var
if os.environ.get('SPACE_ID'):
    import spaces
else:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, duration=None):
            if fn is not None:
                return fn
            def decorator(func):
                return func
            return decorator
    spaces = _MockSpaces()

from huggingface_hub import hf_hub_download, list_repo_files
from ai.core import AlphaZeroNet
from ai.mcts import AlphaZeroAgent
from ai.endgame import DTWCalculator
from ai.baselines import RandomAgent, HeuristicAgent, MinimaxAgent
from game import Board

MODEL_DIR = "model"
HF_REPO_ID = "sean2474/ultra-tictactoe-models" 
models = {}
dtw_calculator = None
baseline_test_stop_flag = False

def get_best_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def ensure_model_on_gpu(model_name: str):
    """Move model to best available GPU (CUDA or MPS) or keep on CPU."""
    if model_name not in models:
        return None
    network = models[model_name]
    device = get_best_device()
    if device.type != 'cpu':
        network.device = device
        network.model = network.model.to(device)
    return network

def load_models_from_hf(repo_id: str):
    """Load models from Hugging Face Hub"""
    global models, dtw_calculator
    
    if not repo_id:
        print("No HF_REPO_ID set, skipping HF model loading")
        return []
    
    # Always load to CPU first for ZeroGPU compatibility
    # Models will be moved to GPU inside @spaces.GPU decorated functions
    device = 'cpu'
    
    print(f"Loading models from Hugging Face: {repo_id}")
    
    try:
        files = list_repo_files(repo_id)
        model_files = [f for f in files if f.endswith('.pt')]
        
        # DTW 캐시 다운로드
        if 'dtw_cache.pkl' in files:
            cache_path = hf_hub_download(repo_id, 'dtw_cache.pkl')
            if dtw_calculator and dtw_calculator.tt:
                dtw_calculator.tt.load_from_file(cache_path)
                print(f"✓ DTW cache loaded from HF")
        
        print(f"Found {len(model_files)} model files: {model_files}")
        
        for model_file in model_files:
            model_name = model_file.replace('.pt', '')
            try:
                print(f"Downloading: {model_file}")
                model_path = hf_hub_download(repo_id, model_file)
                print(f"  Downloaded to: {model_path}")
                
                network = AlphaZeroNet(device=device)
                network.load(model_path)
                models[model_name] = network
                print(f"✓ Loaded: {model_name}")
            except Exception as e:
                import traceback
                print(f"✗ Failed to load {model_name}: {e}")
                traceback.print_exc()
        
        return list(models.keys())
    except Exception as e:
        print(f"✗ Failed to access HF repo: {e}")
        return []

def load_models_from_local():
    """Load all .pt models from the model directory"""
    global models, dtw_calculator
    
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory {MODEL_DIR} not found")
        return []
    
    # Always load to CPU first for ZeroGPU compatibility
    device = 'cpu'
    print("Loading models to CPU (will move to GPU in @spaces.GPU functions)")
    
    # DTW Calculator 생성 (공유)
    dtw_calculator = DTWCalculator(
        use_cache=True,
        endgame_threshold=15
    )
    
    # DTW 캐시 로드
    dtw_cache_path = os.path.join(MODEL_DIR, 'dtw_cache.pkl')
    if os.path.exists(dtw_cache_path):
        try:
            dtw_calculator.tt.load_from_file(dtw_cache_path)
            print(f"✓ DTW cache loaded from {dtw_cache_path}")
        except Exception as e:
            print(f"⚠ Failed to load DTW cache: {e}")
    
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
    
    for model_file in model_files:
        model_name = model_file.replace('.pt', '')
        model_path = os.path.join(MODEL_DIR, model_file)
        
        try:
            print(f"Loading model: {model_name}")
            network = AlphaZeroNet(device=device)
            network.load(model_path)
            models[model_name] = network
            print(f"✓ Successfully loaded: {model_name}")
            
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {str(e)}")
    
    return list(models.keys())

def board_from_dict(board_dict):
    """Create a Board object from dictionary representation"""
    board = Board()
    boards_data = board_dict.get('boards', [[0]*9 for _ in range(9)])
    for r in range(9):
        for c in range(9):
            if boards_data[r][c] != 0:
                board.set_cell(r, c, boards_data[r][c])
    # Set completed_boards (BoardCy compatible)
    cb_data = board_dict.get('completed_boards', [[0]*3 for _ in range(3)])
    if hasattr(board, 'set_completed_boards_2d'):
        board.set_completed_boards_2d(cb_data)
    else:
        board.completed_boards = cb_data
    # Sync completed_mask
    completed = board.get_completed_boards_2d() if hasattr(board, 'get_completed_boards_2d') else board.completed_boards
    for sub_idx in range(9):
        sub_r, sub_c = sub_idx // 3, sub_idx % 3
        if completed[sub_r][sub_c] != 0:
            board.completed_mask |= (1 << sub_idx)
    board.current_player = board_dict.get('current_player', 1)
    # BoardCy uses -1 for no winner, Python Board uses None
    raw_winner = board_dict.get('winner', None)
    board.winner = -1 if raw_winner is None else raw_winner
    board.last_move = tuple(board_dict['last_move']) if board_dict.get('last_move') else None
    return board

def get_principal_variation(root, depth=10):
    """Extract principal variation (best move sequence) from MCTS tree"""
    pv = []
    node = root
    
    for _ in range(depth):
        if not node.children or node.is_terminal():
            break
        
        best_action = max(node.children.items(), key=lambda x: x[1].visits)[0]
        best_child = node.children[best_action]
        
        move_r = best_action // 9
        move_c = best_action % 9
        
        pv.append({
            'action': int(best_action),
            'move': [int(move_r), int(move_c)],
            'visits': int(best_child.visits),
            'value': float(best_child.value()),
            'eval': float(best_child.value() * 100)
        })
        
        node = best_child
    
    return pv

def get_models_list():
    """Return list of available models"""
    return {
        'models': list(models.keys()),
        'count': len(models)
    }

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
        if model_name not in models:
            return None, f"Model '{model_name}' not found"
        network = ensure_model_on_gpu(model_name)
        agent = AlphaZeroAgent(
            network=network,
            num_simulations=num_sims,
            c_puct=1.0,
            temperature=temperature,
            batch_size=8,
            dtw_calculator=dtw_calculator
        )
        return agent, f"{model_name}({num_sims}sims)"
    return None, "Unknown agent type"

@spaces.GPU(duration=1800)
def arena_match(p1_type, p1_model, p1_sims, p2_type, p2_model, p2_sims, num_games, temperature, alternate_colors, seed):
    """
    Flexible arena: any agent vs any agent with configurable settings.
    """
    import random as rand_module
    
    num_games = int(num_games)
    p1_sims = int(p1_sims)
    p2_sims = int(p2_sims)
    seed = int(seed) if seed else None
    
    if seed is not None:
        rand_module.seed(seed)
    
    try:
        # Create agents
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
        
        for game_num in range(1, num_games + 1):
            game_start = time.time()
            
            # Alternate colors if enabled
            if alternate_colors:
                p1_is_X = (game_num % 2 == 1)
            else:
                p1_is_X = True
            
            board = Board()
            move_count = 0
            
            while board.winner in (None, -1) and move_count < 81:
                is_p1_turn = (board.current_player == 1) == p1_is_X
                current_agent = agent1 if is_p1_turn else agent2
                
                # Handle different agent types
                if hasattr(current_agent, 'select_action'):
                    if isinstance(current_agent, AlphaZeroAgent):
                        action = current_agent.select_action(board, temperature=temperature)
                    else:
                        action = current_agent.select_action(board)
                else:
                    action = current_agent.select_action(board)
                
                move_r, move_c = action // 9, action % 9
                board.make_move(move_r, move_c)
                move_count += 1
            
            game_time = time.time() - game_start
            results['total_time'] += game_time
            
            # Determine winner
            if board.winner == 3 or board.winner in (None, -1):
                results['draws'] += 1
                result_str = "Draw"
            elif (board.winner == 1 and p1_is_X) or (board.winner == 2 and not p1_is_X):
                results['p1_wins'] += 1
                if p1_is_X:
                    results['p1_as_X_wins'] += 1
                else:
                    results['p1_as_O_wins'] += 1
                result_str = f"{name1} wins"
            else:
                results['p2_wins'] += 1
                result_str = f"{name2} wins"
            
            # Calculate win rates
            total = game_num
            p1_rate = results['p1_wins'] / total
            p2_rate = results['p2_wins'] / total
            draw_rate = results['draws'] / total
            
            # Score: win=1, draw=0.5, loss=0
            p1_score = (results['p1_wins'] + results['draws'] * 0.5) / total
            
            progress = f"""
╔══════════════════════════════════════════════════════════════╗
║  🎮 ARENA: {name1} vs {name2}
╠══════════════════════════════════════════════════════════════╣
║  Game {game_num}/{num_games}: {result_str} ({move_count} moves, {game_time:.1f}s)
║  P1 played as: {'X (first)' if p1_is_X else 'O (second)'}
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
║  Games: {num_games}  |  Seed: {seed if seed else 'None'}
╠══════════════════════════════════════════════════════════════╣
║  {name1}: {results['p1_wins']} wins ({results['p1_wins']/total*100:.1f}%)
║    - As X (first): {results['p1_as_X_wins']}
║    - As O (second): {results['p1_as_O_wins']}
║  {name2}: {results['p2_wins']} wins ({results['p2_wins']/total*100:.1f}%)
║  Draws: {results['draws']} ({results['draws']/total*100:.1f}%)
╠══════════════════════════════════════════════════════════════╣
║  📊 P1 Score: {p1_score:.3f}
║  ⏱️  Avg time/game: {avg_time:.2f}s
╚══════════════════════════════════════════════════════════════╝
"""
        yield final
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"Error: {e}\n{traceback.format_exc()}"

@spaces.GPU
def predict(model_name, board_json, num_simulations=200):
    """
    Predict best moves for given board state using specified model
    """
    num_simulations = int(num_simulations)
    
    try:
        board_data = json.loads(board_json)
        
        if model_name not in models:
            return json.dumps({
                'error': f'Model {model_name} not found',
                'available_models': list(models.keys())
            }, indent=2)
        
        # Move model to GPU inside @spaces.GPU context
        network = ensure_model_on_gpu(model_name)
        
        board = board_from_dict(board_data)
        
        if board.winner not in (None, -1):
            return json.dumps({
                'error': 'Game is already over',
                'winner': int(board.winner)
            }, indent=2)
        
        agent = AlphaZeroAgent(
            network=network,
            num_simulations=num_simulations,
            c_puct=1.0,
            temperature=0.0,
            batch_size=8,
            dtw_calculator=dtw_calculator
        )
        
        root = agent.search(board)
        
        _, value = network.predict(board)
        
        action_visits = [(action, child.visits, child.value()) 
                        for action, child in root.children.items()]
        action_visits.sort(key=lambda x: x[1], reverse=True)
        
        best_moves = []
        total_visits = sum(visits for _, visits, _ in action_visits)
        
        for action, visits, child_value in action_visits[:5]:
            move_r = action // 9
            move_c = action % 9
            probability = visits / total_visits if total_visits > 0 else 0
            
            best_moves.append({
                'action': int(action),
                'move': [int(move_r), int(move_c)],
                'visits': int(visits),
                'value': float(child_value),
                'probability': float(probability)
            })
        
        pv = get_principal_variation(root, depth=10)
        
        # Value is now 0~1 (loss=0, draw=0.5, win=1)
        evaluation = float(value)
        eval_percentage = evaluation * 100  # Already 0~1
        
        result = {
            'model': model_name,
            'evaluation': evaluation,
            'eval_percentage': eval_percentage,
            'best_moves': best_moves,
            'principal_variation': pv,
            'total_simulations': num_simulations,
            'current_player': board.current_player
        }
        
        return json.dumps(result, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            'error': 'Invalid JSON format',
            'details': str(e)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'error': str(e),
            'type': type(e).__name__
        }, indent=2)

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
    global baseline_test_stop_flag
    baseline_test_stop_flag = True
    return "⏹️ Stop requested. Test will stop after current game."

def reset_baseline_stop():
    """Reset stop flag before starting new test."""
    global baseline_test_stop_flag
    baseline_test_stop_flag = False

def get_baseline_test_duration(model_name, baseline_name, num_games, num_simulations):
    """Calculate dynamic GPU duration based on test parameters."""
    # Estimate ~3 seconds per game for simpler baselines, ~10 for harder ones
    base_time_per_game = 5 if baseline_name in ['Random', 'Heuristic'] else 10
    # More simulations = more time
    sim_factor = int(num_simulations) / 200  # baseline is 200 sims
    estimated_time = int(num_games) * base_time_per_game * sim_factor
    # Add 30% buffer, cap at 1800 seconds (30 min)
    return min(int(estimated_time * 1.3) + 60, 1800)

# Store partial results for timeout recovery
baseline_test_results = {'current': None}

@spaces.GPU(duration=get_baseline_test_duration)
def test_vs_baseline(model_name, baseline_name, num_games, num_simulations):
    """Test AlphaZero model against baseline opponent."""
    global baseline_test_stop_flag, baseline_test_results
    reset_baseline_stop()
    
    num_games = int(num_games)
    num_simulations = int(num_simulations)
    
    try:
        if model_name not in models:
            yield json.dumps({
                'error': f'Model {model_name} not found',
                'available_models': list(models.keys())
            }, indent=2)
            return
        
        if baseline_name not in baseline_agents:
            yield json.dumps({
                'error': f'Baseline {baseline_name} not found',
                'available_baselines': list(baseline_agents.keys())
            }, indent=2)
            return
        
        # Move model to GPU inside @spaces.GPU context
        network = ensure_model_on_gpu(model_name)
        
        az_agent = AlphaZeroAgent(
            network=network,
            num_simulations=num_simulations,
            c_puct=1.0,
            temperature=0.0,
            batch_size=8,
            dtw_calculator=dtw_calculator
        )
        baseline = baseline_agents[baseline_name]
        
        results = {
            'model': model_name,
            'baseline': baseline_name,
            'num_simulations': num_simulations,
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
        
        for game_num in range(1, num_games + 1):
            # Check stop flag
            if baseline_test_stop_flag:
                results['status'] = 'stopped'
                results['summary']['completed_games'] = game_num - 1
                if game_num > 1:
                    results['summary']['win_rate'] = f"{results['summary']['model_wins'] / (game_num - 1) * 100:.1f}%"
                yield json.dumps({
                    'status': 'stopped',
                    'message': f'⏹️ Test stopped after {game_num - 1} games',
                    'summary': results['summary']
                }, indent=2)
                return
            
            game_start_time = time.time()
            model_is_p1 = (game_num % 2 == 1)
            
            board = Board()
            game_record = {
                'game_number': game_num,
                'model_plays_as': 1 if model_is_p1 else 2,
                'moves': [],
                'winner': None,
                'elapsed_time': 0
            }
            
            move_count = 0
            while board.winner in (None, -1) and move_count < 81:
                is_model_turn = (board.current_player == 1) == model_is_p1
                
                if is_model_turn:
                    action = az_agent.select_action(board, temperature=0.0)
                else:
                    action = baseline.select_action(board)
                
                move_r = action // 9
                move_c = action % 9
                
                game_record['moves'].append({
                    'move_number': move_count + 1,
                    'player': 'model' if is_model_turn else 'baseline',
                    'action': int(action),
                    'position': [int(move_r), int(move_c)]
                })
                
                board.make_move(move_r, move_c)
                move_count += 1
                
                if board.winner not in (None, -1):
                    break
            
            game_elapsed = time.time() - game_start_time
            game_record['winner'] = int(board.winner) if board.winner not in (None, -1) else None
            game_record['elapsed_time'] = round(game_elapsed, 2)
            game_record['total_moves'] = move_count
            
            if board.winner == 3 or board.winner in (None, -1):
                results['summary']['draws'] += 1
                game_record['result'] = 'draw'
            elif (board.winner == 1 and model_is_p1) or (board.winner == 2 and not model_is_p1):
                results['summary']['model_wins'] += 1
                if model_is_p1:
                    results['summary']['model_as_p1_wins'] += 1
                else:
                    results['summary']['model_as_p2_wins'] += 1
                game_record['result'] = 'model_win'
            else:
                results['summary']['baseline_wins'] += 1
                game_record['result'] = 'baseline_win'
            
            results['summary']['total_time'] = round(
                results['summary']['total_time'] + game_elapsed, 2
            )
            results['games'].append(game_record)
            
            # Store partial results for timeout recovery
            results['summary']['completed_games'] = game_num
            baseline_test_results['current'] = results.copy()
            
            win_rate = results['summary']['model_wins'] / game_num * 100
            
            progress = {
                'status': 'in_progress',
                'completed_games': game_num,
                'total_games': num_games,
                'win_rate': f"{win_rate:.1f}%",
                'latest_game': {
                    'game_number': game_num,
                    'model_played_as': 'P1 (X)' if model_is_p1 else 'P2 (O)',
                    'result': game_record['result'],
                    'moves': move_count,
                    'time': f"{game_elapsed:.1f}s"
                },
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
        if baseline_test_results['current'] and baseline_test_results['current'].get('summary', {}).get('completed_games', 0) > 0:
            partial = baseline_test_results['current']
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

# DTW Calculator 초기화 (HF 로딩 전에)
dtw_calculator = DTWCalculator(
    use_cache=True,
    endgame_threshold=15
)

print("Loading models...")
# 1. 로컬에서 로드 시도
available_models = load_models_from_local()

# 2. HF에서 추가 로드 (HF_REPO_ID 설정된 경우)
if HF_REPO_ID:
    hf_models = load_models_from_hf(HF_REPO_ID)
    available_models = list(models.keys())

print(f"Loaded {len(models)} models: {available_models}")

with gr.Blocks(title="Ultra Tic-Tac-Toe AI") as demo:
    gr.Markdown("# 🎮 Ultra Tic-Tac-Toe AI Analysis")
    gr.Markdown("Pure AlphaZero (Lc0-style) for Ultimate Tic-Tac-Toe with DTW endgame solver")
    
    with gr.Tab("Predict"):
        gr.Markdown("### Model Prediction & Analysis")
        
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=available_models,
                label="Select Model",
                value=available_models[0] if available_models else None
            )
            num_sims_slider = gr.Slider(
                minimum=50,
                maximum=1000,
                value=200,
                step=50,
                label="MCTS Simulations"
            )
        
        board_input = gr.Textbox(
            label="Board State (JSON)",
            placeholder='{"boards": [[0,0,0,...], ...], "completed_boards": [[0,0,0], ...], "current_player": 1, "winner": null, "last_move": null}',
            lines=10
        )
        
        predict_btn = gr.Button("Analyze Position", variant="primary")
        
        output = gr.Textbox(
            label="Analysis Result",
            lines=20
        )
        
        predict_btn.click(
            fn=predict,
            inputs=[model_dropdown, board_input, num_sims_slider],
            outputs=output,
            api_name="predict"
        )
        
        gr.Markdown("""
        ### 📝 Input Format
        ```json
        {
          "boards": [[0,0,0,0,0,0,0,0,0], ...],  // 9x9 array (0: empty, 1: player1, 2: player2)
          "completed_boards": [[0,0,0], [0,0,0], [0,0,0]],  // 3x3 array
          "current_player": 1,  // 1 or 2
          "winner": null,  // null, 1, 2, or 3 (draw)
          "last_move": [4, 5]  // [row, col] or null
        }
        ```
        
        ### 📊 Output Includes
        - **Evaluation**: Position evaluation (0 to 1, where 0=loss, 0.5=draw, 1=win)
        - **Best Moves**: Top 5 moves with visit counts and probabilities
        - **Principal Variation**: Best move sequence (like chess engine analysis)
        - **DTW**: Endgame positions (≤15 cells) use exact alpha-beta search
        """)
    
    with gr.Tab("Arena"):
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
            arena_games = gr.Slider(minimum=2, maximum=200, value=20, step=2, label="Number of Games")
            arena_temp = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature")
        
        with gr.Row():
            alternate_checkbox = gr.Checkbox(label="Alternate Colors (recommended)", value=True)
            seed_input = gr.Number(label="Seed (optional, for reproducibility)", value=None, precision=0)
        
        arena_btn = gr.Button("⚔️ Start Arena Match", variant="primary", size="lg")
        
        arena_output = gr.Textbox(label="Results", lines=20, interactive=False)
        
        arena_btn.click(
            fn=arena_match,
            inputs=[p1_type, p1_model, p1_sims, p2_type, p2_model, p2_sims, 
                    arena_games, arena_temp, alternate_checkbox, seed_input],
            outputs=arena_output,
            api_name="arena"
        )
        
        gr.Markdown("""
        ### � Test Cases
        | Test | P1 | P2 | Expected Score |
        |------|----|----|----------------|
        | Random vs Random | Random | Random | ~0.5 |
        | Heuristic vs Random | Heuristic | Random | >0.5 |
        | Model vs Model | Model(200) | Model(200) | ~0.5 |
        | MCTS benefit | Model(200) | Model(0) | >0.5 |
        | Reproducibility | Same seed twice | | Same results |
        """)
    
    with gr.Tab("Baseline Test"):
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
                minimum=2,
                maximum=100,
                value=20,
                step=2,
                label="Number of Games (alternating first player)"
            )
            baseline_sims_slider = gr.Slider(
                minimum=50,
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
    
    with gr.Tab("Models"):
        gr.Markdown("### Available Models")
        
        def show_models():
            info = get_models_list()
            return f"**Total Models:** {info['count']}\n\n**Models:**\n" + "\n".join(f"- {m}" for m in info['models'])
        
        def reload_models():
            """Reload all models from HuggingFace and local directory"""
            global models, available_models
            
            old_count = len(models)
            
            # Clear existing models
            models.clear()
            
            # Reload from local
            load_models_from_local()
            
            # Reload from HF
            if HF_REPO_ID:
                load_models_from_hf(HF_REPO_ID)
            
            new_count = len(models)
            available_models = list(models.keys())
            
            info = get_models_list()
            status = f"🔄 **Reloaded!** {old_count} → {new_count} models\n\n"
            status += f"**Total Models:** {info['count']}\n\n**Models:**\n"
            status += "\n".join(f"- {m}" for m in info['models'])
            
            return status, gr.update(choices=available_models), gr.update(choices=available_models), gr.update(choices=available_models), gr.update(choices=available_models)
        
        models_output = gr.Markdown(show_models())
        refresh_btn = gr.Button("🔄 Reload Models from HF", variant="primary")
        refresh_btn.click(
            fn=reload_models, 
            outputs=[models_output, model_dropdown, p1_model, p2_model, baseline_model_dropdown]
        )
    
    with gr.Tab("API"):
        gr.Markdown("""
        ### 🔌 API Endpoints
        
        This Gradio interface also provides API access:
        
        #### POST /predict
        ```python
        from gradio_client import Client
        
        client = Client("YOUR_SPACE_URL")
        result = client.predict(
            model_name="best",
            board_json='{"boards": [...], ...}',
            num_simulations=200
        )
        ```
        
        #### POST /arena
        ```python
        result = client.predict(
            p1_type="Model",      # Random, Heuristic, Minimax-2, Minimax-3, Model
            p1_model="best",
            p1_sims=200,
            p2_type="Random",
            p2_model="best",
            p2_sims=0,
            num_games=20,
            temperature=0.0,
            alternate_colors=True,
            seed=42,
            api_name="/arena"
        )
        ```
        """)

if __name__ == "__main__":
    demo.launch(share=True)
