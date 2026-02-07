"""
Ultra Tic-Tac-Toe AI - Gradio Interface
AlphaZero-style engine analysis for Ultimate Tic-Tac-Toe
"""
import os
import json
import time
import gradio as gr
import torch
import spaces
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

def load_models_from_hf(repo_id: str):
    """Load models from Hugging Face Hub"""
    global models, dtw_calculator
    
    if not repo_id:
        print("No HF_REPO_ID set, skipping HF model loading")
        return []
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
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
    
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple MPS GPU")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # DTW Calculator 생성 (공유)
    dtw_calculator = DTWCalculator(
        use_cache=True,
        endgame_threshold=15,
        midgame_threshold=45,
        shallow_depth=8
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
    board.boards = board_dict.get('boards', [[0]*9 for _ in range(9)])
    board.completed_boards = board_dict.get('completed_boards', [[0]*3 for _ in range(3)])
    board.current_player = board_dict.get('current_player', 1)
    board.winner = board_dict.get('winner', None)
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

@spaces.GPU(duration=600)
def compare_models(model1_name, model2_name, num_games, num_simulations, temperature):
    """
    Compare two models by playing multiple self-play games
    """
    # Ensure integer types from sliders
    num_games = int(num_games)
    num_simulations = int(num_simulations)
    
    try:
        if model1_name not in models or model2_name not in models:
            yield json.dumps({
                'error': 'One or both models not found',
                'available_models': list(models.keys())
            }, indent=2)
            return
        
        agent1 = AlphaZeroAgent(
            network=models[model1_name],
            num_simulations=num_simulations,
            c_puct=1.0,
            temperature=temperature,
            batch_size=8,
            dtw_calculator=dtw_calculator
        )
        
        agent2 = AlphaZeroAgent(
            network=models[model2_name],
            num_simulations=num_simulations,
            c_puct=1.0,
            temperature=temperature,
            batch_size=8,
            dtw_calculator=dtw_calculator
        )
        
        results = {
            'model1': model1_name,
            'model2': model2_name,
            'num_simulations': num_simulations,
            'temperature': temperature,
            'games': [],
            'summary': {
                'model1_wins': 0,
                'model2_wins': 0,
                'draws': 0,
                'total_time': 0
            }
        }
        
        for game_num in range(1, num_games + 1):
            game_start_time = time.time()
            
            board = Board()
            game_record = {
                'game_number': game_num,
                'moves': [],
                'winner': None,
                'elapsed_time': 0
            }
            
            move_count = 0
            while board.winner is None and move_count < 81:
                current_agent = agent1 if board.current_player == 1 else agent2
                
                action = current_agent.select_action(board, temperature=temperature)
                move_r = action // 9
                move_c = action % 9
                
                game_record['moves'].append({
                    'move_number': move_count + 1,
                    'player': board.current_player,
                    'action': int(action),
                    'position': [int(move_r), int(move_c)]
                })
                
                board.make_move(move_r, move_c)
                move_count += 1
                
                if board.winner is not None:
                    break
            
            game_elapsed = time.time() - game_start_time
            game_record['winner'] = int(board.winner) if board.winner is not None else None
            game_record['elapsed_time'] = round(game_elapsed, 2)
            game_record['total_moves'] = move_count
            
            if board.winner == 1:
                results['summary']['model1_wins'] += 1
            elif board.winner == 2:
                results['summary']['model2_wins'] += 1
            else:
                results['summary']['draws'] += 1
            
            results['summary']['total_time'] = round(
                results['summary']['total_time'] + game_elapsed, 2
            )
            results['games'].append(game_record)
            
            progress = {
                'status': 'in_progress',
                'completed_games': game_num,
                'total_games': num_games,
                'latest_game': game_record,
                'current_summary': results['summary'].copy()
            }
            
            yield json.dumps(progress, indent=2)
        
        results['status'] = 'completed'
        results['summary']['avg_time_per_game'] = round(
            results['summary']['total_time'] / num_games, 2
        )
        
        yield json.dumps(results, indent=2)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield json.dumps({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }, indent=2)

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
        
        board = board_from_dict(board_data)
        
        if board.winner is not None:
            return json.dumps({
                'error': 'Game is already over',
                'winner': board.winner
            }, indent=2)
        
        agent = AlphaZeroAgent(
            network=models[model_name],
            num_simulations=num_simulations,
            c_puct=1.0,
            temperature=0.0,
            batch_size=8,
            dtw_calculator=dtw_calculator
        )
        
        root = agent.search(board)
        
        _, value = models[model_name].predict(board)
        
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
        
        evaluation = float(value)
        eval_percentage = (evaluation + 1) / 2 * 100
        
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

@spaces.GPU(duration=300)
def test_vs_baseline(model_name, baseline_name, num_games, num_simulations):
    """Test AlphaZero model against baseline opponent."""
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
        
        az_agent = AlphaZeroAgent(
            network=models[model_name],
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
            while board.winner is None and move_count < 81:
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
                
                if board.winner is not None:
                    break
            
            game_elapsed = time.time() - game_start_time
            game_record['winner'] = int(board.winner) if board.winner is not None else None
            game_record['elapsed_time'] = round(game_elapsed, 2)
            game_record['total_moves'] = move_count
            
            if board.winner == 3 or board.winner is None:
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
        
        yield json.dumps(results, indent=2)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield json.dumps({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }, indent=2)

# DTW Calculator 초기화 (HF 로딩 전에)
dtw_calculator = DTWCalculator(
    use_cache=True,
    endgame_threshold=15,
    midgame_threshold=45,
    shallow_depth=8
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
    gr.Markdown("AlphaZero-style engine analysis for Ultimate Tic-Tac-Toe with DTW endgame solver")
    
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
        - **Evaluation**: Position evaluation (-1 to 1)
        - **Best Moves**: Top 5 moves with visit counts and probabilities
        - **Principal Variation**: Best move sequence (like chess engine analysis)
        - **DTW**: Endgame positions (≤15 cells) use exact alpha-beta search
        """)
    
    with gr.Tab("Compare Models"):
        gr.Markdown("### 🤖 Model Comparison - Self-Play Arena")
        gr.Markdown("Compare two AI models by having them play against each other. Results stream in real-time!")
        
        with gr.Row():
            model1_dropdown = gr.Dropdown(
                choices=available_models,
                label="Model 1 (Player 1 - X)",
                value=available_models[0] if available_models else None
            )
            model2_dropdown = gr.Dropdown(
                choices=available_models,
                label="Model 2 (Player 2 - O)",
                value=available_models[1] if len(available_models) > 1 else (available_models[0] if available_models else None)
            )
        
        with gr.Row():
            num_games_slider = gr.Slider(
                minimum=1,
                maximum=50,
                value=10,
                step=1,
                label="Number of Games"
            )
            compare_sims_slider = gr.Slider(
                minimum=50,
                maximum=1000,
                value=200,
                step=50,
                label="MCTS Simulations per Move"
            )
        
        temperature_slider = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=0.0,
            step=0.1,
            label="Temperature (0 = deterministic, higher = more random)"
        )
        
        compare_btn = gr.Button("🎮 Start Comparison", variant="primary", size="lg")
        
        compare_output = gr.Textbox(
            label="Results (Updates in Real-Time)",
            lines=25,
            interactive=False
        )
        
        compare_btn.click(
            fn=compare_models,
            inputs=[model1_dropdown, model2_dropdown, num_games_slider, compare_sims_slider, temperature_slider],
            outputs=compare_output,
            api_name="compare"
        )
        
        gr.Markdown("""
        ### 📊 Output Format
        
        **During Games (Real-time updates)**:
        - Current progress (completed games / total)
        - Latest game result with full move history
        - Running summary (wins/draws/time)
        
        **Final Result**:
        - Complete game records for all matches
        - Summary statistics (wins, draws, avg time)
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
        
        baseline_btn = gr.Button("🧪 Run Baseline Test", variant="primary", size="lg")
        
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
        
        models_output = gr.Markdown(show_models())
        refresh_btn = gr.Button("Refresh Models")
        refresh_btn.click(fn=show_models, outputs=models_output)
    
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
        
        #### POST /compare
        ```python
        result = client.predict(
            model1_name="model_10",
            model2_name="model_20",
            num_games=10,
            num_simulations=200,
            temperature=0.0,
            api_name="/compare"
        )
        ```
        """)

if __name__ == "__main__":
    demo.launch(share=True)
