"""Predict tab - Model prediction and analysis"""
import json
import gradio as gr

from ai.mcts import AlphaZeroAgent
from app import config
from app.utils import board_from_dict, get_principal_variation


@config.spaces.GPU
def predict(model_name, board_json, num_simulations=200):
    """Predict best moves for given board state using specified model"""
    num_simulations = int(num_simulations)
    
    try:
        board_data = json.loads(board_json)
        
        if model_name not in config.models:
            return json.dumps({
                'error': f'Model {model_name} not found',
                'available_models': list(config.models.keys())
            }, indent=2)
        
        # Move model to GPU inside @spaces.GPU context
        network = config.ensure_model_on_gpu(model_name)
        
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
            dtw_calculator=config.dtw_calculator
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
        eval_percentage = evaluation * 100
        
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


def create_predict_tab(available_models):
    """Create the Predict tab UI"""
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
    
    return model_dropdown
