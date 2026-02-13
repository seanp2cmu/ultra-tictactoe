"""API tab - API documentation"""
import gradio as gr


def create_api_tab():
    """Create the API documentation tab UI"""
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
