"""
Ultra Tic-Tac-Toe AI - Gradio Interface
AlphaZero-style engine analysis for Ultimate Tic-Tac-Toe
"""
# Build extensions first (before any imports that need them)
from .build import build_extensions
build_extensions()

import gradio as gr

from . import config
from .models import load_models_from_local, load_models_from_hf
from .pages import (
    create_predict_tab,
    create_arena_tab,
    create_baseline_tab,
    create_models_tab,
    create_api_tab
)


def create_app():
    """Create and return the Gradio app"""
    from ai.endgame import DTWCalculator
    
    # Initialize DTW Calculator
    config.dtw_calculator = DTWCalculator(
        use_cache=True,
        endgame_threshold=15
    )
    
    print("Loading models...")
    # 1. Load from local
    available_models = load_models_from_local()
    
    # 2. Load from HF (if configured)
    if config.HF_REPO_ID:
        load_models_from_hf(config.HF_REPO_ID)
        available_models = list(config.models.keys())
    
    print(f"Loaded {len(config.models)} models: {available_models}")
    
    # Create Gradio interface
    with gr.Blocks(title="Ultra Tic-Tac-Toe AI") as demo:
        gr.Markdown("# 🎮 Ultra Tic-Tac-Toe AI Analysis")
        gr.Markdown("Pure AlphaZero (Lc0-style) for Ultimate Tic-Tac-Toe with DTW endgame solver")
        
        with gr.Tab("Predict"):
            model_dropdown = create_predict_tab(available_models)
        
        with gr.Tab("Arena"):
            p1_model, p2_model = create_arena_tab(available_models)
        
        with gr.Tab("Baseline Test"):
            baseline_model_dropdown = create_baseline_tab(available_models)
        
        with gr.Tab("Models"):
            create_models_tab(model_dropdown, p1_model, p2_model, baseline_model_dropdown)
        
        with gr.Tab("API"):
            create_api_tab()
    
    return demo


def launch(**kwargs):
    """Create and launch the app"""
    demo = create_app()
    demo.launch(**kwargs)
