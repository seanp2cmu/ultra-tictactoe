"""Pages module for Gradio tabs"""
from .predict import create_predict_tab
from .arena import create_arena_tab
from .baseline import create_baseline_tab
from .models_tab import create_models_tab
from .api import create_api_tab

__all__ = [
    'create_predict_tab',
    'create_arena_tab', 
    'create_baseline_tab',
    'create_models_tab',
    'create_api_tab'
]
