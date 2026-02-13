"""Model loading utilities"""
import os
from huggingface_hub import hf_hub_download, list_repo_files

from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator
from . import config


def load_models_from_hf(repo_id: str):
    """Load models from Hugging Face Hub"""
    if not repo_id:
        print("No HF_REPO_ID set, skipping HF model loading")
        return []
    
    # Always load to CPU first for ZeroGPU compatibility
    device = 'cpu'
    
    print(f"Loading models from Hugging Face: {repo_id}")
    
    try:
        files = list_repo_files(repo_id)
        model_files = [f for f in files if f.endswith('.pt')]
        
        # DTW 캐시 다운로드
        if 'dtw_cache.pkl' in files:
            cache_path = hf_hub_download(repo_id, 'dtw_cache.pkl')
            if config.dtw_calculator and config.dtw_calculator.tt:
                config.dtw_calculator.tt.load_from_file(cache_path)
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
                config.models[model_name] = network
                print(f"✓ Loaded: {model_name}")
            except Exception as e:
                import traceback
                print(f"✗ Failed to load {model_name}: {e}")
                traceback.print_exc()
        
        return list(config.models.keys())
    except Exception as e:
        print(f"✗ Failed to access HF repo: {e}")
        return []


def load_models_from_local():
    """Load all .pt models from the model directory"""
    if not os.path.exists(config.MODEL_DIR):
        print(f"Model directory {config.MODEL_DIR} not found")
        return []
    
    # Always load to CPU first for ZeroGPU compatibility
    device = 'cpu'
    print("Loading models to CPU (will move to GPU in @spaces.GPU functions)")
    
    # DTW Calculator 생성 (공유)
    config.dtw_calculator = DTWCalculator(
        use_cache=True,
        endgame_threshold=15
    )
    
    # DTW 캐시 로드
    dtw_cache_path = os.path.join(config.MODEL_DIR, 'dtw_cache.pkl')
    if os.path.exists(dtw_cache_path):
        try:
            config.dtw_calculator.tt.load_from_file(dtw_cache_path)
            print(f"✓ DTW cache loaded from {dtw_cache_path}")
        except Exception as e:
            print(f"⚠ Failed to load DTW cache: {e}")
    
    model_files = [f for f in os.listdir(config.MODEL_DIR) if f.endswith('.pt')]
    
    for model_file in model_files:
        model_name = model_file.replace('.pt', '')
        model_path = os.path.join(config.MODEL_DIR, model_file)
        
        try:
            print(f"Loading model: {model_name}")
            network = AlphaZeroNet(device=device)
            network.load(model_path)
            config.models[model_name] = network
            print(f"✓ Successfully loaded: {model_name}")
            
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {str(e)}")
    
    return list(config.models.keys())


def get_models_list():
    """Return list of available models"""
    return {
        'models': list(config.models.keys()),
        'count': len(config.models)
    }


def reload_all_models():
    """Reload all models from HuggingFace and local directory"""
    old_count = len(config.models)
    
    # Clear existing models
    config.models.clear()
    
    # Reload from local
    load_models_from_local()
    
    # Reload from HF
    if config.HF_REPO_ID:
        load_models_from_hf(config.HF_REPO_ID)
    
    new_count = len(config.models)
    return old_count, new_count, list(config.models.keys())
