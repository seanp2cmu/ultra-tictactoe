"""Model loading utilities"""
import os
import json
from huggingface_hub import hf_hub_download, list_repo_files

from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator
from . import config

RUNS_FILE = "runs.json"


def _load_runs_json(base_dir: str) -> dict:
    path = os.path.join(base_dir, RUNS_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_models_from_hf(repo_id: str):
    """Load models from Hugging Face Hub (run-based folder structure)"""
    if not repo_id:
        print("No HF_REPO_ID set, skipping HF model loading")
        return []
    
    device = 'cpu'
    print(f"Loading models from Hugging Face: {repo_id}")
    
    try:
        files = list_repo_files(repo_id)
        
        # Load runs.json for name mapping
        runs = {}
        if RUNS_FILE in files:
            runs_path = hf_hub_download(repo_id, RUNS_FILE)
            with open(runs_path) as f:
                runs = json.load(f)
        
        # Find model files (run_id/file.pt or legacy file.pt)
        model_files = [f for f in files if f.endswith('.pt')]
        print(f"Found {len(model_files)} model files")
        
        # Load DTW caches from runs
        for f in files:
            if f.endswith('dtw_cache.pkl'):
                try:
                    cache_path = hf_hub_download(repo_id, f)
                    if config.dtw_calculator and config.dtw_calculator.tt:
                        config.dtw_calculator.tt.load_from_file(cache_path)
                        print(f"\u2713 DTW cache loaded: {f}")
                except:
                    pass
        
        for model_file in model_files:
            try:
                # Determine display name
                if '/' in model_file:
                    run_id = model_file.split('/')[0]
                    fname = model_file.split('/')[-1].replace('.pt', '')
                    run_name = runs.get(run_id, {}).get('name', run_id[:8])
                    display_name = f"{run_name}/{fname}"
                else:
                    display_name = model_file.replace('.pt', '')
                
                print(f"Downloading: {model_file} -> {display_name}")
                model_path = hf_hub_download(repo_id, model_file)
                
                network = AlphaZeroNet(device=device)
                network.load(model_path)
                config.models[display_name] = network
                print(f"\u2713 Loaded: {display_name}")
            except Exception as e:
                print(f"\u2717 Failed to load {model_file}: {e}")
        
        return list(config.models.keys())
    except Exception as e:
        print(f"\u2717 Failed to access HF repo: {e}")
        return []


def load_models_from_local():
    """Load all .pt models from the model directory (run-based folder structure)"""
    if not os.path.exists(config.MODEL_DIR):
        print(f"Model directory {config.MODEL_DIR} not found")
        return []
    
    device = 'cpu'
    print("Loading models to CPU (will move to GPU in @spaces.GPU functions)")
    
    # DTW Calculator
    config.dtw_calculator = DTWCalculator(
        use_cache=True,
        endgame_threshold=15
    )
    
    runs = _load_runs_json(config.MODEL_DIR)
    
    # Scan run directories
    for run_id, info in runs.items():
        run_dir = os.path.join(config.MODEL_DIR, run_id)
        if not os.path.isdir(run_dir):
            continue
        
        run_name = info.get('name', run_id[:8])
        
        # Load DTW cache from this run
        dtw_cache_path = os.path.join(run_dir, 'dtw_cache.pkl')
        if os.path.exists(dtw_cache_path):
            try:
                config.dtw_calculator.tt.load_from_file(dtw_cache_path)
                print(f"\u2713 DTW cache loaded: {run_name}")
            except:
                pass
        
        # Load model files
        for model_file in sorted(os.listdir(run_dir)):
            if not model_file.endswith('.pt'):
                continue
            
            fname = model_file.replace('.pt', '')
            display_name = f"{run_name}/{fname}"
            model_path = os.path.join(run_dir, model_file)
            
            try:
                network = AlphaZeroNet(device=device)
                network.load(model_path)
                config.models[display_name] = network
                print(f"\u2713 Loaded: {display_name}")
            except Exception as e:
                print(f"\u2717 Failed to load {display_name}: {e}")
    
    # Also load legacy root-level models (backward compat)
    for model_file in os.listdir(config.MODEL_DIR):
        if model_file.endswith('.pt'):
            model_name = model_file.replace('.pt', '')
            model_path = os.path.join(config.MODEL_DIR, model_file)
            try:
                network = AlphaZeroNet(device=device)
                network.load(model_path)
                config.models[model_name] = network
                print(f"\u2713 Loaded (legacy): {model_name}")
            except:
                pass
    
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
