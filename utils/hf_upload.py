"""
HuggingFace Upload Utilities
Async background upload to HuggingFace Hub
"""
import os
import sys
import threading
import logging

# Suppress HF hub logging globally
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

HF_REPO_ID = os.environ.get("HF_REPO_ID", "sean2474/ultra-tictactoe-models")
HF_UPLOAD_ENABLED = os.environ.get("HF_UPLOAD", "false").lower() == "true"


_upload_threads = []


def _upload_worker(local_path: str, repo_path: str):
    """Background upload worker"""
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        print(f"  ✓ [HF] Uploaded: {repo_path} ({file_size_mb:.1f}MB)")
    except Exception as e:
        print(f"  ⚠ [HF] Upload failed: {e}")


def upload_to_hf(local_path: str, repo_path: str = None):
    """Upload file to HuggingFace Hub (async background)"""
    if not HF_UPLOAD_ENABLED:
        return
    
    if repo_path is None:
        repo_path = os.path.basename(local_path)
    
    thread = threading.Thread(target=_upload_worker, args=(local_path, repo_path), daemon=True)
    thread.start()
    _upload_threads.append(thread)
    print(f"  ↑ [Async] Upload started: {repo_path}")


def wait_for_uploads():
    """Wait for all pending uploads to complete"""
    for thread in _upload_threads:
        thread.join()
    _upload_threads.clear()
