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
HF_UPLOAD_ENABLED = os.environ.get("HF_UPLOAD", "true").lower() == "true"


_upload_threads = []


def _log_hf(msg: str):
    """Log HF upload status to training.log"""
    import datetime
    log_path = "./model/training.log"
    try:
        with open(log_path, 'a') as f:
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except:
        pass

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
        _log_hf(f"[HF] Uploaded: {repo_path} ({file_size_mb:.1f}MB)")
    except Exception as e:
        _log_hf(f"[HF] Upload failed: {e}")


def upload_to_hf(local_path: str, repo_path: str = None):
    """Upload file to HuggingFace Hub (async background)"""
    if not HF_UPLOAD_ENABLED:
        return
    
    if repo_path is None:
        repo_path = os.path.basename(local_path)
    
    thread = threading.Thread(target=_upload_worker, args=(local_path, repo_path), daemon=True)
    thread.start()
    _upload_threads.append(thread)
    _log_hf(f"[HF] Upload started: {repo_path}")


def _delete_worker(repo_path: str):
    """Background delete worker"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.delete_file(repo_path, repo_id=HF_REPO_ID, repo_type="model")
        _log_hf(f"[HF] Deleted: {repo_path}")
    except Exception as e:
        _log_hf(f"[HF] Delete failed: {repo_path} - {e}")


def delete_from_hf(repo_path: str):
    """Delete file from HuggingFace Hub (async background)"""
    if not HF_UPLOAD_ENABLED:
        return
    thread = threading.Thread(target=_delete_worker, args=(repo_path,), daemon=True)
    thread.start()
    _upload_threads.append(thread)


def wait_for_uploads():
    """Wait for all pending uploads to complete"""
    for thread in _upload_threads:
        thread.join()
    _upload_threads.clear()
