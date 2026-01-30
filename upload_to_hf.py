"""Hugging Face Hub에 모델 업로드"""
import os
import argparse
from huggingface_hub import HfApi, create_repo
import torch


def upload_model(model_path, repo_id, token, commit_message="Upload model"):
    """
    모델을 Hugging Face Hub에 업로드
    
    Args:
        model_path: 로컬 모델 파일 경로 (.pth)
        repo_id: Hugging Face repo ID (예: "username/ultra-tictactoe-model")
        token: Hugging Face API 토큰
        commit_message: 커밋 메시지
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # API 초기화
    api = HfApi()
    
    # Repository 생성 (이미 있으면 무시)
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Repository creation: {e}")
    
    # 모델 정보 로드
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # README.md 생성
    readme_content = f"""---
license: mit
tags:
- reinforcement-learning
- alphazero
- ultimate-tic-tac-toe
- pytorch
---

# Ultimate Tic-Tac-Toe AlphaZero Model

This is an AlphaZero model trained for Ultimate Tic-Tac-Toe.

## Model Info
- Architecture: ResNet with {checkpoint.get('num_res_blocks', 'N/A')} residual blocks
- Channels: {checkpoint.get('num_channels', 'N/A')}

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

# Download model
model_path = hf_hub_download(repo_id="{repo_id}", filename="model.pth")

# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu')
print("Model loaded successfully!")
```

## Training Details
Trained using Monte Carlo Tree Search (MCTS) and self-play.
"""
    
    readme_path = "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # 파일 업로드
    try:
        # 모델 파일 업로드
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="model.pth",
            repo_id=repo_id,
            token=token,
            commit_message=commit_message
        )
        print(f"✓ Model uploaded: model.pth")
        
        # README 업로드
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            commit_message="Add README"
        )
        print(f"✓ README uploaded")
        
        # 로컬 README 삭제
        os.remove(readme_path)
        
        print(f"\n✅ Upload complete!")
        print(f"🔗 View at: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        if os.path.exists(readme_path):
            os.remove(readme_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument("model_path", help="Path to model file (.pth)")
    parser.add_argument("repo_id", help="Hugging Face repo ID (username/repo-name)")
    parser.add_argument("--token", help="Hugging Face API token (or use HF_TOKEN env var)")
    parser.add_argument("--message", default="Upload model", help="Commit message")
    
    args = parser.parse_args()
    
    # 토큰 가져오기 (인자 또는 환경변수)
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Please provide --token or set HF_TOKEN environment variable")
    
    upload_model(args.model_path, args.repo_id, token, args.message)
