#!/usr/bin/env python3
"""
Compare NNUE training with different AZ lambda values.
Runs identical pipelines with λ=1.0, 0.75, 0.5 sequentially.
"""
from nnue.train import run, ConcurrentConfig
from nnue.config import TrainingConfig


# Smaller scale for fair comparison (same compute per run)
COMPARE_CFG = dict(
    az_total_games=5000,
    az_batch_games=2500,
    rescore_loops=3,
    eval_games=200,
    eval_games_minimax=100,
    eval_depth=6,
    max_positions=1000000,
)

LAMBDAS = [1.0, 0.75, 0.5]


def main():
    train_cfg = TrainingConfig()

    for lam in LAMBDAS:
        print(f"\n{'#'*60}")
        print(f"# Lambda = {lam}")
        print(f"{'#'*60}\n")

        cfg = ConcurrentConfig(
            az_lambda=lam,
            **COMPARE_CFG,
        )

        run(cfg=cfg, train_cfg=train_cfg)


if __name__ == "__main__":
    main()
