#!/usr/bin/env python3
"""
Compare NNUE training with different lambda_search values.
Runs identical pipelines with λ=1.0, 0.75, 0.5 sequentially.
"""
from nnue.train_nnue import run
from nnue.config import PipelineConfig, TrainingConfig


# Smaller scale for fair comparison (same compute per run)
COMPARE_CFG = dict(
    phase1_games=5000,
    selfplay_loops=10,
    selfplay_games=20000,
    selfplay_threads=64,
    selfplay_depth=8,
    eval_games=200,
    eval_games_minimax=100,
    eval_depth=5,
    max_positions=1000000,
    early_stop_patience=3,
)

LAMBDAS = [1.0, 0.75, 0.5]


def main():
    train_cfg = TrainingConfig()

    for lam in LAMBDAS:
        print(f"\n{'#'*60}")
        print(f"# Lambda = {lam}")
        print(f"{'#'*60}\n")

        cfg = PipelineConfig(
            lambda_search=lam,
            **COMPARE_CFG,
        )

        run(cfg=cfg, train_cfg=train_cfg)


if __name__ == "__main__":
    main()
