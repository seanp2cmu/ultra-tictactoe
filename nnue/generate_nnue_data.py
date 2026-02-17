#!/usr/bin/env python3
"""Generate NNUE training data from trained AlphaZero model."""
from pathlib import Path

from ai.core import AlphaZeroNet
from nnue.data import NNUEDataGenerator
from nnue.config import DataGenConfig


def main():
    config = DataGenConfig()
    
    # Create output directory
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load trained AlphaZero model
    print(f"Loading model from {config.model_path}...")
    network = AlphaZeroNet(device=config.device)
    
    if Path(config.model_path).exists():
        network.load(config.model_path)
        print("Model loaded successfully")
    else:
        print(f"WARNING: Model not found at {config.model_path}, using random weights")
    
    # Create generator
    generator = NNUEDataGenerator(
        network=network,
        config=config,
        num_simulations=config.num_simulations,
        seed=config.seed,
    )
    
    print(f"\n=== NNUE Data Generation ===")
    print(f"Games: {config.num_games}")
    print(f"Simulations: {config.num_simulations}")
    print(f"Skip settings: minply={config.write_minply}, maxply={config.write_maxply}, "
          f"eval_limit={config.eval_limit}, random_skip={config.random_skip_rate}")
    print()
    
    # Generate dataset
    dataset = generator.generate_dataset(
        num_games=config.num_games,
        output_path=str(output_path),
        verbose=True,
    )
    
    generator.print_stats()
    
    print(f"\nDataset shape:")
    print(f"  boards: {dataset['boards'].shape}")
    print(f"  values: {dataset['values'].shape}")
    print(f"\nValue distribution:")
    print(f"  min: {dataset['values'].min():.3f}")
    print(f"  max: {dataset['values'].max():.3f}")
    print(f"  mean: {dataset['values'].mean():.3f}")
    print(f"  std: {dataset['values'].std():.3f}")


if __name__ == '__main__':
    main()
