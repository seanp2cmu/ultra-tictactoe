#!/usr/bin/env python3
"""Generate NNUE training data from trained AlphaZero model."""
import argparse
from pathlib import Path

from ai.core import AlphaZeroNet
from nnue.data import NNUEDataGenerator
from nnue.config import NNUEConfig


def main():
    parser = argparse.ArgumentParser(description='Generate NNUE training data')
    parser.add_argument('--model', type=str, default='model/best_model.pt',
                        help='Path to trained AlphaZero model')
    parser.add_argument('--games', type=int, default=10000,
                        help='Number of self-play games')
    parser.add_argument('--simulations', type=int, default=800,
                        help='MCTS simulations per move')
    parser.add_argument('--output', type=str, default='nnue_data/training_data.npz',
                        help='Output path for training data')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device (cpu, cuda, mps)')
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load trained AlphaZero model
    print(f"Loading model from {args.model}...")
    network = AlphaZeroNet(device=args.device)
    
    if Path(args.model).exists():
        network.load(args.model)
        print("Model loaded successfully")
    else:
        print(f"WARNING: Model not found at {args.model}, using random weights")
    
    # Configure NNUE data generation
    config = NNUEConfig(
        write_minply=4,
        write_maxply=60,
        eval_limit=0.9,
        random_skip_rate=0.3,
        skip_endgame=True,
    )
    
    # Create generator
    generator = NNUEDataGenerator(
        network=network,
        config=config,
        num_simulations=args.simulations,
        seed=args.seed,
    )
    
    print(f"\n=== NNUE Data Generation ===")
    print(f"Games: {args.games}")
    print(f"Simulations: {args.simulations}")
    print(f"Skip settings: minply={config.write_minply}, maxply={config.write_maxply}, "
          f"eval_limit={config.eval_limit}, random_skip={config.random_skip_rate}")
    print()
    
    # Generate dataset
    dataset = generator.generate_dataset(
        num_games=args.games,
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
