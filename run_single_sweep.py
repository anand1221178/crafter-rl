#!/usr/bin/env python
"""Run a single hyperparameter sweep configuration."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--config_id', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Load configurations
    with open(args.config_file, 'r') as f:
        configs = json.load(f)

    if args.config_id >= len(configs):
        print(f"Error: config_id {args.config_id} >= {len(configs)}")
        sys.exit(1)

    config = configs[args.config_id]

    # Generate config name
    name = f"lr{config['lr']:.0e}_ent{config['entropy_coef']:.0e}_beta{config['icm_beta']:.2f}_hid{config['hidden_dim']}_clip{config['clip_epsilon']:.2f}"

    print(f"=" * 60)
    print(f"Running sweep {args.config_id}/{len(configs)}")
    print(f"Config: {config}")
    print(f"=" * 60)

    # Run training
    train_dir = Path(args.output_dir) / f"run_{args.config_id}_{name}"
    train_cmd = [
        'python3', 'train_ppo_icm.py',
        '--steps', '1000000',
        '--lr', str(config['lr']),
        '--entropy_coef', str(config['entropy_coef']),
        '--icm_beta', str(config['icm_beta']),
        '--hidden_dim', str(config['hidden_dim']),
        '--clip_epsilon', str(config['clip_epsilon']),
        '--outdir', str(train_dir)
    ]

    print(f"Running: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd)

    if result.returncode != 0:
        print(f"✗ Training failed for config {args.config_id}")
        sys.exit(1)

    # Find trained model
    model_paths = list(train_dir.glob('**/ppo_icm_final.pt'))
    if not model_paths:
        print(f"✗ Model not found in {train_dir}")
        sys.exit(1)

    model_path = model_paths[0]
    print(f"Found model: {model_path}")

    # Run evaluation
    eval_dir = Path(args.output_dir) / f"eval_{args.config_id}_{name}"
    eval_cmd = [
        'python3', 'evaluate.py',
        '--model_path', str(model_path),
        '--algorithm', 'ppo',
        '--episodes', '100',
        '--outdir', str(eval_dir)
    ]

    print(f"Running: {' '.join(eval_cmd)}")
    result = subprocess.run(eval_cmd)

    if result.returncode != 0:
        print(f"✗ Evaluation failed for config {args.config_id}")
        sys.exit(1)

    print(f"✓ Sweep {args.config_id} complete!")
    print(f"Results in: {eval_dir}")


if __name__ == '__main__':
    main()
