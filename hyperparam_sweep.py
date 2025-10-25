"""
Hyperparameter sweep for PPO + ICM on Crafter.

This script generates configurations to test and can run them locally or on a cluster.
Based on learnings from manual experiments, we focus on the most impactful parameters.
"""

import itertools
import json
import argparse
from pathlib import Path
import subprocess
import sys


# Define hyperparameter search space based on our experiments
SEARCH_SPACE = {
    # Learning rate: 5e-4 worked best, explore nearby
    'lr': [3e-4, 5e-4, 7e-4],

    # Entropy: 0.001 worked, try slightly higher/lower
    'entropy_coef': [0.0005, 0.001, 0.0015],

    # ICM beta: 0.15 best, 0.2 good, explore this range
    'icm_beta': [0.10, 0.15, 0.20],

    # Hidden dimension: 1024 best, but expensive
    'hidden_dim': [512, 1024],

    # Clip epsilon: standard is 0.2, try tighter/looser
    'clip_epsilon': [0.15, 0.2, 0.25],
}


def generate_grid_configs():
    """Generate all combinations (grid search)."""
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)

    return configs


def generate_random_configs(n_samples=20, seed=42):
    """Generate random samples from search space (random search)."""
    import random
    random.seed(seed)

    configs = []
    for i in range(n_samples):
        config = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
        configs.append(config)

    return configs


def config_to_name(config):
    """Generate a short name for a configuration."""
    parts = []
    parts.append(f"lr{config['lr']:.0e}")
    parts.append(f"ent{config['entropy_coef']:.0e}")
    parts.append(f"beta{config['icm_beta']:.2f}")
    parts.append(f"hid{config['hidden_dim']}")
    parts.append(f"clip{config['clip_epsilon']:.2f}")
    return "_".join(parts)


def save_configs(configs, output_file):
    """Save configurations to JSON file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(configs, f, indent=2)

    print(f"✓ Saved {len(configs)} configurations to {output_file}")


def create_slurm_script(config_id, config, output_dir='sweep_results'):
    """Create a SLURM batch script for a single configuration."""
    name = config_to_name(config)

    script = f"""#!/bin/bash
#SBATCH --job-name=ppo_sweep_{config_id}
#SBATCH --output={output_dir}/logs/sweep_{config_id}_%j.out
#SBATCH --error={output_dir}/logs/sweep_{config_id}_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load environment
source ~/.bashrc
conda activate crafter

# Run training
python train_ppo_icm.py \\
    --steps 1000000 \\
    --lr {config['lr']} \\
    --entropy_coef {config['entropy_coef']} \\
    --icm_beta {config['icm_beta']} \\
    --hidden_dim {config['hidden_dim']} \\
    --clip_epsilon {config['clip_epsilon']} \\
    --outdir {output_dir}/run_{config_id}_{name}

# Evaluate the trained model
MODEL_PATH=$(find {output_dir}/run_{config_id}_{name} -name "ppo_icm_final.pt" | head -1)

if [ -f "$MODEL_PATH" ]; then
    python evaluate.py \\
        --model_path "$MODEL_PATH" \\
        --algorithm ppo \\
        --episodes 100 \\
        --outdir {output_dir}/eval_{config_id}_{name}

    echo "✓ Sweep {config_id} complete"
else
    echo "✗ Model not found for sweep {config_id}"
    exit 1
fi
"""

    return script


def create_slurm_array_script(n_configs, output_dir='sweep_results', configs_file='sweep_configs.json'):
    """Create a SLURM array job script that runs all configs."""

    script = f"""#!/bin/bash
#SBATCH --job-name=ppo_sweep
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output={output_dir}/logs/sweep_%A_%a.out
#SBATCH --error={output_dir}/logs/sweep_%A_%a.err
#SBATCH --array=0-{n_configs-1}

echo "========================================================"
echo "PPO Hyperparameter Sweep on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Config: $SLURM_ARRAY_TASK_ID / {n_configs-1}"
echo "Started at: $(date)"
echo "========================================================"

# Setup paths
export PATH="/usr/local/cuda-12.6/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"

# Change to project directory
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Working directory: $(pwd)"

# Show GPU info
echo -e "\\nGPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Use system Python
PYTHON_CMD="/usr/bin/python3"
echo -e "\\nUsing Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${{PWD}}:${{PYTHONPATH}}"

# Verify GPU access
echo -e "\\nVerifying GPU access..."
$PYTHON_CMD -c "import torch; print(f'CUDA available: {{torch.cuda.is_available()}}')"

# Run the sweep script with array task ID
echo -e "\\n========================================================"
echo "Running sweep configuration $SLURM_ARRAY_TASK_ID"
echo "========================================================"

$PYTHON_CMD run_single_sweep.py \\
    --config_file {configs_file} \\
    --config_id $SLURM_ARRAY_TASK_ID \\
    --output_dir {output_dir}

EXITCODE=$?

echo -e "\\n========================================================"
echo "Sweep $SLURM_ARRAY_TASK_ID finished at: $(date)"
echo "Exit code: $EXITCODE"
echo "========================================================"

exit $EXITCODE
"""

    return script


def create_sweep_runner_script():
    """Create a Python script that runs a single sweep configuration."""

    script = """#!/usr/bin/env python
\"\"\"Run a single hyperparameter sweep configuration.\"\"\"

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
"""

    return script


def main():
    parser = argparse.ArgumentParser(description='Generate hyperparameter sweep configurations')
    parser.add_argument('--mode', choices=['grid', 'random'], default='random',
                       help='Search strategy: grid (exhaustive) or random (sample)')
    parser.add_argument('--n_samples', type=int, default=20,
                       help='Number of random samples (if mode=random)')
    parser.add_argument('--output_dir', type=str, default='sweep_results',
                       help='Output directory for sweep results')
    parser.add_argument('--cluster', choices=['slurm', 'local'], default='local',
                       help='Target platform: slurm (cluster) or local')

    args = parser.parse_args()

    # Generate configurations
    if args.mode == 'grid':
        configs = generate_grid_configs()
        print(f"Generated {len(configs)} grid search configurations")
    else:
        configs = generate_random_configs(n_samples=args.n_samples)
        print(f"Generated {len(configs)} random search configurations")

    # Show sample configs
    print(f"\nSample configurations:")
    for i, config in enumerate(configs[:3]):
        print(f"{i}: {config}")
    if len(configs) > 3:
        print(f"... ({len(configs) - 3} more)")

    # Save configurations
    configs_file = Path(args.output_dir) / 'sweep_configs.json'
    save_configs(configs, configs_file)

    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / 'logs').mkdir(parents=True, exist_ok=True)

    # Create runner script
    runner_script = Path('run_single_sweep.py')
    with open(runner_script, 'w') as f:
        f.write(create_sweep_runner_script())
    runner_script.chmod(0o755)
    print(f"✓ Created {runner_script}")

    if args.cluster == 'slurm':
        # Create SLURM array job script
        slurm_script = Path(args.output_dir) / 'submit_sweep.sh'
        with open(slurm_script, 'w') as f:
            f.write(create_slurm_array_script(len(configs), args.output_dir, configs_file))
        slurm_script.chmod(0o755)
        print(f"✓ Created {slurm_script}")

        print(f"\n{'='*60}")
        print("To run on cluster:")
        print(f"1. Upload this directory to cluster")
        print(f"2. Run: sbatch {slurm_script}")
        print(f"3. Monitor: squeue -u $USER")
        print(f"4. Results will be in {args.output_dir}/")
        print(f"{'='*60}")

    else:
        print(f"\n{'='*60}")
        print("To run locally (WARNING: will take a LONG time):")
        print(f"for i in {{0..{len(configs)-1}}}; do")
        print(f"    python run_single_sweep.py --config_file {configs_file} --config_id $i --output_dir {args.output_dir}")
        print(f"done")
        print(f"{'='*60}")

    # Estimate compute time
    hours_per_run = 2.5
    total_hours = len(configs) * hours_per_run

    if args.cluster == 'slurm':
        parallel_hours = hours_per_run  # Assuming parallel execution
        print(f"\nEstimated time:")
        print(f"  - Sequential: {total_hours:.1f} hours ({total_hours/24:.1f} days)")
        print(f"  - Parallel (cluster): ~{parallel_hours:.1f} hours (if {len(configs)} GPUs available)")
    else:
        print(f"\nEstimated time: {total_hours:.1f} hours ({total_hours/24:.1f} days)")


if __name__ == '__main__':
    main()
