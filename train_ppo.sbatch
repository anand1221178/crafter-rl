#!/bin/bash

#SBATCH --job-name=Crafter_PPO_Baseline
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/ppo_baseline_%j.out
#SBATCH --error=logs/ppo_baseline_%j.err

echo "========================================================"
echo "Crafter PPO Training on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "========================================================"

# Setup CUDA paths
export PATH="/usr/local/cuda-12.6/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"

# Change to project directory
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Working directory: $(pwd)"

# Show GPU info
echo -e "\nGPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

# Initialize conda (adjust path if needed)
echo -e "\nInitializing conda..."
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh

# Create/update conda environment from YAML
echo -e "\nSetting up conda environment..."
if conda env list | grep -q "^crafter_env "; then
    echo "Environment 'crafter_env' exists, updating..."
    conda env update -f crafter_env.yaml --prune
else
    echo "Creating environment 'crafter_env' from crafter_env.yaml..."
    conda env create -f crafter_env.yaml
fi

# Activate environment
echo -e "\nActivating conda environment 'crafter_env'..."
conda activate crafter_env

# Verify installations
echo -e "\nVerifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
python -c "import crafter; print('Crafter: installed')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Verify GPU access
echo -e "\nVerifying GPU access..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Create logs directory
mkdir -p logs

# Run training
echo -e "\n========================================================"
echo "Starting PPO Baseline Training on Crafter..."
echo "Algorithm: Custom PPO (Policy Gradient)"
echo "Training steps: 1,000,000"
echo "========================================================"
echo "Hyperparameters:"
echo "  n_steps: 2048 (rollout size)"
echo "  batch_size: 64"
echo "  n_epochs: 10"
echo "  lr: 3e-4"
echo "  gamma: 0.99"
echo "  gae_lambda: 0.95"
echo "  clip_epsilon: 0.2"
echo "  entropy_coef: 0.01"
echo "========================================================"

python train_ppo.py \
    --steps 1000000 \
    --seed 42 \
    --n_steps 2048 \
    --batch_size 64 \
    --n_epochs 10 \
    --lr 3e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_epsilon 0.2 \
    --entropy_coef 0.01 \
    --save_freq 50000 \
    --outdir logs/ppo

EXITCODE=$?

echo -e "\n========================================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXITCODE"
echo "========================================================"

# Show final results location
if [ $EXITCODE -eq 0 ]; then
    echo -e "\nTraining completed successfully!"
    echo "Results saved in: logs/ppo/ppo_baseline_*/"
    echo ""
    echo "Next steps:"
    echo "1. Evaluate model: python evaluate.py --model_path logs/ppo/ppo_baseline_*/ppo_baseline_final.pt"
    echo "2. Analyze results: Check logs/ppo/ppo_baseline_*/stats.jsonl for Crafter metrics"
    echo "3. Compare with DQN baseline for report"
fi

exit $EXITCODE
