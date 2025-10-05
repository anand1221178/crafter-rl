#!/bin/bash

#SBATCH --job-name=Crafter_PPO
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/ppo_%j.out
#SBATCH --error=logs/ppo_%j.err

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
python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
python -c "import crafter; print('Crafter: installed')"

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
echo "Starting PPO training on Crafter..."
echo "Training steps: 1,000,000"
echo "========================================================"

python train.py \
    --algorithm ppo \
    --steps 1000000 \
    --seed 42 \
    --eval_freq 50000

EXITCODE=$?

echo -e "\n========================================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXITCODE"
echo "========================================================"

# Show final results location
if [ $EXITCODE -eq 0 ]; then
    echo -e "\nTraining completed successfully!"
    echo "Results saved in: logdir/crafter_ppo_*/"
fi

exit $EXITCODE
