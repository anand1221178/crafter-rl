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


# Setup paths
export PATH="/usr/local/cuda-12.6/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"

# Change to project directory
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Working directory: $(pwd)"

# Show GPU info
echo -e "\nGPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

# Use system Python directly
PYTHON_CMD="/usr/bin/python3"
echo -e "\nUsing system Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Install pip for user if needed
echo -e "\nEnsuring pip is available..."
$PYTHON_CMD -m ensurepip --user 2>/dev/null || {
    echo "Installing pip manually..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON_CMD get-pip.py --user
    rm get-pip.py
}

# Upgrade pip
echo -e "\nUpgrading pip..."
$PYTHON_CMD -m pip install --user --upgrade pip

# Install packages from requirements.txt to user directory
echo -e "\nInstalling packages from requirements.txt..."
$PYTHON_CMD -m pip install --user -r requirements.txt

# Verify installations
echo -e "\nVerifying installations..."
$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "PyTorch not found"
$PYTHON_CMD -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')" || echo "Gymnasium not found"
$PYTHON_CMD -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')" || echo "SB3 not found"
$PYTHON_CMD -c "import crafter; print(f'Crafter: installed')" || echo "Crafter not found"
$PYTHON_CMD -c "import numpy; print(f'NumPy: {numpy.__version__}')" || echo "NumPy not found"

# Verify GPU access
echo -e "\nVerifying GPU access..."
$PYTHON_CMD -c "
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

# Add user site-packages to Python path
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training
echo -e "\n========================================================"
echo "Starting PPO training on Crafter..."
echo "Training steps: 1,000,000"
echo "========================================================"

# Run PPO training (1M steps as per assignment)
$PYTHON_CMD train.py \
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
    echo "Model saved in: models/"
fi

exit $EXITCODE