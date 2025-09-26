#!/bin/bash

echo "==================================="
echo "Crafter RL Project Setup"
echo "==================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Remove old environment if it exists
echo "Checking for existing environment..."
if conda env list | grep -q "^crafter_rl_env "; then
    echo "Removing existing crafter_rl_env..."
    conda env remove -n crafter_rl_env -y
fi

# Create new environment
echo ""
echo "Creating new conda environment with Python 3.10..."
conda create -n crafter_rl_env python=3.10 -y

echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate crafter_rl_env

# Install packages step by step
echo ""
echo "Installing core packages..."
pip install --upgrade pip==23.3.1
pip install setuptools==65.5.0 wheel numpy==1.24.3

echo ""
echo "Installing OpenCV and visualization tools..."
pip install opencv-python==4.8.1.78 matplotlib==3.7.2 Pillow==10.0.0

echo ""
echo "Installing Gym and Gymnasium..."
pip install pygame==2.5.2
pip install gym==0.26.2
pip install gymnasium==0.29.1
pip install shimmy==1.3.0

echo ""
echo "Installing PyTorch (CPU version for Mac compatibility)..."
pip install torch==2.0.1 torchvision==0.15.2

echo ""
echo "Installing Stable Baselines3..."
pip install stable-baselines3==2.1.0

echo ""
echo "Installing Crafter from GitHub..."
pip install git+https://github.com/danijar/crafter.git

echo ""
echo "Installing additional utilities and experiment tracking..."
pip install tensorboard==2.14.0 tqdm==4.66.1 pyyaml==6.0.1 pandas==2.0.3
pip install wandb seaborn jupyter

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate crafter_rl_env"
echo ""
echo "To test the setup:"
echo "  python test_env.py"
echo ""
echo "To start training with W&B tracking:"
echo "  wandb login  # First time only"
echo "  python train.py --algorithm ppo --steps 100000 --wandb_project crafter-rl"