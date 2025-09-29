#!/bin/bash

echo "==================================="
echo "Crafter RL Project Setup"
echo "==================================="
echo ""

# Check if Python 3.10+ is available
if ! python3 --version | grep -E "3\.(10|11|12)" > /dev/null; then
    echo "Error: Python 3.10+ is required"
    echo "Current version: $(python3 --version)"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and setuptools first
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install packages from requirements.txt
echo ""
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

# All packages now installed from requirements.txt including crafter
echo ""
echo "Installation complete!"

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the setup:"
echo "  python test_env.py  # (create this file to test)"
echo ""
echo "To start training:"
echo "  python train.py --algorithm ppo --steps 1000000"
echo "  python train.py --algorithm drqv2 --steps 1000000"
echo ""
echo "To evaluate models:"
echo "  python evaluate.py --model_path models/model.zip --algorithm ppo"