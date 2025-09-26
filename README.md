# Crafter RL Project

## Project Overview
Implementation of reinforcement learning agents for the Crafter survival game environment. This project implements two RL algorithms:
1. **Course Algorithm**: TBD (will be chosen based on course content)
2. **External Algorithm**: DrQ-v2 (Data-Regularized Q-Learning v2) - A state-of-the-art model-free RL algorithm for visual control

Each algorithm will undergo iterative improvements to optimize performance in the challenging Crafter survival environment.

## Assignment Details
- **Due Date**: October 22, 2025, 23:59
- **Team Size**: Maximum 4 people
- **Environment**: Crafter with partial observability (CrafterPartial-v1)
- **Observation Space**: 64x64 RGB images
- **Action Space**: 17 discrete actions

## Project Structure

```
crafter-rl-project/
├── src/
│   ├── agents/
│   │   ├── course_algorithm/     # Algorithm from course (e.g., PPO, DQN, A3C)
│   │   ├── external_algorithm/   # External algorithm not covered in course
│   │   ├── base/                 # Base implementations
│   │   └── improvements/         # Iterative improvements
│   ├── environment/
│   │   └── wrappers.py          # Environment wrappers and preprocessing
│   ├── utils/
│   │   ├── config.py           # Configuration management
│   │   ├── logger.py           # Logging utilities
│   │   └── metrics.py          # Performance metrics
│   ├── evaluation/
│   │   └── evaluator.py        # Agent evaluation scripts
│   ├── visualization/
│   │   └── plotter.py          # Results visualization
│   └── preprocessing/
│       └── observation.py      # Image preprocessing techniques
├── configs/                     # Configuration files
├── experiments/                 # Experiment tracking
│   ├── algorithm1/
│   │   ├── base/
│   │   ├── improvement1/
│   │   └── improvement2/
│   └── algorithm2/
│       ├── base/
│       ├── improvement1/
│       └── improvement2/
├── results/
│   ├── plots/                  # Generated plots
│   ├── logs/                   # Training logs
│   └── checkpoints/            # Model checkpoints
├── docs/                       # Documentation
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks for analysis
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
└── README.md                   # This file
```

## Key Metrics to Track
1. **Achievement Unlock Rate**: Percentage of times each of 22 achievements is unlocked
2. **Geometric Mean**: Overall score combining all achievements
3. **Survival Time**: Average timesteps survived per episode
4. **Cumulative Reward**: Total reward per episode

## Implementation Pipeline

### For Each Algorithm:
1. **Base Implementation** → Evaluate (Eval 1)
2. **Improvement 1** → Evaluate (Eval 2)
3. **Improvement 2** → Evaluate (Eval 3)
4. **Final Comparison** between both algorithms

## Algorithms

### Course Algorithm (TBD)
Will be selected based on course content (likely PPO, DQN, or A3C)

### External Algorithm: DrQ-v2
**Data-Regularized Q-Learning v2** is a model-free RL algorithm specifically designed for visual control tasks. Key features:
- Uses data augmentation (random shifts) for improved sample efficiency
- Combines DQN with continuous control techniques
- Particularly effective for pixel-based observations
- State-of-the-art performance on visual RL benchmarks

## Potential Improvements to Explore
- **Image preprocessing**: Grayscale conversion, frame stacking, normalization
- **Data augmentation**: Random crops, color jittering (DrQ-v2 specialty)
- **Reward shaping**: Survival bonuses, achievement rewards, exploration incentives
- **Network architectures**: CNN encoders, attention mechanisms
- **Exploration strategies**: ε-greedy variants, noise-based exploration
- **Replay buffer enhancements**: Prioritized replay, n-step returns
- **Curriculum learning**: Progressive task difficulty

## Setup Instructions

### Prerequisites
- Conda (Miniforge/Anaconda/Miniconda)
- Git
- Python 3.10 (will be installed via conda)

### Quick Setup

Run the automated setup script:
```bash
chmod +x setup_clean.sh
./setup_clean.sh
```

### Manual Setup (Alternative)

1. **Create conda environment with Python 3.10:**
```bash
conda create -n crafter_rl_env python=3.10 -y
conda activate crafter_rl_env
```

2. **Install dependencies in order:**
```bash
# Upgrade pip and setuptools
pip install --upgrade pip==23.3.1 setuptools==65.5.0 wheel

# Core packages
pip install numpy==1.24.3 opencv-python matplotlib

# Gym and Gymnasium with compatibility layer
pip install pygame gym==0.26.2 gymnasium shimmy

# PyTorch (CPU version for Mac compatibility)
pip install torch==2.0.1 torchvision==0.15.2

# RL Libraries
pip install stable-baselines3==2.1.0

# Install Crafter from GitHub
pip install git+https://github.com/danijar/crafter.git

# Additional utilities
pip install tensorboard tqdm pyyaml pandas
```

3. **Fix OpenMP issue (Mac only):**
If you encounter OpenMP errors on macOS:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
# Or run the fix script:
chmod +x fix_openmp.sh
./fix_openmp.sh
```

### Verify Installation

Test that everything is working:
```bash
python test_env.py
```

You should see:
- ✓ Environment created successfully
- ✓ Random actions executed successfully
- ✓ All tests passed!

## Running Experiments

### Training Agents

**Train with PPO (placeholder for course algorithm):**
```bash
python train.py --algorithm ppo --steps 1e6 --env_type partial
```

**Train with DQN:**
```bash
python train.py --algorithm dqn --steps 1e6 --env_type partial
```

**Train with DrQ-v2 (external algorithm - to be implemented):**
```bash
# Coming soon - will be in src/agents/external_algorithm/drqv2.py
python train_drqv2.py --steps 1e6 --env_type partial
```

### Training Options
- `--algorithm`: Choose from ['ppo', 'dqn', 'sac']
- `--env_type`: Choose from ['partial', 'reward', 'noreward']
- `--steps`: Total training steps (default: 1e6)
- `--eval_freq`: Evaluation frequency (default: 10000)
- `--save_freq`: Checkpoint save frequency (default: 50000)
- `--seed`: Random seed for reproducibility
- `--device`: Device to use ['cpu', 'cuda', 'auto']

### Monitoring Training

Training logs and checkpoints are saved in `results/logs/[algorithm]_[env_type]_[timestamp]/`

View tensorboard logs:
```bash
tensorboard --logdir results/logs
```

### Evaluating Agents
```bash
# Coming soon
python evaluate.py --checkpoint results/checkpoints/[model_name] --episodes 100
```

## Technical Notes

### Gymnasium/Gym Compatibility
This project uses:
- **Crafter**: Built with old Gym API (0.21-0.26)
- **Stable Baselines3**: Requires modern Gymnasium API
- **Shimmy**: Compatibility layer that bridges old Gym to Gymnasium

The `src/environment/crafter_env.py` wrapper handles all conversions automatically, so you can use Crafter with modern RL libraries seamlessly.

### Common Issues & Solutions

1. **OpenMP Error on Mac**:
   ```bash
   export KMP_DUPLICATE_LIB_OK=TRUE
   ```

2. **ImportError with distutils**: Make sure you're using Python 3.10, not 3.12

3. **Gym deprecation warnings**: These are suppressed in our wrapper, but are expected since Crafter uses old Gym internally

## Team Members
- Member 1: [Name] ([Student ID])
- Member 2: [Name] ([Student ID])
- Member 3: [Name] ([Student ID])
- Member 4: [Name] ([Student ID])

## References
- Crafter Paper: https://arxiv.org/pdf/2109.06780
- Crafter GitHub: https://github.com/danijar/crafter
- DrQ-v2 Paper: https://arxiv.org/abs/2107.09645
- Skeleton Code: https://github.com/rayrsys/Reinforcement-Learning-Project-2026-Crafter.git