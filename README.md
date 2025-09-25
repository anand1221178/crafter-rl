# Crafter RL Project

## Project Overview
Implementation of reinforcement learning agents for the Crafter survival game environment. This project explores two different RL algorithms with iterative improvements to optimize performance in a challenging open-world survival setting.

## Assignment Requirements
- **Due Date**: October 22, 2025, 23:59
- **Team Size**: Maximum 4 people
- **Environment**: Crafter with partial observability (CrafterPartial-v1)

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

## Potential Improvements to Explore
- Image preprocessing and feature extraction
- Reward shaping
- Action/observation space modifications
- Curriculum learning
- Different network architectures
- Exploration strategies
- Memory/experience replay enhancements

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/rayrsys/Reinforcement-Learning-Project-2026-Crafter.git
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate crafter-rl
```

### 3. Install Crafter
```bash
pip install crafter
```

### 4. Verify installation
```bash
python -c "import gymnasium as gym; env = gym.make('CrafterPartial-v1')"
```

## Running Experiments

### Training Base Agent
```bash
python src/train.py --algorithm [algorithm_name] --config configs/base_config.yaml
```

### Evaluating Agent
```bash
python src/evaluate.py --checkpoint results/checkpoints/[model_name] --episodes 100
```

## Team Members
- Member 1: [Name] ([Student ID])
- Member 2: [Name] ([Student ID])
- Member 3: [Name] ([Student ID])
- Member 4: [Name] ([Student ID])

## References
- Crafter Paper: https://arxiv.org/pdf/2109.06780
- Crafter GitHub: https://github.com/danijar/crafter
- Skeleton Code: https://github.com/rayrsys/Reinforcement-Learning-Project-2026-Crafter.git