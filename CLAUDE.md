# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Reinforcement Learning project implementing agents for the Crafter survival game environment. The project requires implementing two RL algorithms (one from course, one external) with iterative improvements to optimize performance.

## Key Requirements
- Due Date: October 22, 2025
- Environment: CrafterPartial-v1 (64x64 RGB images, 17 actions, 22 achievements)
- Two algorithms with minimum 2 improvements each
- Evaluation metrics: Achievement unlock rate, survival time, cumulative reward

## Common Commands

### Setup and Installation
```bash
# Create conda environment
conda env create -f environment.yml
conda activate crafter-rl

# Install dependencies
pip install -r requirements.txt

# Verify Crafter installation
python -c "import gymnasium as gym; env = gym.make('CrafterPartial-v1')"
```

### Training
```bash
# Train base agent
python train.py --algorithm [ppo|dqn|rainbow|sac] --variant base --config configs/[algorithm]_base_config.yaml

# Train improved agent
python train.py --algorithm [algorithm] --variant improvement[1|2] --config configs/[algorithm]_improvement[1|2]_config.yaml
```

### Evaluation
```bash
# Evaluate agent performance
python evaluate.py --checkpoint results/checkpoints/[model_name].pth --episodes 100

# Compare two agents
python compare_agents.py --agent1 [name1] --agent2 [name2]
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_agents.py -v
```

## Architecture Overview

### Agent Implementation Flow
1. **Base Implementation**: Start in `src/agents/course_algorithm/` or `src/agents/external_algorithm/`
2. **Improvements**: Implement in `src/agents/improvements/` with clear versioning
3. **Configuration**: Define hyperparameters in `configs/` directory
4. **Training**: Use `train.py` with appropriate config
5. **Evaluation**: Track metrics using `src/evaluation/evaluator.py`

### Key Components

#### Environment Wrapper (`src/environment/wrappers.py`)
- Handles Gymnasium API compatibility
- Preprocesses observations (64x64 RGB images)
- Manages action space (17 discrete actions)
- Tracks achievements and rewards

#### Agent Base Class (`src/agents/base/base_agent.py`)
- Defines common interface for all agents
- Methods: `act()`, `learn()`, `save()`, `load()`

#### Evaluation System (`src/evaluation/`)
- Tracks 22 Crafter achievements
- Calculates geometric mean of achievement rates
- Monitors survival time and cumulative reward

## Important Implementation Notes

### Crafter Environment Specifics
- **Observation Space**: 64x64x3 RGB images
- **Action Space**: 17 discrete actions (movement, interaction, crafting)
- **Rewards**: Survival (+1 per timestep) + Achievement bonuses
- **Achievements**: 22 total (wood collection, tool crafting, combat, etc.)

### Required Improvements (Non-trivial)
Examples of valid improvements:
- Image preprocessing (grayscale, normalization, feature extraction)
- Reward shaping strategies
- Network architecture modifications
- Exploration strategy enhancements
- Curriculum learning implementation
- Memory/replay buffer optimizations

Invalid improvements:
- Simple hyperparameter tuning only
- Minor code refactoring
- Documentation changes

### Evaluation Pipeline
For each algorithm:
1. Base implementation → Eval 1
2. Improvement 1 → Eval 2
3. Improvement 2 → Eval 3
4. Final comparison between algorithms

## File Organization

### Experiment Tracking
Each experiment should have:
- `config.yaml`: Full configuration used
- `metrics.json`: Performance metrics
- `notes.md`: Design decisions and rationale
- Model checkpoints in `results/checkpoints/`

### Naming Conventions
- Checkpoints: `[algorithm]_[variant]_[episode].pth`
- Configs: `[algorithm]_[variant]_config.yaml`
- Logs: `[date]_[algorithm]_[variant].log`

## Common Issues and Solutions

### GPU Memory
- Reduce batch size if OOM errors occur
- Use gradient accumulation for effective larger batches

### Training Instability
- Check learning rate scheduling
- Verify reward normalization
- Monitor gradient norms

### Poor Performance
- Ensure proper observation preprocessing
- Verify action masking if applicable
- Check exploration parameters

## Dependencies
Key packages (see requirements.txt for full list):
- gymnasium
- crafter
- stable-baselines3 (if using)
- torch/tensorflow
- numpy
- matplotlib
- wandb (for experiment tracking)