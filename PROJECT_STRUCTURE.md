# Project Structure Guide

## Directory Organization and Purpose

### 📁 `src/` - Source Code
Main codebase containing all implementation files.

#### 📂 `src/agents/`
Contains all RL agent implementations.

- **`course_algorithm/`**: Place your chosen algorithm from the course here (e.g., PPO, DQN, A3C, REINFORCE)
  - Example files: `ppo_agent.py`, `dqn_agent.py`

- **`external_algorithm/`**: Place your external algorithm here (e.g., Rainbow, SAC, TD3, IMPALA)
  - Must include explanation in documentation
  - Example files: `rainbow_agent.py`, `sac_agent.py`

- **`base/`**: Base implementations of both algorithms before improvements
  - `base_agent.py`: Abstract base class
  - `algorithm1_base.py`: First algorithm base implementation
  - `algorithm2_base.py`: Second algorithm base implementation

- **`improvements/`**: Iterative improvements for both algorithms
  - `algorithm1_improvement1.py`: First improvement for algorithm 1
  - `algorithm1_improvement2.py`: Second improvement for algorithm 1
  - `algorithm2_improvement1.py`: First improvement for algorithm 2
  - `algorithm2_improvement2.py`: Second improvement for algorithm 2

#### 📂 `src/environment/`
Environment-related code and wrappers.

- `wrappers.py`: Gymnasium wrappers for Crafter environment
- `crafter_env.py`: Custom environment modifications if needed
- `reward_shaping.py`: Reward shaping implementations

#### 📂 `src/utils/`
Utility functions and helper modules.

- `config.py`: Configuration management and hyperparameter handling
- `logger.py`: Logging setup and management
- `metrics.py`: Metric calculation functions
- `buffer.py`: Experience replay buffer implementations
- `networks.py`: Neural network architectures

#### 📂 `src/evaluation/`
Evaluation and testing scripts.

- `evaluator.py`: Main evaluation script for agents
- `benchmark.py`: Benchmarking utilities
- `achievement_tracker.py`: Track 22 Crafter achievements

#### 📂 `src/visualization/`
Plotting and visualization tools.

- `plotter.py`: Generate plots for results
- `video_recorder.py`: Record agent gameplay
- `dashboard.py`: Real-time training dashboard

#### 📂 `src/preprocessing/`
Data preprocessing modules.

- `observation.py`: Image preprocessing (grayscale, normalization, feature extraction)
- `state_encoder.py`: State encoding techniques
- `augmentation.py`: Data augmentation strategies

### 📁 `configs/` - Configuration Files
YAML/JSON configuration files for experiments.

- `base_config.yaml`: Default configuration
- `ppo_config.yaml`: PPO-specific configuration
- `dqn_config.yaml`: DQN-specific configuration
- `hyperparameters/`: Directory for hyperparameter sweep configs

### 📁 `experiments/` - Experiment Tracking
Organized tracking of all experiments.

#### 📂 `algorithm1/` (e.g., PPO)
- **`base/`**: Base implementation results
  - `config.yaml`: Configuration used
  - `metrics.json`: Performance metrics
  - `notes.md`: Observations and analysis

- **`improvement1/`**: First improvement results
  - `config.yaml`: Updated configuration
  - `metrics.json`: Performance metrics
  - `notes.md`: What was changed and why

- **`improvement2/`**: Second improvement results
  - `config.yaml`: Final configuration
  - `metrics.json`: Performance metrics
  - `notes.md`: Final improvements

#### 📂 `algorithm2/` (e.g., Rainbow)
Same structure as algorithm1/

### 📁 `results/` - Output Files
All generated outputs from training and evaluation.

#### 📂 `plots/`
Generated visualizations.
- `learning_curves/`: Training progress plots
- `achievement_rates/`: Achievement unlock visualizations
- `comparisons/`: Algorithm comparison plots
- `heatmaps/`: Performance heatmaps

#### 📂 `logs/`
Training and evaluation logs.
- `tensorboard/`: TensorBoard log files
- `training_logs/`: Text-based training logs
- `debug_logs/`: Debugging information

#### 📂 `checkpoints/`
Saved model weights.
- `algorithm1_base.pth`: Base model checkpoint
- `algorithm1_imp1.pth`: Improvement 1 checkpoint
- `algorithm1_imp2.pth`: Improvement 2 checkpoint
- Similar for algorithm2

### 📁 `docs/` - Documentation
Project documentation and reports.

- `algorithm_explanation.md`: Explanation of external algorithm
- `improvement_rationale.md`: Reasoning for each improvement
- `results_analysis.md`: Detailed results analysis
- `final_report.pdf`: Final submission report

### 📁 `tests/` - Unit Tests
Testing files for code validation.

- `test_agents.py`: Test agent implementations
- `test_environment.py`: Test environment wrappers
- `test_preprocessing.py`: Test preprocessing functions

### 📁 `notebooks/` - Jupyter Notebooks
Interactive analysis and experimentation.

- `data_exploration.ipynb`: Explore Crafter environment
- `results_analysis.ipynb`: Analyze experimental results
- `hyperparameter_tuning.ipynb`: Hyperparameter experiments
- `visualization.ipynb`: Create custom visualizations

### 📄 Root Files

- **`requirements.txt`**: Python package dependencies
- **`environment.yml`**: Conda environment specification
- **`train.py`**: Main training script
- **`evaluate.py`**: Main evaluation script
- **`compare_agents.py`**: Compare final agents
- **`.gitignore`**: Git ignore patterns
- **`Makefile`**: Common commands and shortcuts

## File Naming Conventions

### Code Files
- Use snake_case: `ppo_agent.py`, `reward_shaping.py`
- Be descriptive: `image_preprocessing.py` not `preproc.py`

### Config Files
- Include algorithm and variant: `ppo_base_config.yaml`, `dqn_improvement1_config.yaml`

### Result Files
- Include timestamp and description: `2025_10_15_ppo_base_results.json`
- Checkpoints: `[algorithm]_[variant]_[episode].pth`

### Documentation
- Use clear titles: `PPO_Algorithm_Explanation.md`
- Include dates in experiment notes

## Workflow Guide

### 1. Starting with Base Implementation
1. Choose your two algorithms
2. Implement in `src/agents/course_algorithm/` and `src/agents/external_algorithm/`
3. Create configs in `configs/`
4. Run initial experiments, save to `experiments/algorithm1/base/`

### 2. Making Improvements
1. Analyze results from base implementation
2. Implement improvement in `src/agents/improvements/`
3. Update configuration
4. Run experiments, save to `experiments/algorithm1/improvement1/`
5. Document rationale in `experiments/algorithm1/improvement1/notes.md`

### 3. Evaluation
1. Use scripts in `src/evaluation/`
2. Save metrics to `results/logs/`
3. Generate plots, save to `results/plots/`
4. Save checkpoints to `results/checkpoints/`

### 4. Final Comparison
1. Load best checkpoints from each algorithm
2. Run `compare_agents.py`
3. Generate comparison plots
4. Write final analysis in `docs/`

## Important Notes

- **Version Control**: Commit after each experiment iteration
- **Documentation**: Document every design decision
- **Reproducibility**: Always save configs and random seeds
- **Organization**: Keep experiments separated by algorithm and improvement
- **Backups**: Regularly backup checkpoints and results

## Quick Start Commands

```bash
# Train base PPO agent
python train.py --algorithm ppo --variant base --config configs/ppo_base_config.yaml

# Train improved PPO agent
python train.py --algorithm ppo --variant improvement1 --config configs/ppo_improvement1_config.yaml

# Evaluate trained agent
python evaluate.py --checkpoint results/checkpoints/ppo_improvement1.pth --episodes 100

# Compare two agents
python compare_agents.py --agent1 ppo_improvement2 --agent2 rainbow_improvement2

# Generate plots
python src/visualization/plotter.py --experiment experiments/algorithm1/

# Run tests
pytest tests/
```