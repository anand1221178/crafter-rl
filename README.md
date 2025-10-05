# Crafter RL Project

## Project Overview
Implementation of reinforcement learning agents for the Crafter survival game environment. This project implements two RL algorithms:
1. **Course Algorithm**: DQN (Deep Q-Network) - Model-free value-based RL
2. **External Algorithm**: Dyna-Q - Model-based RL with integrated planning

Each algorithm will undergo iterative improvements to optimize performance in the challenging Crafter survival environment.

## Assignment Details
- **Due Date**: October 22, 2025, 23:59
- **Team Size**: 2 members
- **Environment**: Crafter with partial observability (CrafterPartial-v1)
- **Observation Space**: 64x64 RGB images
- **Action Space**: 17 discrete actions

## Project Structure

```
crafter-rl-project/
├── crafter_env.yaml            # Conda environment specification
├── CLAUDE.md                   # Project tracker with Dyna-Q strategy
├── train.py                    # Unified training script (PPO, DQN, Dyna-Q)
├── evaluate.py                 # Comprehensive evaluation script
├── test_model.py               # Quick model testing script
├── train_ppo.sbatch            # Cluster training script (PPO)
├── train_dqn.sbatch            # Cluster training script (DQN)
├── train_dynaq.sbatch          # Cluster training script (Dyna-Q)
├── src/
│   ├── agents/
│   │   ├── base_agent.py       # Abstract base class for all agents
│   │   └── dynaq_agent.py      # Dyna-Q implementation (to be implemented)
│   ├── models/
│   │   ├── world_model.py      # Environment dynamics model (to be implemented)
│   │   └── prioritized_sweeping.py  # Priority queue for planning (to be implemented)
│   ├── utils/
│   │   ├── networks.py         # Q-network (CNN for 64x64 RGB)
│   │   └── replay_buffer.py    # Experience replay
│   └── evaluation/
│       ├── plot_reward.py      # Reward plotting utilities
│       ├── plot_scores.py      # Achievement score plotting
│       └── read_metrics.py     # Metrics reading from Crafter logs
├── models/                     # Saved model checkpoints
├── results/                    # Training results and evaluations
├── logdir/                     # Training logs (Crafter metrics)
└── README.md                   # This file
```

## Key Metrics to Track
1. **Achievement Unlock Rate**: Percentage of times each of 22 achievements is unlocked
2. **Geometric Mean (Crafter Score)**: Overall score combining all achievements
3. **Survival Time**: Average timesteps survived per episode
4. **Cumulative Reward**: Total reward per episode

## Implementation Pipeline

### For Each Algorithm:
1. **Base Implementation** → Evaluate (Eval 1)
2. **Improvement 1** → Evaluate (Eval 2)
3. **Improvement 2** → Evaluate (Eval 3)
4. **Final Comparison** between both algorithms

## Algorithms

### Course Algorithm: DQN (Deep Q-Network)
Model-free value-based RL using Stable-Baselines3 implementation:
- Q-learning with neural network function approximation
- Experience replay for sample efficiency
- Target network for stable training
- Epsilon-greedy exploration

### External Algorithm: Dyna-Q (Model-Based RL)
**Integrated Planning and Learning** - Classic model-based RL algorithm:
- Learns environment dynamics (world model)
- Combines real experience with simulated planning
- Sample efficient: 1 real experience → N planning updates
- Three phases of improvement:
  1. **Eval 1**: Baseline Dyna-Q (random planning)
  2. **Eval 2**: + Prioritized Sweeping (focused planning)
  3. **Eval 3**: + Dyna-Q+ (exploration bonuses)

**Why Model-Based RL for Crafter?**
- Sparse rewards benefit from planning (simulate rare experiences)
- Multi-step reasoning (chop tree → wood → stick → sword)
- Sample efficient learning from limited interactions

## Setup Instructions

### Quick Setup (Recommended)

**1. Create conda environment from YAML:**
```bash
# Clone repository
git clone <your-repo-url>
cd crafter-rl-project

# Create environment
conda env create -f crafter_env.yaml
conda activate crafter_env
```

**2. Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import crafter; print('Crafter: installed')"
python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
```

### Manual Setup (Alternative)

If you prefer manual installation:

```bash
# Create environment
conda create -n crafter_env python=3.10 -y
conda activate crafter_env

# Install PyTorch (adjust for your CUDA version)
conda install pytorch=2.8.0 -c pytorch -y

# Install dependencies
pip install stable-baselines3 crafter gymnasium shimmy wandb imageio imageio-ffmpeg
```

## 🧪 Local Testing Commands

### Create Environment
```bash
# Create conda environment locally
conda env create -f crafter_env.yaml
conda activate crafter_env
```

### Training Algorithms Locally
```bash
# Quick PPO test (10K steps, ~5 minutes)
python train.py --algorithm ppo --steps 10000

# Quick DQN test (10K steps, ~5 minutes)
python train.py --algorithm dqn --steps 10000

# Dyna-Q test (when implemented)
python train.py --algorithm dynaq --steps 10000 --planning_steps 5

# Full training (1M steps, ~4-8 hours on GPU)
python train.py --algorithm dqn --steps 1000000
```

### Evaluating Models
```bash
# Comprehensive evaluation (100 episodes, detailed analysis)
python evaluate.py \
    --model_path models/dqn_final.zip \
    --algorithm dqn \
    --episodes 100

# Quick test (10 episodes, basic metrics)
python test_model.py models/dqn_final.zip dqn 10

# Evaluate from training logdir (analyze existing stats.jsonl)
python evaluate.py \
    --logdir logdir/crafter_dqn_20251005_180000/ \
    --algorithm dqn \
    --episodes 100
```

### Evaluation Outputs
Evaluation generates:
- 📊 Crafter Score (geometric mean of achievements)
- 📈 Achievement unlock rates (all 22 achievements)
- 🎯 Average reward and episode length
- 📊 Plots (achievement rates, summary metrics)
- 📄 JSON + text reports

## 🚀 Cluster Training Commands

### Submit Training Jobs

```bash
# Submit DQN training (partner's course algorithm)
sbatch train_dqn.sbatch

# Submit Dyna-Q training (external algorithm)
sbatch train_dynaq.sbatch

# Submit PPO training (baseline comparison)
sbatch train_ppo.sbatch
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f logs/dqn_<job_id>.out
tail -f logs/dynaq_<job_id>.out

# Cancel job
scancel <job_id>
```

### Training Options

All training scripts support the following arguments:

```bash
python train.py [OPTIONS]

Options:
  --algorithm {ppo,dqn,dynaq}  Algorithm to train
  --steps STEPS                Training steps (default: 1M)
  --seed SEED                  Random seed (default: 42)
  --eval_freq FREQ             Evaluation frequency (default: 50K)

  # Dyna-Q specific:
  --planning_steps N           Planning steps per real step (default: 5)
  --prioritized                Use prioritized sweeping
  --exploration_bonus KAPPA    Exploration bonus for Dyna-Q+ (default: 0.0)
```

## Training Results

Results are saved in timestamped directories:
```
logdir/crafter_{algorithm}_{timestamp}/
├── stats.jsonl              # Episode statistics (Crafter format)
├── {algorithm}_final.zip    # Final model checkpoint (SB3 format)
└── {algorithm}_final.pt     # Final model checkpoint (PyTorch format)
```

## Implementation Status

### ✅ Completed
- [x] Project setup and conda environment
- [x] Unified training script (PPO, DQN, Dyna-Q)
- [x] Evaluation infrastructure
- [x] Cluster training scripts (conda-based)
- [x] Documentation (CLAUDE.md with Dyna-Q strategy)

### 🚧 In Progress
- [ ] Dyna-Q agent implementation (`src/agents/dynaq_agent.py`)
- [ ] World model (`src/models/world_model.py`)
- [ ] Prioritized sweeping (`src/models/prioritized_sweeping.py`)

### 📋 Planned
- [ ] Baseline Dyna-Q training (Eval 1)
- [ ] Prioritized sweeping improvement (Eval 2)
- [ ] Dyna-Q+ with exploration bonuses (Eval 3)
- [ ] Final comparison (DQN vs Dyna-Q)
- [ ] Report writing

## Expected Results

| Evaluation | Algorithm | Target Score | Key Feature |
|------------|-----------|--------------|-------------|
| **Eval 1** | Baseline Dyna-Q | 0.5-2% | Planning (5 steps/real step) |
| **Eval 2** | + Prioritized Sweeping | 3-8% | Focused planning |
| **Eval 3** | + Exploration Bonus | 8-15% | Directed exploration |

**Sample Efficiency Analysis:**
- **DQN (model-free)**: ~800K-1M steps to achieve 1% score
- **Dyna-Q (model-based)**: ~200K-400K steps to achieve 1% score (2-5× faster)

## Technical Notes

### Crafter Environment
- **Direct API**: Uses `crafter.Env()` directly (bypasses Gym/Gymnasium)
- **Wrapper**: `CrafterWrapper` in train.py handles API normalization
- **Recorder**: `crafter.Recorder` automatically logs stats.jsonl
- **Observations**: 64×64×3 RGB numpy arrays
- **Actions**: 17 discrete actions (0-16)

### Conda vs Pip
This project uses **conda** for reproducible environments:
- ✅ Consistent Python version (3.10)
- ✅ Compatible PyTorch + CUDA on cluster
- ✅ Faster package resolution
- ✅ Better dependency isolation

### Cluster Configuration
Scripts are configured for:
- **Partition**: `bigbatch`
- **GPUs**: 1 GPU per job (CUDA 12.6)
- **CPUs**: 16 cores
- **Time**: 24 hours
- **Conda**: Auto-creates environment from YAML

## Team Members
- **Anand Patel** (Student #: _TO_BE_FILLED_)
  - Role: Dyna-Q Implementation (External Algorithm)
- **Partner Name** (Student #: _TO_BE_FILLED_)
  - Role: DQN Implementation (Course Algorithm)

## References

### Papers
1. **Dyna-Q**: Sutton & Barto 1996/2018 - Reinforcement Learning: An Introduction (Chapter 8)
2. **Prioritized Sweeping**: Moore & Atkeson 1993
3. **Crafter Benchmark**: Hafner 2021 - https://arxiv.org/abs/2109.06780

### Code
- **Crafter GitHub**: https://github.com/danijar/crafter
- **Skeleton Code**: https://github.com/rayrsys/Reinforcement-Learning-Project-2026-Crafter.git
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/

## License
This project is for educational purposes as part of COMS4061A/COMS7071A coursework.
