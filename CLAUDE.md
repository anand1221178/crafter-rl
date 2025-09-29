# Reinforcement Learning Assignment - CLAUDE Project Tracker
**COMS4061A/COMS7071A - Group Project**

## Group Information
- **Due Date**: October 22, 2025, 23:59
- **Group Size**: 2 members
- **Environment**: Crafter (Gymnasium interface)

### Group Members & Responsibilities
- **Anand Patel** (Student #: _TO_BE_FILLED_)
  - **Role**: DrQ-v2 Implementation (External Algorithm)
  - **Focus**: Visual RL, data augmentation, replay buffers
- **Partner Name** (Student #: _TO_BE_FILLED_)
  - **Role**: Course Algorithm Implementation (e.g., PPO/DQN)
  - **Focus**: Policy optimization, stable baselines integration

## Assignment Overview
This project involves implementing and iteratively improving two RL agents in the Crafter environment:
1. One algorithm **from the course**
2. One algorithm **NOT covered in the course**

Each agent must go through iterative improvement cycles: Base → Eval 1 → Improvement 1 → Eval 2 → Improvement 2 → Eval 3

## Environment Details
- **Crafter**: Procedurally generated 2D survival game
- **Observation Space**: 64x64 RGB images
- **Action Space**: 17 discrete actions
- **Environment**: `gym.make("CrafterPartial-v1")`
- **Key Mechanics**: Resource gathering, crafting, combat, survival (hunger/health)
- **Rewards**: +1 survival reward per timestep + achievement rewards (22 total)

## Project Requirements Checklist

### Core Implementation (Divided by Person)

### 🎯 ANAND'S WORK: DrQ-v2 (External Algorithm)
- [ ] **Research & Explain DrQ-v2** (for report section)
- [ ] **Base Implementation**
  - [ ] DrQ-v2 agent class inheriting from BaseAgent
  - [ ] Q-network architecture for 64x64 RGB inputs
  - [ ] Replay buffer with data augmentation
  - [ ] Target network and soft updates
- [ ] **Evaluation 1**: Baseline DrQ-v2 performance
- [ ] **Improvement 1**: Enhanced data augmentation (random crops, color jitter)
- [ ] **Evaluation 2**: Post-improvement 1 performance
- [ ] **Improvement 2**: Prioritized replay buffer or n-step returns
- [ ] **Evaluation 3**: Final DrQ-v2 performance

### 🎯 PARTNER'S WORK: PPO (Course Algorithm)
- [ ] **PPO Agent Implementation**
  - [ ] PPOAgent class inheriting from BaseAgent
  - [ ] Refactor existing train.py into proper class structure
  - [ ] Policy and value networks for visual observations
- [ ] **Evaluation 1**: Baseline PPO performance
- [ ] **Improvement 1**: CNN architecture optimization or frame stacking
- [ ] **Evaluation 2**: Post-improvement 1 performance
- [ ] **Improvement 2**: Reward shaping or exploration bonuses
- [ ] **Evaluation 3**: Final PPO performance

### 🤝 SHARED WORK:
- [ ] **Final Comparison**: Compare DrQ-v2 vs PPO performance
- [ ] **Report Writing**: Combine both algorithm sections
- [ ] **Code Integration**: Ensure both agents work with shared evaluation code

### Evaluation Metrics (Standard Crafter Metrics)
- [ ] Achievement unlock rate (per achievement)
- [ ] Geometric mean of achievement unlock rates
- [ ] Survival time (avg timesteps per episode)
- [ ] Cumulative reward (total reward per episode)

### Deliverables
- [ ] **Report** with all requirements
- [ ] **Source Code** (well-commented, with README)
- [ ] **GitHub Repository** link in report
- [ ] **Hyperparameters** documented in report

## Technical Setup
- **Environment**: `gym.make("CrafterPartial-v1")` with Gymnasium interface
- **Libraries Allowed**: Stable Baselines3, etc.
- **YAML**: Use provided package versions for compatibility
- **Wrapper**: Minimal Gymnasium wrapper provided (handles API compatibility)

## Improvement Ideas (Non-trivial changes required)
### Potential Improvements (NOT just hyperparameter tuning)
- [ ] Image preprocessing/feature extraction
- [ ] Reward shaping
- [ ] Action space modifications
- [ ] Policy model architecture changes
- [ ] Observation preprocessing
- [ ] Sub-task curricula
- [ ] Experience replay modifications
- [ ] Multi-objective learning approaches

## Project Status

### Current Phase: Project Setup
- [x] Read and understand assignment requirements
- [x] Create project tracking system (CLAUDE.md)
- [ ] Analyze existing codebase structure
- [ ] Set up development environment
- [ ] Choose algorithms to implement

### Algorithm Selection & Ownership
- **Algorithm 1 (From Course)**: PPO (Proximal Policy Optimization)
  - **Owner**: Partner
  - **Status**: Basic version in train.py, needs BaseAgent refactoring
- **Algorithm 2 (External - NOT From Course)**: DrQ-v2 (Data-Regularized Q-Learning v2)
  - **Owner**: Anand Patel
  - **Status**: Not yet implemented, will create from scratch

### Current Project Status
**✅ COMPLETED:**
- [x] Basic project structure set up
- [x] Environment setup (Crafter + Gymnasium compatibility)
- [x] PPO implementation (basic version in train.py)
- [x] Base agent interface (src/agents/base_agent.py with TODO(human) sections)
- [x] Evaluation infrastructure (plot_reward.py, plot_scores.py, read_metrics.py)
- [x] Conda environment and dependencies
- [x] Some initial training runs (wandb logs exist)

**🚧 IN PROGRESS:**
- [ ] Need to implement full BaseAgent interface
- [ ] DrQ-v2 algorithm implementation
- [ ] Structured agent classes following BaseAgent pattern

**❌ TODO:**
- [ ] Complete PPO agent inheriting from BaseAgent
- [ ] Implement DrQ-v2 agent
- [ ] Run baseline evaluations for both agents
- [ ] Plan and implement improvements
- [ ] Final comparison and report

### Development Timeline (Updated based on current status)
- [x] **Week 1**: Environment setup + algorithm selection ✅ DONE
- [x] **Week 2**: Base implementations ✅ PPO basic version done, DrQ-v2 pending
- [ ] **Week 3**: Complete implementations + baseline evaluations
- [ ] **Week 4**: First and second improvements + evaluations
- [ ] **Week 5**: Final comparison + report writing

## Notes and Insights
- Marking focuses on **process and design decisions**, not just performance
- Results must be **reproducible**
- Use **line graphs** for presenting episode returns (not tables)
- Include **standard reward metrics** even if using reward shaping
- Code quality matters: comments, README, run scripts, versioning

## Important Links
- **Crafter GitHub**: https://github.com/danijar/crafter
- **Research Paper**: https://arxiv.org/pdf/2109.06780
- **Skeleton Code**: https://github.com/rayrsys/Reinforcement-Learning-Project-2026-Crafter.git

## Current Action Items (Updated)
### IMMEDIATE PRIORITIES:
1. **Complete BaseAgent implementations**: src/agents/base_agent.py has TODO(human) sections that need implementation
2. **Implement DrQ-v2 agent**: External algorithm not covered in course
3. **Run baseline evaluations**: Both PPO and DrQ-v2 need Eval 1 (baseline performance)

### DISCOVERED INFRASTRUCTURE:
- ✅ **Environment**: CrafterPartial-v1 properly set up with Gymnasium compatibility
- ✅ **PPO**: Basic implementation exists in train.py (uses Stable-Baselines3)
- ✅ **Evaluation**: Infrastructure exists (plotting, metrics reading)
- ✅ **Logging**: Wandb integration already working
- ⚠️ **Architecture**: Need to refactor to use BaseAgent pattern for consistency

### CLEANED PROJECT STRUCTURE:
```
crafter-rl-project/
├── src/
│   ├── agents/
│   │   ├── base_agent.py      # Shared interface
│   │   └── drqv2_agent.py     # Anand's DrQ-v2
│   ├── evaluation/            # Existing plotting tools
│   └── utils/                 # DrQ-v2 utilities (replay buffer, etc.)
├── results/                   # Training outputs
├── models/                    # Saved checkpoints
├── wandb/                     # Existing training logs
├── train.py                   # Shared training script
├── CLAUDE.md                  # This tracker
└── README.md                  # Project overview
```

**REMOVED:** Empty directories (configs/, docs/, notebooks/, tests/, experiments/)

### IMMEDIATE ACTION PLAN:
**Anand (You):**
1. Complete TODO(human) sections in BaseAgent for DrQ-v2 needs
2. Create `src/agents/drqv2_agent.py`
3. Implement replay buffer and data augmentation utilities

**Partner:**
1. Complete TODO(human) sections in train.py for PPO (currently using Stable-Baselines3)
2. Optionally create `src/agents/ppo_agent.py` for custom implementation

**Both:**
- Use shared `train.py` with `--algorithm` flag
- Training: `python train.py --algorithm ppo` or `python train.py --algorithm drqv2`
- Default: 1M steps, saves to `logdir/crafter_{algorithm}_{timestamp}/`

### EVALUATION & TESTING:
```bash
# Full evaluation (100 episodes, comprehensive analysis)
python evaluate.py --model_path models/ppo_model.zip --algorithm ppo --episodes 100

# Quick test (10 episodes, basic metrics)
python test_model.py models/ppo_model.zip ppo 10

# Analyze existing training logdir
python evaluate.py --logdir logdir/crafter_ppo_20250929_180000/ --algorithm ppo
```

**Outputs:**
- 📊 Crafter Score (geometric mean of achievements)
- 📈 Achievement unlock rates (all 22 achievements)
- 🎯 Average reward and episode length
- 📊 Plots (achievement rates, summary metrics)
- 📄 JSON + text reports

---
*Last Updated: 2025-09-29*
*Updated by: Claude*