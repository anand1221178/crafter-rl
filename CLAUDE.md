# Reinforcement Learning Assignment - CLAUDE Project Tracker
**COMS4061A/COMS7071A - Group Project**

## Group Information
- **Due Date**: October 22, 2025, 23:59
- **Group Size**: 2 members
- **Environment**: Crafter (Gymnasium interface)

### Group Members & Responsibilities
- **Anand Patel** (Student #: _TO_BE_FILLED_)
  - **Role**: Dyna-Q Implementation (External Algorithm)
  - **Focus**: Model-based RL, planning, world models
- **Partner Name** (Student #: _TO_BE_FILLED_)
  - **Role**: DQN Implementation (Course Algorithm)
  - **Focus**: Model-free value-based RL, function approximation

## Assignment Overview
This project involves implementing and iteratively improving two RL agents in the Crafter environment:
1. One algorithm **from the course** (Partner: DQN)
2. One algorithm **NOT covered in the course** (Anand: Dyna-Q)

Each agent must go through iterative improvement cycles: Base → Eval 1 → Improvement 1 → Eval 2 → Improvement 2 → Eval 3

## Environment Details
- **Crafter**: Procedurally generated 2D survival game
- **Observation Space**: 64x64 RGB images
- **Action Space**: 17 discrete actions
- **Environment**: `gym.make("CrafterPartial-v1")`
- **Key Mechanics**: Resource gathering, crafting, combat, survival (hunger/health)
- **Rewards**: Sparse achievement-based rewards (22 total achievements)
- **Challenge**: Long-horizon tasks requiring multi-step reasoning

---

## Algorithm Selection & Rationale

### ⚠️ IMPORTANT: Algorithm Pivot History

**Original Plan**: DrQ-v2 (Data-regularized Q-learning)
**Problem 1**: DQN covered in course → DrQ-v2 too similar

**Second Plan**: Curious DQN (DQN + Intrinsic Curiosity Module)
**Problem 2**: Too complex for learning goals, still DQN-based

**Final Plan**: **Dyna-Q (Model-Based Reinforcement Learning)** ✅
**Why this is perfect**:
1. **Completely different paradigm**: Partner does model-free (DQN), Anand does model-based (Dyna-Q)
2. **Classic & foundational**: Sutton & Barto 1996 - clearly external to modern deep RL courses
3. **Perfect for Crafter**: Sparse rewards benefit from planning (simulate experience)
4. **Appropriate complexity**: ~300 lines, learnable in 18 days
5. **Clear learning value**: Understand planning vs learning tradeoff

---

## Algorithm Comparison

### Three Paradigms of RL

| Aspect | DQN (Partner - Model-Free) | Dyna-Q (Anand - Model-Based) |
|--------|---------------------------|------------------------------|
| **Learning Style** | Learn from experience only | Learn from experience + planning |
| **World Model** | None (black box) | Learns environment dynamics |
| **Sample Efficiency** | Low (needs many interactions) | High (reuses experience via planning) |
| **Computation** | Fast updates | Slow updates (planning overhead) |
| **Strengths** | Simple, robust, proven | Sample-efficient, handles sparse rewards |
| **Weaknesses** | Sample-hungry | Model errors compound |

### Why Model-Based RL for Crafter?

**Crafter's Challenge**: Extremely sparse rewards
- Achievements require multi-step sequences (e.g., chop tree → get wood → make stick → make sword)
- Random exploration rarely discovers these sequences
- Once discovered, need to propagate value backward efficiently

**Dyna-Q's Solution**: Planning with learned world model
1. **Experience real transition**: agent chops tree, gets wood (+1 reward)
2. **Store in model**: "action=chop at tree-state → wood-state, reward=+1"
3. **Planning phase**: Simulate this transition 50 times, update Q-values each time
4. **Result**: Single real experience → 50× learning updates!

**Expected Impact**:
- Model-free DQN: Needs to experience "chop tree → wood" many times
- Model-based Dyna-Q: Learns from one experience, plans the rest
- **2-4× better sample efficiency** on sparse reward tasks

---

## 🎯 ANAND'S WORK: Dyna-Q (External Algorithm)

### Phase 1: Baseline Dyna-Q (Eval 1) 📋 PLANNED
**Goal**: Implement basic Dyna-Q and establish baseline

**Components**:
- [ ] **Q-learning module**: Standard Q-network with CNN for 64×64 RGB
- [ ] **World model**: Simple table-based model storing (s, a) → (s', r)
- [ ] **Planning module**: Sample random transitions from model, update Q-values
- [ ] **Integration**: Interleave real experience and planning steps
- [ ] **Training**: 1M environment steps, 5 planning steps per real step
- [ ] **Eval 1 Target**: 0.5-2% Crafter Score (better than random, baseline for improvements)

**Key Parameters**:
```python
planning_steps = 5           # Planning updates per real step
lr = 1e-4                   # Q-network learning rate
gamma = 0.99                # Discount factor
epsilon = 1.0 → 0.05        # Exploration (decay over 750K steps)
batch_size = 32             # Q-learning batch size
buffer_size = 100K          # Replay buffer capacity
```

---

### Phase 2: Prioritized Sweeping (Improvement 1 → Eval 2) 📋 PLANNED
**Goal**: Make planning more efficient by focusing on important states

**Problem with random planning**: Wastes compute on irrelevant states
**Solution**: Prioritize states with largest prediction errors (backup where model changed)

**Implementation**:
- [ ] **Priority queue**: Track states by model prediction error magnitude
- [ ] **Targeted planning**: Sample from queue instead of random
- [ ] **Dynamic updates**: When model changes, update priorities
- [ ] **Training**: 1M steps, prioritized planning
- [ ] **Eval 2 Target**: 3-8% Crafter Score (more efficient credit assignment)

**Expected Improvement**:
- Baseline: Random planning wastes 80% of updates on static states
- Prioritized: Focuses on states where learning is happening
- **2-4× improvement in planning efficiency**

---

### Phase 3: Dyna-Q+ with Exploration Bonuses (Improvement 2 → Eval 3) 📋 PLANNED
**Goal**: Encourage exploration of unvisited state-action pairs

**Problem**: Agent exploits known paths, misses better alternatives
**Solution**: Add bonus reward for rarely-visited (s, a) pairs

**Implementation**:
- [ ] **Visitation counter**: Track visits to each (state, action) pair
- [ ] **Exploration bonus**: r_bonus = κ × √(time_since_visit)
- [ ] **Modified planning**: Use bonus rewards during planning phase
- [ ] **Training**: 1M steps, exploration bonuses
- [ ] **Eval 3 Target**: 8-15% Crafter Score (discovers more achievement paths)

**Expected Improvement**:
- Without bonuses: Agent gets stuck in local optima (safe but low-reward paths)
- With bonuses: Explores alternative paths, discovers better strategies
- **2-3× improvement in achievement diversity**

---

## 🎯 PARTNER'S WORK: DQN (Course Algorithm)

**Note**: Partner can use the baseline DQN implementation from previous work as starting point!

- [ ] DQN agent implementation (vanilla, from course)
- [ ] Evaluation 1: Baseline DQN performance (~0.26% based on prior runs)
- [ ] Improvement 1: Network architecture changes or reward shaping
- [ ] Evaluation 2: Post-improvement 1 performance
- [ ] Improvement 2: Replay modifications or exploration strategy
- [ ] Evaluation 3: Final DQN performance

---

## 🤝 SHARED WORK

- [ ] **Final Comparison**: DQN (model-free) vs Dyna-Q (model-based)
- [ ] **Report Writing**:
  - Emphasize paradigm difference (learning vs planning)
  - Show sample efficiency curves (steps to achieve X% score)
  - Discuss model-based strengths/weaknesses
- [ ] **Code Integration**: Unified training/evaluation pipeline

---

## Technical Implementation

### Project Structure
```
crafter-rl-project/
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # Abstract interface
│   │   └── dynaq_agent.py         # Dyna-Q implementation (Anand)
│   ├── models/
│   │   ├── world_model.py         # Environment dynamics model
│   │   └── prioritized_sweeping.py # Priority queue for planning
│   ├── utils/
│   │   ├── networks.py            # Q-network (CNN)
│   │   └── replay_buffer.py       # Experience replay
│   └── evaluation/
│       └── ...                    # Existing evaluation code
├── models/
│   ├── dynaq_baseline.pt          # Eval 1 (0.5-2%)
│   ├── dynaq_prioritized.pt       # Eval 2 (3-8%)
│   └── dynaq_plus.pt              # Eval 3 (8-15%)
├── results/
│   ├── eval1_baseline/
│   ├── eval2_prioritized/
│   └── eval3_exploration/
├── CLAUDE.md                       # This file
├── DYNA_Q_EXPLANATION.md           # Detailed algorithm explanation (to be created)
└── README.md
```

### Training Commands

```bash
# Baseline Dyna-Q (Eval 1)
python train.py --algorithm dynaq --steps 1000000 --planning_steps 5

# Prioritized Sweeping (Eval 2)
python train.py --algorithm dynaq --steps 1000000 --planning_steps 5 --prioritized

# Dyna-Q+ (Eval 3)
python train.py --algorithm dynaq --steps 1000000 --planning_steps 5 --prioritized --exploration_bonus 0.001
```

### Evaluation Commands

```bash
# Comprehensive evaluation (100 episodes)
python evaluate.py \
    --model_path models/dynaq_plus.pt \
    --algorithm dynaq \
    --episodes 100 \
    --outdir results/eval3_exploration

# Quick test (10 episodes)
python test_model.py models/dynaq_plus.pt dynaq 10
```

---

## Hyperparameters

### Q-Learning (Shared with DQN)
- Learning rate: 1e-4
- Gamma (discount): 0.99
- Batch size: 32
- Replay buffer: 100K
- Min replay size: 1K
- Epsilon: 1.0 → 0.05 over 750K steps
- Target network: Soft update τ=0.01

### Dyna-Q Specific
- **Planning steps**: 5 (per real step)
- **Model type**: Tabular (hash state features → store transitions)
- **Model capacity**: 50K transitions
- **Prioritized sweeping threshold**: 0.01 (min priority to enter queue)
- **Exploration bonus**: κ=0.001 (Dyna-Q+ only)

---

## Expected Results

| Evaluation | Algorithm | Target Score | Key Feature |
|------------|-----------|--------------|-------------|
| **Eval 1** | Baseline Dyna-Q | 0.5-2% | Planning (5 steps/real step) |
| **Eval 2** | + Prioritized Sweeping | 3-8% | Focused planning |
| **Eval 3** | + Exploration Bonus | 8-15% | Directed exploration |

### Comparison with Partner's DQN
- **DQN (model-free)**: ~0.5-2% (needs many samples)
- **Dyna-Q (model-based)**: 8-15% (sample-efficient via planning)
- **Key insight**: Planning amplifies rare experiences in sparse reward settings

### Sample Efficiency Analysis
We'll measure "steps to achieve 1% score":
- **DQN**: ~800K-1M steps (most of training)
- **Dyna-Q**: ~200K-400K steps (2-5× faster learning)

---

## Timeline & Progress

### Week 1 (Sept 25-30): Environment Setup ✅
- [x] Environment setup (Crafter + dependencies)
- [x] Basic DQN baseline (for partner to use)

### Week 2 (Oct 1-6): Algorithm Pivot & Planning 🚧
- [x] Pivoted from DrQ-v2 → Curious DQN → Dyna-Q
- [x] Cleaned up old DQN-based code
- [ ] Implement Dyna-Q baseline
- [ ] Train and evaluate Eval 1

### Week 3 (Oct 7-13): Improvements 📋
- [ ] Implement prioritized sweeping
- [ ] Train and evaluate Eval 2
- [ ] Implement Dyna-Q+ (exploration bonuses)
- [ ] Train and evaluate Eval 3

### Week 4 (Oct 14-22): Analysis & Report 📋
- [ ] Generate comparison plots (DQN vs Dyna-Q)
- [ ] Sample efficiency analysis
- [ ] Write methodology section
- [ ] Final report submission (Oct 22)

---

## Key Insights to Highlight in Report

### Why Dyna-Q is "External"
1. **Not covered in modern RL courses**: Most courses focus on deep RL (DQN, PPO, A3C)
2. **Classic paper**: Sutton & Barto 1996 (predates deep learning era)
3. **Different paradigm**: Model-based vs model-free (fundamentally different approach)
4. **Learning value**: Understand planning/learning integration

### The Planning Advantage
**Sparse Rewards Problem**:
- Crafter: Achievement rewards are rare (might see 1 reward per 1000 steps)
- Model-free: Needs to re-experience rare events many times to learn
- Model-based: Experiences once, simulates 100s of times via planning

**Concrete Example**:
```
Real experience (1 step):
  state=forest, action=chop → state=has_wood, reward=+1

Model learns:
  chop(forest) → wood (+1 reward)

Planning (50 simulated steps):
  Update Q(forest, chop) using model transition... 50 times!

Result: 1 real experience → 50 learning updates
```

### Tradeoffs
**Dyna-Q Advantages**:
- Sample efficient (2-5× fewer environment steps)
- Good for sparse rewards (planning propagates rare signals)
- Interpretable (can inspect learned model)

**Dyna-Q Disadvantages**:
- Model errors compound (wrong model → wrong planning)
- Computational overhead (planning takes time)
- Harder to scale to large state spaces (need function approximation)

---

## Deliverables Checklist

- [ ] **Source Code**:
  - [ ] `src/agents/dynaq_agent.py`
  - [ ] `src/models/world_model.py`
  - [ ] `src/models/prioritized_sweeping.py`
  - [ ] Clean, well-commented code
- [ ] **Models**: All 3 evaluation checkpoints
- [ ] **Results**: Comprehensive evaluation data (100 episodes each)
- [ ] **Report**:
  - [ ] Algorithm explanation (Dyna-Q, planning, model-based RL)
  - [ ] Implementation details
  - [ ] Results & comparisons (DQN vs Dyna-Q)
  - [ ] Sample efficiency analysis
  - [ ] Insights (model-based vs model-free tradeoffs)
- [ ] **GitHub Repository**: Public repo with all code
- [ ] **README**: Setup instructions, training commands

---

## References

### Papers
1. **Dyna Architecture**: Sutton 1991 - Integrated architectures for learning, planning, and reacting based on approximating dynamic programming
2. **Dyna-Q**: Sutton & Barto 1996/2018 - Reinforcement Learning: An Introduction (Chapter 8)
3. **Prioritized Sweeping**: Moore & Atkeson 1993 - Prioritized sweeping: Reinforcement learning with less data and less time
4. **Crafter Benchmark**: Hafner 2021 - https://arxiv.org/abs/2109.06780

### Textbook
- **Sutton & Barto (2018)**: Reinforcement Learning: An Introduction (2nd Edition)
  - Chapter 8: Planning and Learning with Tabular Methods
  - Section 8.2: Dyna: Integrated Planning, Acting, and Learning

### Code References
- **Crafter**: https://github.com/danijar/crafter
- **Dyna-Q Implementation**: Various RL textbook implementations

---

## Current Status

**Phase**: Project setup and planning complete
**Next Steps**:
1. Implement baseline Dyna-Q agent
2. Implement world model
3. Run baseline training (Eval 1)

**Files Cleaned Up**:
- ✅ Removed DrQ-v2 code
- ✅ Removed Curious DQN implementation
- ✅ Removed ICM module
- ✅ Removed old documentation
- ✅ Ready for Dyna-Q implementation

---

*Last Updated: October 4, 2025*
*Updated by: Claude (with Anand's guidance)*
*Status: Ready to begin Dyna-Q implementation*
