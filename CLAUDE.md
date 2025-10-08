# Reinforcement Learning Assignment - CLAUDE Project Tracker
**COMS4061A/COMS7071A - Group Project**

## Group Information
- **Due Date**: October 22, 2025, 23:59
- **Group Size**: 2 members
- **Environment**: Crafter (Gymnasium interface)

### Group Members & Responsibilities
- **Anand Patel** (Student #: _TO_BE_FILLED_)
  - **Role**: PPO Implementation (External Algorithm)
  - **Focus**: Policy gradient methods, actor-critic, on-policy learning
- **Partner Name** (Student #: _TO_BE_FILLED_)
  - **Role**: DQN Implementation (Course Algorithm)
  - **Focus**: Model-free value-based RL, function approximation

## Assignment Overview
This project involves implementing and iteratively improving two RL agents in the Crafter environment:
1. One algorithm **from the course** (Partner: DQN)
2. One algorithm **NOT covered in the course** (Anand: PPO)

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

**Third Plan**: Dyna-Q (Model-Based Reinforcement Learning)
**Problem 3**: Struggles with visual observations, poor exploration in sparse reward settings

**Final Plan**: **PPO (Proximal Policy Optimization)** ✅
**Why this is perfect**:
1. **Completely different paradigm**: Partner does value-based (DQN), Anand does policy gradient (PPO)
2. **Definitely external**: Confirmed not in course materials (checked slides)
3. **Perfect for Crafter**: Natural exploration via stochastic policies, excellent with sparse rewards
4. **Industry standard**: Used in OpenAI Five, ChatGPT RLHF, robotics - strong practical relevance
5. **Appropriate complexity**: ~400 lines, well-documented, strong theoretical foundation
6. **Superior performance**: Expected 15-25% Crafter Score vs 0.5-2% for DQN

---

## Algorithm Comparison

### Value-Based vs Policy Gradient Methods

| Aspect | DQN (Partner - Value-Based) | PPO (Anand - Policy Gradient) |
|--------|---------------------------|------------------------------|
| **Learning Style** | Learn Q-values, derive policy | Learn policy directly |
| **Action Selection** | Deterministic (ε-greedy) | Stochastic (probability distribution) |
| **Exploration** | External (ε-greedy) | Natural (policy entropy) |
| **Sample Efficiency** | Off-policy (replay buffer) | On-policy (recent experience) |
| **Stability** | Can diverge (deadly triad) | Stable (clipped updates) |
| **Strengths** | Sample efficient, simple | Stable, natural exploration, handles continuous actions |
| **Weaknesses** | Poor exploration, Q-value overestimation | Requires more samples than off-policy methods |
| **Performance on Crafter** | ~0.5-2% (struggles with exploration) | ~15-25% (explores effectively) |

### Why PPO for Crafter?

**Crafter's Challenge**: Sparse rewards requiring diverse exploration
- Achievements require multi-step sequences (e.g., chop tree → get wood → make stick → make sword)
- Random exploration (ε-greedy) rarely discovers these sequences
- Need directed exploration that naturally tries different strategies

**PPO's Solution**: Policy gradient with controlled updates
1. **Stochastic Policy**: Agent outputs action probabilities (e.g., 40% chop, 30% move, 30% attack)
2. **Natural Exploration**: Policy inherently explores - tries different actions based on learned probabilities
3. **Clipped Updates**: Prevents catastrophic policy changes (key innovation over vanilla policy gradient)
4. **Actor-Critic**: Value function (critic) helps estimate advantage, policy (actor) improves based on advantage

**PPO Algorithm Core**:
```
1. Collect trajectories using current policy π_old
2. For each state-action pair (s, a):
   - Compute advantage: A(s,a) = Q(s,a) - V(s)  [how much better than average]
   - Compute ratio: r(θ) = π_new(a|s) / π_old(a|s)  [policy change]
   - Compute clipped objective: min(r·A, clip(r, 1-ε, 1+ε)·A)  [limit change]
3. Update policy to maximize clipped objective
4. Update value function to minimize prediction error
```

**Expected Impact**:
- Value-based DQN: Gets stuck in exploitation, poor exploration → low achievement rate
- Policy gradient PPO: Naturally explores, stable learning → high achievement rate
- **8-15× better performance** on sparse reward tasks (15-25% vs 0.5-2%)

---

## 🎯 ANAND'S WORK: PPO (External Algorithm)

### Phase 1: Baseline PPO (Eval 1) 📋 IN PROGRESS
**Goal**: Implement vanilla PPO and establish strong baseline

**Components**:
- [ ] **Actor Network**: CNN → FC → Policy head (action probabilities)
- [ ] **Critic Network**: CNN → FC → Value head (state value V(s))
- [ ] **Trajectory Collection**: Rollout buffer storing (s, a, r, log_prob, value)
- [ ] **Advantage Estimation**: GAE (Generalized Advantage Estimation) λ=0.95
- [ ] **PPO Update**: Clipped objective with multiple epochs (10 epochs per batch)
- [ ] **Training**: 1M environment steps, batch size 2048 steps
- [ ] **Eval 1 Target**: 5-10% Crafter Score (solid baseline, better than DQN)

**Key Parameters**:
```python
# Policy & Value Networks
lr_actor = 3e-4             # Policy learning rate
lr_critic = 1e-3            # Value learning rate
gamma = 0.99                # Discount factor
gae_lambda = 0.95           # GAE parameter (bias-variance tradeoff)

# PPO Specific
clip_epsilon = 0.2          # Clipping parameter (limits policy change)
value_clip = 0.2            # Value function clipping
entropy_coef = 0.01         # Entropy bonus (encourages exploration)
vf_coef = 0.5               # Value function loss weight

# Training
n_steps = 2048              # Steps per rollout
batch_size = 64             # Minibatch size for updates
n_epochs = 10               # Update epochs per rollout
max_grad_norm = 0.5         # Gradient clipping
```

**Architecture**:
```python
# Shared CNN Feature Extractor
Conv2d(3, 32, 3, stride=2) → ReLU
Conv2d(32, 64, 3, stride=2) → ReLU
Conv2d(64, 64, 3, stride=2) → ReLU
Flatten → FC(2048)

# Actor Head (Policy)
FC(2048) → ReLU → FC(17 actions) → Softmax

# Critic Head (Value)
FC(2048) → ReLU → FC(1 value) → Linear
```

---

### Phase 2: Adaptive Clipping + Entropy Scheduling (Improvement 1 → Eval 2) 📋 PLANNED
**Goal**: Improve exploration-exploitation balance dynamically

**Problem with fixed clipping**: Early training needs exploration (high entropy), late training needs exploitation (low entropy)
**Solution**: Adaptive mechanisms that adjust exploration over time

**Implementation**:
- [ ] **Entropy Decay**: Start with high entropy bonus (0.05), decay to 0.001 over training
- [ ] **Adaptive Clipping**: Adjust clip_epsilon based on KL divergence (if KL too high, reduce clip)
- [ ] **Early Stopping**: Stop updates if KL divergence exceeds threshold (prevents destructive updates)
- [ ] **Training**: 1M steps, adaptive parameters
- [ ] **Eval 2 Target**: 12-18% Crafter Score (better exploration → more achievements)

**Expected Improvement**:
- Baseline: Fixed exploration → either under-explores or over-explores
- Adaptive: High exploration early (discover achievements) → low exploration late (exploit best strategies)
- **1.5-2× improvement** in achievement diversity and score

**Mechanism**:
```python
# Entropy scheduling
entropy_coef = max(0.001, 0.05 * (1 - step/total_steps))

# Adaptive clipping
kl_div = compute_kl(old_policy, new_policy)
if kl_div > target_kl:
    clip_epsilon *= 0.5  # Reduce clipping if policy changing too fast
else:
    clip_epsilon = min(0.2, clip_epsilon * 1.05)  # Increase if too conservative
```

---

### Phase 3: Recurrent Policy (LSTM) (Improvement 2 → Eval 3) 📋 PLANNED
**Goal**: Add memory to handle partial observability in Crafter

**Problem**: Crafter is partially observable - agent can't see full map, needs memory for planning
**Solution**: Replace feedforward policy with LSTM - agent remembers past observations

**Implementation**:
- [ ] **LSTM Layer**: Add LSTM(256 hidden) after CNN features
- [ ] **Hidden State Management**: Track hidden states in rollout buffer
- [ ] **Sequence Training**: Update LSTM using trajectory sequences (not individual transitions)
- [ ] **Training**: 1M steps, LSTM-based policy
- [ ] **Eval 3 Target**: 18-25% Crafter Score (memory enables multi-step planning)

**Expected Improvement**:
- Feedforward: No memory → can't plan sequences (e.g., "I saw a tree 10 steps ago, go back to chop it")
- Recurrent: Memory → remembers map layout, plans multi-step strategies
- **1.5-2× improvement** in long-horizon task completion

**Architecture**:
```python
# Recurrent Feature Extractor
Conv2d layers (same as before) → Flatten
LSTM(2048 input, 256 hidden, 2 layers)

# Actor/Critic heads use LSTM output
LSTM output → Actor head (policy)
LSTM output → Critic head (value)
```

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

- [ ] **Final Comparison**: DQN (value-based) vs PPO (policy gradient)
- [ ] **Report Writing**:
  - Emphasize paradigm difference (value-based vs policy gradient)
  - Show learning curves and achievement rates
  - Discuss exploration strategies (ε-greedy vs stochastic policies)
  - Analyze stability (DQN divergence vs PPO clipping)
- [ ] **Code Integration**: Unified training/evaluation pipeline

---

## Technical Implementation

### Project Structure
```
crafter-rl-project/
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # Abstract interface
│   │   └── ppo_agent.py           # PPO implementation (Anand)
│   ├── models/
│   │   ├── actor_critic.py        # Actor-Critic network
│   │   └── recurrent_policy.py    # LSTM-based policy (Improvement 2)
│   ├── utils/
│   │   ├── networks.py            # Shared CNN architectures
│   │   ├── rollout_buffer.py      # On-policy trajectory storage
│   │   └── gae.py                 # Generalized Advantage Estimation
│   └── evaluation/
│       └── ...                    # Existing evaluation code
├── models/
│   ├── ppo_baseline.pt            # Eval 1 (5-10%)
│   ├── ppo_adaptive.pt            # Eval 2 (12-18%)
│   └── ppo_lstm.pt                # Eval 3 (18-25%)
├── results/
│   ├── eval1_baseline/
│   ├── eval2_adaptive/
│   └── eval3_recurrent/
├── CLAUDE.md                       # This file
├── PPO_EXPLANATION.md              # Detailed PPO algorithm explanation
└── README.md
```

### Training Commands

```bash
# Baseline PPO (Eval 1)
python train_ppo.py --steps 1000000 --n_steps 2048 --n_epochs 10

# Adaptive PPO (Eval 2)
python train_ppo.py --steps 1000000 --adaptive --entropy_decay

# Recurrent PPO (Eval 3)
python train_ppo.py --steps 1000000 --recurrent --lstm_hidden 256
```

### SLURM Training (for cluster)

```bash
# Submit baseline training
sbatch train_ppo.sbatch

# Monitor job
squeue -u $USER
watch -n 1 squeue -u $USER

# Check logs
tail -f slurm-<job_id>.out
```

### Evaluation Commands

```bash
# Comprehensive evaluation (100 episodes)
python evaluate.py \
    --model_path models/ppo_lstm.pt \
    --algorithm ppo \
    --episodes 100 \
    --outdir results/eval3_recurrent

# Quick test (10 episodes)
python test_model.py models/ppo_lstm.pt ppo 10
```

---

## Hyperparameters

### PPO Core
- **Actor learning rate**: 3e-4
- **Critic learning rate**: 1e-3
- **Gamma (discount)**: 0.99
- **GAE lambda**: 0.95
- **Clip epsilon**: 0.2
- **Value clip**: 0.2
- **Entropy coefficient**: 0.01 (baseline), 0.05→0.001 (adaptive)
- **Value function coefficient**: 0.5
- **Max gradient norm**: 0.5

### Training
- **Rollout steps**: 2048 (steps per update)
- **Minibatch size**: 64
- **Update epochs**: 10 (multiple passes over data)
- **Total steps**: 1M environment steps
- **Normalize advantages**: True
- **Normalize observations**: True (running mean/std)

### Architecture (Baseline)
- **CNN channels**: [3, 32, 64, 64]
- **Kernel size**: 3×3, stride 2
- **Feature dimension**: 2048
- **Actor hidden**: [2048, 512, 17]
- **Critic hidden**: [2048, 512, 1]

### Architecture (Recurrent)
- **LSTM hidden**: 256
- **LSTM layers**: 2
- **Sequence length**: 16 (for BPTT)

---

## Expected Results

| Evaluation | Algorithm | Target Score | Key Feature |
|------------|-----------|--------------|-------------|
| **Eval 1** | Baseline PPO | 5-10% | Natural exploration via stochastic policy |
| **Eval 2** | + Adaptive Exploration | 12-18% | Entropy decay + adaptive clipping |
| **Eval 3** | + Recurrent Policy (LSTM) | 18-25% | Memory for partial observability |

### Comparison with Partner's DQN
- **DQN (value-based)**: ~0.5-2% (poor exploration, ε-greedy gets stuck)
- **PPO (policy gradient)**: 18-25% (natural exploration, stable learning)
- **Key insight**: Stochastic policies explore effectively in sparse reward settings
- **Performance gap**: ~10-20× better achievement rate

### Performance Analysis
**Achievement Breakdown** (expected for Eval 3):
- **Basic survival**: Collect wood, stone, coal (>90% success)
- **Crafting**: Make tools (wood pickaxe, stone pickaxe) (>70% success)
- **Combat**: Defeat zombies, skeletons (>50% success)
- **Advanced**: Make iron pickaxe, diamond (>30% success)

**Learning Speed**:
- **DQN**: Slow initial learning (random until lucky), plateaus early
- **PPO**: Fast initial learning (explores systematically), continues improving

---

## Timeline & Progress

### Week 1 (Sept 25-30): Environment Setup ✅
- [x] Environment setup (Crafter + dependencies)
- [x] Basic DQN baseline (for partner to use)

### Week 2 (Oct 1-6): Algorithm Pivot & Planning 🚧
- [x] Pivoted from DrQ-v2 → Curious DQN → Dyna-Q → PPO
- [x] Confirmed PPO not in course materials
- [x] Cleaned up old code
- [x] Comprehensive PPO implementation plan created

### Week 3 (Oct 7-13): Baseline + Improvement 1 📋 CURRENT
- [ ] **Day 1-2** (Oct 7-8): Implement baseline PPO (Actor-Critic, GAE, clipped objective)
- [ ] **Day 3** (Oct 9): Train baseline (Eval 1) - submit to cluster
- [ ] **Day 4-5** (Oct 10-11): Implement adaptive exploration (entropy decay, adaptive clipping)
- [ ] **Day 6** (Oct 12): Train adaptive (Eval 2) - submit to cluster
- [ ] **Day 7** (Oct 13): Analyze Eval 1 & 2 results

### Week 4 (Oct 14-22): Improvement 2 + Report 📋
- [ ] **Day 8-10** (Oct 14-16): Implement recurrent PPO (LSTM policy)
- [ ] **Day 11** (Oct 17): Train recurrent (Eval 3) - submit to cluster
- [ ] **Day 12-13** (Oct 18-19): Generate comparison plots, analyze results
- [ ] **Day 14-17** (Oct 20-21): Write report, create figures
- [ ] **Day 18** (Oct 22): Final report submission

---

## Key Insights to Highlight in Report

### Why PPO is "External"
1. **Not in course materials**: Confirmed by checking course slides - no policy gradient methods covered
2. **Published 2017**: Schulman et al. (OpenAI) - post-dates traditional RL courses
3. **Different paradigm**: Policy gradient vs value-based (fundamentally different approach from DQN)
4. **Learning value**: Understand actor-critic, importance sampling, trust region optimization
5. **Industry relevance**: Used in ChatGPT RLHF, robotics, game AI

### The Policy Gradient Advantage
**Sparse Rewards Problem**:
- Crafter: Achievement rewards are rare, require exploration
- Value-based (DQN): ε-greedy exploration is random → rarely discovers achievement sequences
- Policy gradient (PPO): Stochastic policy naturally explores → systematically tries different strategies

**Concrete Example**:
```
Situation: Agent near tree, needs wood

DQN (ε-greedy):
  - 95% of time: Take best Q-value action (might be "move away" if never experienced "chop")
  - 5% of time: Random action (uniform over 17 actions)
  - Result: Rarely tries "chop" → never learns tree → wood

PPO (stochastic policy):
  - Policy outputs probabilities: [chop: 0.3, move_left: 0.2, move_right: 0.2, ...]
  - Samples action from distribution: tries "chop" 30% of time
  - Discovers: chop → wood → reward → increases P(chop|near_tree) to 0.8
  - Result: Natural exploration → discovers achievement → reinforces successful behavior
```

### Tradeoffs
**PPO Advantages**:
- **Natural exploration**: Stochastic policy inherently explores
- **Stability**: Clipped objective prevents catastrophic updates
- **Continuous actions**: Can handle continuous action spaces (not relevant for Crafter, but general advantage)
- **Better performance**: 10-20× higher achievement rate than DQN on Crafter

**PPO Disadvantages**:
- **On-policy**: Must collect new data after each update (can't use replay buffer)
- **Sample efficiency**: Requires more environment steps than off-policy methods (but still learns faster on Crafter due to better exploration)
- **Hyperparameter sensitivity**: More hyperparameters to tune (clip epsilon, entropy coef, GAE lambda)

### Value-Based vs Policy Gradient: The Fundamental Difference

| Aspect | Value-Based (DQN) | Policy Gradient (PPO) |
|--------|-------------------|----------------------|
| **What it learns** | Q(s,a) - expected return | π(a\|s) - action probabilities |
| **Policy derivation** | Implicit (argmax Q) | Explicit (learned directly) |
| **Exploration** | External (ε-greedy) | Natural (policy entropy) |
| **Action selection** | Deterministic (greedy) | Stochastic (sample from π) |
| **Update rule** | TD error | Policy gradient + clipping |
| **Best for** | Sample efficiency | Exploration, stability |

---

## Deliverables Checklist

- [ ] **Source Code**:
  - [ ] `src/agents/ppo_agent.py` - Main PPO implementation
  - [ ] `src/models/actor_critic.py` - Actor-Critic networks
  - [ ] `src/models/recurrent_policy.py` - LSTM-based policy (Improvement 2)
  - [ ] `src/utils/rollout_buffer.py` - Trajectory storage
  - [ ] `src/utils/gae.py` - Generalized Advantage Estimation
  - [ ] `train_ppo.py` - Training script
  - [ ] `train_ppo.sbatch` - SLURM job script
  - [ ] Clean, well-commented code with docstrings
- [ ] **Models**: All 3 evaluation checkpoints (ppo_baseline.pt, ppo_adaptive.pt, ppo_lstm.pt)
- [ ] **Results**: Comprehensive evaluation data (100 episodes each evaluation)
- [ ] **Report**:
  - [ ] Algorithm explanation (PPO, policy gradients, actor-critic)
  - [ ] Implementation details (architecture, hyperparameters)
  - [ ] Results & comparisons (DQN vs PPO)
  - [ ] Exploration analysis (ε-greedy vs stochastic policies)
  - [ ] Insights (value-based vs policy gradient tradeoffs)
  - [ ] Learning curves and achievement breakdowns
- [ ] **Documentation**:
  - [ ] `PPO_EXPLANATION.md` - Detailed PPO algorithm explanation
  - [ ] `README.md` - Updated with PPO instructions
- [ ] **GitHub Repository**: Public repo with all code

---

## References

### Papers
1. **PPO**: Schulman et al. 2017 - Proximal Policy Optimization Algorithms - https://arxiv.org/abs/1707.06347
2. **GAE**: Schulman et al. 2016 - High-Dimensional Continuous Control Using Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
3. **TRPO**: Schulman et al. 2015 - Trust Region Policy Optimization - https://arxiv.org/abs/1502.05477
4. **Policy Gradients**: Sutton et al. 2000 - Policy Gradient Methods for Reinforcement Learning with Function Approximation
5. **Actor-Critic**: Konda & Tsitsiklis 2000 - Actor-Critic Algorithms
6. **Crafter Benchmark**: Hafner 2021 - Benchmarking the Spectrum of Agent Capabilities - https://arxiv.org/abs/2109.06780

### Textbook
- **Sutton & Barto (2018)**: Reinforcement Learning: An Introduction (2nd Edition)
  - Chapter 13: Policy Gradient Methods

### Code References
- **Crafter**: https://github.com/danijar/crafter
- **OpenAI Baselines PPO**: https://github.com/openai/baselines
- **Stable-Baselines3 PPO**: https://github.com/DLR-RM/stable-baselines3
- **CleanRL PPO**: https://github.com/vwxyzjn/cleanrl

---

## Current Status

**Phase**: PPO implementation ready to begin
**Algorithm**: PPO (Proximal Policy Optimization) - Confirmed external to course
**Next Steps**:
1. Implement Actor-Critic networks (CNN + policy/value heads)
2. Implement rollout buffer and GAE
3. Implement PPO update loop (clipped objective)
4. Create training script and SLURM job
5. Run baseline training (Eval 1)

**Files Cleaned Up**:
- ✅ Removed DrQ-v2 code
- ✅ Removed Curious DQN implementation
- ✅ Removed Dyna-Q planning code
- ✅ Removed ICM module
- ✅ Updated all documentation for PPO
- ✅ Ready for PPO implementation

**Why PPO is the Right Choice**:
- ✅ Confirmed not in course materials (policy gradient methods not covered)
- ✅ Completely different paradigm from partner's DQN (policy gradient vs value-based)
- ✅ Industry standard (OpenAI Five, ChatGPT RLHF, robotics)
- ✅ Excellent performance on Crafter (15-25% expected vs 0.5-2% for DQN)
- ✅ Appropriate complexity (~400 lines, well-documented)
- ✅ Strong learning value (actor-critic, importance sampling, trust regions)

---

*Last Updated: October 8, 2025*
*Updated by: Claude (with Anand's guidance)*
*Status: Ready to begin PPO implementation - LET'S GO!*
