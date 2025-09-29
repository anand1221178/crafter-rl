# DrQ-v2 Implementation Plan - Anand's Work
**Data-Regularized Q-Learning v2 for Crafter Environment**

## Algorithm Overview
DrQ-v2 is a model-free RL algorithm specifically designed for visual control tasks. It combines DQN with data augmentation to achieve state-of-the-art sample efficiency on pixel-based environments.

**Key Innovations:**
- **Data Augmentation**: Random crops and color jittering during training
- **Improved Architecture**: Better CNN encoders for visual observations
- **Target Networks**: Soft updates for stability
- **Sample Efficiency**: Achieves strong performance with limited data

## Iterative Development Strategy

### 📊 BASELINE - EVALUATION 1
**Goal**: Establish baseline performance with barebones DQN

#### Implementation Scope:
```python
# Core Components (Phase 1)
DrQv2Agent(BaseAgent)           # Main agent inheriting from base
├── QNetwork                    # CNN: 64x64 RGB → Q-values (17 actions)
├── ReplayBuffer               # Basic circular buffer (100k transitions)
├── act()                      # Epsilon-greedy action selection
├── store_experience()         # Store (s, a, r, s', done) tuples
└── update()                   # Standard Q-learning update
```

#### Technical Details:
- **Network Architecture**: 3-layer CNN + 2-layer MLP
- **Learning Rate**: 3e-4
- **Batch Size**: 32
- **Epsilon**: 1.0 → 0.01 over 100k steps
- **Target Update**: Hard update every 1000 steps
- **Replay Buffer**: 100k transitions, uniform sampling

#### Expected Performance:
- **Crafter Score**: 0.5-1.0% (basic DQN baseline)
- **Achievements**: collect_wood, place_table (easiest ones)
- **Episode Length**: ~50-100 steps
- **Training**: Slow initial learning, may struggle with visual complexity

---

### 🚀 IMPROVEMENT 1 - EVALUATION 2
**Goal**: Add core DrQ-v2 features (data augmentation + architecture)

#### New Components:
```python
# Data Augmentation Module
DataAugmentation
├── RandomCrop(84 → 64)         # Spatial augmentation
├── ColorJitter                 # Brightness/contrast changes
└── augment_batch()             # Apply during training only

# Improved Architecture
QNetworkV2                      # Better CNN encoder
├── Larger receptive fields     # 3x3 → 5x5 kernels
├── More channels              # 32 → 64 → 128
└── Batch normalization        # Stabilize training
```

#### Key Changes:
1. **Data Augmentation Pipeline**:
   - Random crops from 84x84 to 64x64 (Crafter's native size)
   - Color jittering: brightness ±0.2, contrast ±0.2
   - Apply only during training (not evaluation)

2. **Improved Q-Network**:
   - Better CNN encoder with more capacity
   - Batch normalization for stability
   - Proper weight initialization

3. **Target Network Improvements**:
   - Soft updates (τ = 0.01) instead of hard updates
   - More stable Q-value estimates

#### Expected Performance:
- **Crafter Score**: 2.0-4.0% (+100-300% improvement)
- **New Achievements**: make_wood_pickaxe, defeat_skeleton, collect_stone
- **Episode Length**: ~100-200 steps
- **Sample Efficiency**: Faster learning due to augmentation

---

### ⚡ IMPROVEMENT 2 - EVALUATION 3
**Goal**: Advanced optimizations for maximum performance

#### Advanced Components:
```python
# Prioritized Experience Replay
PrioritizedReplayBuffer
├── TD-error based sampling     # Focus on important transitions
├── Importance sampling weights # Correct bias from non-uniform sampling
└── Priority decay             # Reduce importance over time

# Multi-step Learning
NStepReturns
├── n=3 step returns           # Look ahead 3 steps
├── Discounted reward sum      # R_t + γR_{t+1} + γ²R_{t+2} + γ³V_{t+3}
└── More stable targets        # Reduce variance

# Double Q-Learning
DoubleQUpdates                  # Reduce overestimation bias
├── Action selection network    # Choose action with main network
└── Value evaluation network    # Evaluate with target network
```

#### Technical Enhancements:
1. **Prioritized Replay**:
   - Sample transitions based on TD-error magnitude
   - Higher probability for surprising/important experiences
   - Importance sampling weights to correct bias

2. **N-step Returns (n=3)**:
   - Multi-step temporal difference learning
   - More stable targets, faster propagation of rewards
   - Balance between bias (n=1) and variance (n=∞)

3. **Double Q-Learning**:
   - Decouple action selection from value evaluation
   - Reduces overestimation bias common in Q-learning
   - More accurate Q-value estimates

4. **Training Optimizations**:
   - Gradient clipping (norm=10.0)
   - Learning rate scheduling
   - Improved exploration schedule

#### Expected Performance:
- **Crafter Score**: 4.0-8.0% (competitive with published results)
- **Advanced Achievements**: collect_iron, make_iron_tools, defeat_zombie
- **Episode Length**: ~200-500 steps
- **Robustness**: More consistent performance across runs

---

## Implementation Timeline

### Week 1: Baseline Implementation
- [ ] Create `src/agents/drqv2_agent.py` with BaseAgent inheritance
- [ ] Implement basic Q-network architecture
- [ ] Create simple replay buffer
- [ ] Integrate with `train.py`
- [ ] **Run Evaluation 1**: Baseline performance

### Week 2: Core DrQ-v2 Features
- [ ] Implement data augmentation pipeline
- [ ] Improve Q-network architecture
- [ ] Add soft target updates
- [ ] Debug and optimize training
- [ ] **Run Evaluation 2**: Measure improvement from augmentation

### Week 3: Advanced Optimizations
- [ ] Implement prioritized experience replay
- [ ] Add n-step returns computation
- [ ] Integrate double Q-learning
- [ ] Fine-tune hyperparameters
- [ ] **Run Evaluation 3**: Final performance

### Week 4: Analysis & Comparison
- [ ] Compare all three evaluations
- [ ] Analyze which components contributed most
- [ ] Generate plots and reports
- [ ] Prepare algorithm explanation for report

---

## File Structure

```
src/
├── agents/
│   ├── base_agent.py           # Shared interface (TODO: complete for DrQ-v2)
│   └── drqv2_agent.py         # Main DrQ-v2 implementation
├── utils/
│   ├── replay_buffer.py       # Basic + Prioritized replay buffers
│   ├── data_augmentation.py   # Random crops, color jittering
│   └── networks.py            # Q-network architectures
└── evaluation/
    └── (existing evaluation tools)
```

## Evaluation Strategy

After each phase:
```bash
# Quick development check (10 episodes)
python test_model.py models/drqv2_v{1,2,3}.zip drqv2 10

# Full evaluation for report (100 episodes)
python evaluate.py --model_path models/drqv2_v{1,2,3}.zip --algorithm drqv2 --episodes 100
```

### Key Metrics to Track:
1. **Crafter Score** (geometric mean of achievements) - Main benchmark
2. **Individual Achievement Rates** - Which ones improve?
3. **Sample Efficiency** - Performance vs. training steps
4. **Episode Length** - Survival time indicator
5. **Training Stability** - Consistent improvement?

## Success Criteria

### Minimum Viable Performance:
- **Baseline**: Score ≥ 0.5%, basic achievements unlocked
- **Improvement 1**: Score ≥ 2.0%, clear benefit from augmentation
- **Improvement 2**: Score ≥ 4.0%, competitive with baseline algorithms

### Stretch Goals:
- **Score > 6.0%**: Competitive with DreamerV2 (6.8%)
- **Advanced Achievements**: collect_iron, make_iron_sword, defeat_zombie
- **Sample Efficiency**: Good performance within 1M steps

## Common Issues & Solutions

### Training Problems:
- **Slow Convergence**: Increase learning rate, reduce batch size
- **Unstable Training**: Add gradient clipping, reduce target update frequency
- **Poor Exploration**: Tune epsilon schedule, add noise to actions
- **Memory Issues**: Reduce replay buffer size, use smaller networks

### Environment-Specific:
- **Sparse Rewards**: Focus on survival reward initially
- **Visual Complexity**: Start with simpler augmentations
- **Long Episodes**: Use proper discounting (γ=0.99)
- **Achievement Unlock**: Monitor specific achievement progress

## Research Context

**Original DrQ-v2 Paper**: [Data-Efficient Reinforcement Learning with Self-Predictive Representations](https://arxiv.org/abs/2107.09645)

**Key Insights for Report**:
1. Data augmentation is crucial for visual RL sample efficiency
2. Random crops work better than other augmentations for control
3. Soft target updates provide more stable learning
4. Architecture improvements compound with augmentation benefits

**Crafter-Specific Adaptations**:
- 64x64 RGB observations (vs. typical 84x84)
- 17 discrete actions (vs. continuous control)
- Sparse achievement rewards + dense survival rewards
- Long episodes with hierarchical objectives

---
*Last Updated: 2025-09-29*
*Owner: Anand Patel*