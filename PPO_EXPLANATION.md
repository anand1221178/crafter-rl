# PPO (Proximal Policy Optimization) - Detailed Explanation

**Author**: Anand Patel
**Course**: COMS4061A/COMS7071A Reinforcement Learning
**Date**: October 8, 2025

---

## Table of Contents
1. [What is PPO?](#what-is-ppo)
2. [Why Policy Gradient Methods?](#why-policy-gradient-methods)
3. [The PPO Algorithm](#the-ppo-algorithm)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Implementation Details](#implementation-details)
6. [Why PPO for Crafter?](#why-ppo-for-crafter)
7. [Comparison with DQN](#comparison-with-dqn)

---

## What is PPO?

**Proximal Policy Optimization (PPO)** is a policy gradient reinforcement learning algorithm developed by OpenAI in 2017. It's designed to be:
- **Simple to implement** (no complex second-order optimization like TRPO)
- **Sample efficient** (uses multiple epochs of minibatch updates)
- **Stable** (clipped objective prevents destructive policy updates)
- **General-purpose** (works on continuous and discrete action spaces)

### Key Idea
PPO directly learns a **stochastic policy** π(a|s) that outputs action probabilities. Instead of learning Q-values and deriving a policy (like DQN), PPO learns the policy directly.

The "Proximal" in PPO means **nearby** - the algorithm ensures the new policy stays close to the old policy during updates, preventing catastrophic performance collapse.

---

## Why Policy Gradient Methods?

### The Problem with Value-Based Methods (DQN)

**DQN learns**: Q(s, a) → value of each action
**DQN's policy**: π(s) = argmax_a Q(s, a) (pick highest Q-value)
**Problem**: Deterministic policy with external exploration (ε-greedy)

**Example - DQN in Crafter**:
```
State: Agent near tree
Q-values: [chop: 0.1, move_left: 0.2, move_right: 0.15, ...]

With ε=0.05:
  - 95% of time: Take argmax (move_left)
  - 5% of time: Random action (uniform over 17 actions)

Issue: If Q(chop) is low initially (haven't tried it), DQN almost never tries chop
       → Never discovers tree → wood → reward
       → Q(chop) stays low forever
```

### The Policy Gradient Solution

**PPO learns**: π(a|s) → probability distribution over actions
**PPO's policy**: Sample action from π(a|s)
**Advantage**: Natural exploration built into policy

**Example - PPO in Crafter**:
```
State: Agent near tree
Policy: π = [chop: 0.3, move_left: 0.2, move_right: 0.2, attack: 0.1, ...]

Samples action from distribution:
  - Tries chop 30% of time
  - Tries move_left 20% of time
  - Tries other actions proportionally

Discovers: chop → wood → +1 reward

Update: Increase probability of successful action
New policy: π = [chop: 0.8, move_left: 0.1, ...]

Result: Natural exploration → discovers achievements → reinforces successful behaviors
```

---

## The PPO Algorithm

### High-Level Overview

```
1. Initialize policy network π_θ (actor) and value network V_φ (critic)
2. For each iteration:
   a. Collect trajectories using current policy π_old
   b. Compute advantages using GAE (Generalized Advantage Estimation)
   c. Update policy using clipped objective (multiple epochs)
   d. Update value function to minimize prediction error
```

### Detailed Steps

#### Step 1: Collect Trajectories
```python
# Rollout current policy for N steps
for t in range(n_steps):
    action, log_prob, value = policy.act(state)
    next_state, reward, done = env.step(action)

    # Store in rollout buffer
    buffer.store(state, action, reward, log_prob, value, done)
    state = next_state
```

#### Step 2: Compute Advantages (GAE)
```python
# Generalized Advantage Estimation
# Advantage = "how much better was this action than average?"

advantages = []
gae = 0
for t in reversed(range(n_steps)):
    delta = rewards[t] + gamma * values[t+1] - values[t]  # TD error
    gae = delta + gamma * gae_lambda * gae
    advantages[t] = gae

# Compute returns (targets for value function)
returns = advantages + values
```

**Why GAE?**
- **Bias-variance tradeoff**: λ=0 (low variance, high bias), λ=1 (high variance, low bias)
- **Smooth credit assignment**: Distributes credit over multiple timesteps
- **Reduces variance**: Makes policy gradient more stable

#### Step 3: PPO Update (Clipped Objective)
```python
# Multiple epochs over collected data
for epoch in range(n_epochs):
    for batch in minibatch_iterator(buffer):
        # Compute probability ratio
        new_log_probs = policy.log_prob(batch.states, batch.actions)
        old_log_probs = batch.log_probs
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * batch.advantages
        surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * batch.advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss
        value_pred = value_network(batch.states)
        value_loss = F.mse_loss(value_pred, batch.returns)

        # Entropy bonus (encourages exploration)
        entropy = policy.entropy(batch.states).mean()

        # Total loss
        loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

        # Update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()
```

---

## Mathematical Foundations

### 1. Policy Gradient Theorem

**Goal**: Maximize expected return J(θ) = E[Σ r_t]

**Policy Gradient**:
```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) * A(s,a)]

Where:
- π_θ(a|s): Policy (probability of action a in state s)
- A(s,a): Advantage (how good is action a compared to average)
```

**Intuition**:
- If advantage > 0 (action better than average): Increase probability of that action
- If advantage < 0 (action worse than average): Decrease probability of that action

### 2. Importance Sampling

**Problem**: On-policy methods need fresh data (can't reuse old trajectories)

**Solution**: Importance sampling - weight old data by probability ratio

```
Objective = E_{s,a ~ π_old}[ (π_new(a|s) / π_old(a|s)) * A(s,a) ]

Ratio r(θ) = π_new(a|s) / π_old(a|s)
- If r > 1: New policy assigns higher probability to action than old policy
- If r < 1: New policy assigns lower probability to action than old policy
```

**Problem with vanilla importance sampling**: Ratio can be very large → unstable updates

### 3. Clipped Objective (PPO's Key Innovation)

**Solution**: Clip the ratio to prevent large policy changes

```
L^CLIP(θ) = E[ min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) ]

Where:
- r(θ) = π_new(a|s) / π_old(a|s)
- clip(r, 1-ε, 1+ε) restricts r to [1-ε, 1+ε] (typically ε=0.2)
- min(...) takes the more conservative objective
```

**Intuition**:
```
Case 1: Advantage > 0 (good action)
- Want to increase π(a|s)
- If r > 1+ε: Clip to 1+ε (prevent too large increase)

Case 2: Advantage < 0 (bad action)
- Want to decrease π(a|s)
- If r < 1-ε: Clip to 1-ε (prevent too large decrease)
```

**Why this works**:
- Prevents catastrophic policy updates
- Allows multiple epochs of updates without destroying old policy
- Balances exploration and exploitation

### 4. Generalized Advantage Estimation (GAE)

**Goal**: Estimate advantage function A(s,a) = Q(s,a) - V(s)

**TD error**: δ_t = r_t + γV(s_{t+1}) - V(s_t)

**GAE formula**:
```
A_t^GAE = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}

Where:
- λ=0: A = δ (low variance, high bias) - single-step TD
- λ=1: A = Σ(γ^l * r_{t+l}) - V(s_t) (high variance, low bias) - Monte Carlo
```

**Typical choice**: λ=0.95 (good bias-variance tradeoff)

---

## Implementation Details

### Network Architecture

#### Actor-Critic Network
```python
class ActorCritic(nn.Module):
    def __init__(self, num_actions=17):
        super().__init__()

        # Shared CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 64x64x3 → 32x32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 32x32x32 → 16x16x64
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), # 16x16x64 → 8x8x64
            nn.ReLU(),
            nn.Flatten()  # 8x8x64 = 4096
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.conv(x)
        return self.actor(features), self.critic(features)

    def get_action(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
```

### Hyperparameters (Baseline)

```python
# Network
lr_actor = 3e-4
lr_critic = 1e-3

# PPO
gamma = 0.99              # Discount factor
gae_lambda = 0.95         # GAE parameter
clip_epsilon = 0.2        # Clipping parameter
value_clip = 0.2          # Value function clipping
entropy_coef = 0.01       # Entropy bonus weight
vf_coef = 0.5             # Value function loss weight

# Training
n_steps = 2048            # Steps per rollout
batch_size = 64           # Minibatch size
n_epochs = 10             # Update epochs per rollout
max_grad_norm = 0.5       # Gradient clipping
```

### Training Loop

```python
def train_ppo(env, policy, n_iterations):
    for iteration in range(n_iterations):
        # 1. Collect trajectories
        rollout = collect_rollout(env, policy, n_steps)

        # 2. Compute advantages
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            gamma,
            gae_lambda
        )

        # 3. Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. PPO update
        for epoch in range(n_epochs):
            for batch in iterate_minibatches(rollout, batch_size):
                # Compute losses
                policy_loss, value_loss, entropy = compute_ppo_loss(
                    policy, batch, advantages, returns, clip_epsilon
                )

                # Update
                loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # 5. Log metrics
        log_metrics(iteration, rollout, policy_loss, value_loss)
```

---

## Why PPO for Crafter?

### 1. Sparse Rewards Require Exploration

**Crafter's Challenge**:
- 22 achievements (e.g., collect wood, make pickaxe, defeat zombie)
- Rewards are sparse - might see 1 reward per 1000 steps
- Achievements require multi-step sequences (chop tree → collect wood → make stick → make pickaxe)

**PPO's Advantage**:
- **Stochastic policy naturally explores**: Instead of ε-greedy (5% random), PPO's policy outputs probabilities that evolve with learning
- **Entropy bonus encourages diversity**: Policy is rewarded for being uncertain (high entropy) early in training
- **Advantage-based updates**: Focuses learning on actions that are better than average, not just positive rewards

### 2. Partial Observability Benefits from Memory

**Crafter is partially observable**:
- Agent sees 9×9 grid around itself (not full map)
- Can't see resources outside view
- Needs memory to plan (e.g., "I saw a tree 20 steps ago, go back")

**PPO with LSTM (Improvement 2)**:
- Recurrent policy remembers past observations
- Builds internal map representation
- Plans multi-step strategies

### 3. Stability is Critical

**Long training required**:
- 1M environment steps ≈ 8 hours on cluster
- Can't afford catastrophic forgetting mid-training

**PPO's stability mechanisms**:
- **Clipped objective**: Prevents policy from changing too fast
- **Multiple epochs**: Reuses data efficiently without instability
- **Value function clipping**: Prevents value predictions from diverging

---

## Comparison with DQN

### Paradigm Difference

| Aspect | DQN (Value-Based) | PPO (Policy Gradient) |
|--------|-------------------|----------------------|
| **What it learns** | Q(s,a) - expected return | π(a\|s) - action probabilities |
| **Policy** | Implicit (argmax Q) | Explicit (learned directly) |
| **Exploration** | External (ε-greedy) | Natural (policy entropy) |
| **Action selection** | Deterministic | Stochastic |
| **Update rule** | TD error | Policy gradient + clipping |
| **Data usage** | Off-policy (replay buffer) | On-policy (recent data) |
| **Stability** | Can diverge (deadly triad) | Stable (clipped objective) |

### Performance on Crafter

**Expected Results** (1M steps):
- **DQN**: 0.5-2% achievement rate (poor exploration)
- **PPO**: 18-25% achievement rate (effective exploration)
- **Improvement**: ~10-20× better performance

**Why the Gap?**

1. **Exploration**:
   - DQN's ε-greedy: 5% random actions (uniform over 17 actions) → 0.3% chance per action
   - PPO's stochastic policy: 20-40% chance for promising actions, adapts over time

2. **Credit Assignment**:
   - DQN: Relies on Bellman updates to propagate rewards backward (slow)
   - PPO: GAE directly computes advantages over trajectories (fast)

3. **Stability**:
   - DQN: Q-value overestimation, potential divergence
   - PPO: Clipped objective prevents destructive updates

---

## Three Improvement Phases

### Phase 1: Baseline PPO (Eval 1)
**Target**: 5-10% achievement rate

**Implementation**:
- Actor-Critic with CNN feature extractor
- GAE with λ=0.95
- Clipped objective with ε=0.2
- Fixed entropy bonus (0.01)

### Phase 2: Adaptive Exploration (Eval 2)
**Target**: 12-18% achievement rate

**Improvements**:
- **Entropy decay**: Start high (0.05) → end low (0.001)
  - Early: Explore aggressively
  - Late: Exploit learned strategies
- **Adaptive clipping**: Adjust ε based on KL divergence
  - If policy changing too fast (high KL): Reduce ε (more conservative)
  - If policy too cautious (low KL): Increase ε (allow larger updates)

### Phase 3: Recurrent Policy (Eval 3)
**Target**: 18-25% achievement rate

**Improvements**:
- **LSTM policy**: Replace feedforward network with LSTM
  - Remember past observations
  - Build internal map
  - Plan multi-step strategies
- **Sequence training**: Update LSTM using trajectory sequences (BPTT)

---

## Key Takeaways

1. **Policy gradient vs value-based**: PPO learns policy directly, DQN learns Q-values and derives policy
2. **Natural exploration**: Stochastic policies explore inherently, no need for ε-greedy
3. **Clipped objective**: PPO's key innovation - prevents catastrophic policy updates
4. **Actor-Critic**: Combines policy learning (actor) with value estimation (critic)
5. **GAE**: Balances bias-variance in advantage estimation
6. **Stability**: PPO is more stable than vanilla policy gradient and on-par with DQN
7. **Performance**: Expected 10-20× better than DQN on Crafter due to superior exploration

---

## References

1. **PPO Paper**: Schulman et al. 2017 - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. **GAE Paper**: Schulman et al. 2016 - [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
3. **TRPO Paper**: Schulman et al. 2015 - [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
4. **Spinning Up in Deep RL**: [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
5. **Sutton & Barto**: Reinforcement Learning: An Introduction (Chapter 13)

---

*This document provides a comprehensive explanation of PPO for the RL course project. For implementation details, see `src/agents/ppo_agent.py`.*
