"""
Dyna-Q Agent Implementation for Crafter

This module implements the Dyna-Q algorithm, which integrates direct reinforcement learning,
model learning, and planning for sample-efficient learning in sparse reward environments.

References:
    Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.).
    MIT Press. Chapter 8, Section 8.2: Dyna: Integrated Planning, Acting, and Learning.

    Primary Algorithm Source: Figure 8.2 (Tabular Dyna-Q), p. 164

    Key Quote (p. 160):
    "The Dyna-Q agent is an instance of the general Dyna architecture that uses Q-learning
    for direct RL, a simple table lookup model, and a random-sample one-step tabular
    Q-planning method."

Neural Network Components Adapted From:
    Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
    Nature, 518(7540), 529-533.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque

from .base_agent import BaseAgent
from ..utils.networks import QNetwork
from ..utils.replay_buffer import ReplayBuffer
from ..models.world_model import WorldModel


class DynaQAgent(BaseAgent):
    """
    Dyna-Q agent combining deep Q-learning with model-based planning.

    Architecture:
        - Q-function: Deep neural network (CNN for visual observations)
        - World model: Tabular storage with feature hashing
        - Planning: Random-sample one-step Q-planning

    Implementation follows Sutton & Barto (2018), Figure 8.2, with adaptations
    for deep RL and high-dimensional visual observations.

    Args:
        observation_shape: Shape of observations (H, W, C), e.g., (64, 64, 3)
        num_actions: Number of discrete actions (17 for Crafter)
        device: Device to use ('cpu' or 'cuda')
        learning_rate: Q-network learning rate (default: 1e-4)
        gamma: Discount factor (default: 0.99)
        batch_size: Minibatch size for Q-learning (default: 32)
        epsilon_start: Initial exploration rate (default: 1.0)
        epsilon_end: Final exploration rate (default: 0.05)
        epsilon_decay_steps: Steps to decay epsilon (default: 750000)
        tau: Target network soft update rate (default: 0.01)
        replay_buffer_size: Experience replay capacity (default: 100000)
        min_replay_size: Minimum samples before training (default: 1000)
        planning_steps: Number of planning updates per real step (default: 5)
        model_capacity: World model capacity (default: 50000)
    """

    def __init__(
        self,
        observation_shape: Tuple[int, int, int] = (64, 64, 3),
        num_actions: int = 17,
        device: str = 'cpu',
        # Q-learning hyperparameters
        learning_rate: float = 3e-4,  # Increased from 1e-4 for faster learning
        gamma: float = 0.99,
        batch_size: int = 32,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,  # Higher final epsilon (more exploration)
        epsilon_decay_steps: int = 900_000,  # Slower decay (90% of 1M steps)
        tau: float = 0.005,  # Slower target network updates for stability
        replay_buffer_size: int = 100_000,
        min_replay_size: int = 5000,  # More samples before training starts
        # Dyna-Q specific hyperparameters
        planning_steps: int = 5,
        model_capacity: int = 50_000,
    ):
        super().__init__(observation_shape, num_actions, device)

        # Q-learning components
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.min_replay_size = min_replay_size

        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Dyna-Q specific
        self.planning_steps = planning_steps

        # Initialize Q-network (CNN for visual observations)
        # Reference: Mnih et al. (2015) for architecture
        self.q_network = QNetwork(observation_shape, num_actions).to(device)
        self.target_network = QNetwork(observation_shape, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Replay buffer for direct RL
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)

        # World model for planning (Dyna-Q component)
        # Reference: Sutton & Barto (2018), Section 8.2
        self.world_model = WorldModel(capacity=model_capacity)

        # Training statistics
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        print(f"DynaQAgent initialized:")
        print(f"  Planning steps: {planning_steps}")
        print(f"  Model capacity: {model_capacity:,}")
        print(f"  Replay buffer: {replay_buffer_size:,}")
        print(f"  Device: {device}")

    def act(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Reference:
            Sutton & Barto (2018), Section 2.3 (ε-greedy action selection), p. 28

        Args:
            observation: Current state observation
            training: If True, use epsilon-greedy; if False, greedy

        Returns:
            Selected action (integer)
        """
        # Epsilon-greedy exploration during training
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        # Greedy action selection using Q-network
        with torch.no_grad():
            # Prepare observation for network (normalize to [0,1])
            obs_tensor = torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

            # Get Q-values
            q_values = self.q_network(obs_tensor)

            # Select action with highest Q-value
            action = q_values.argmax(dim=1).item()

        return action

    def store_experience(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        """
        Store experience in replay buffer and update world model.

        Implements Steps (d) and (e) from Sutton & Barto (2018), Figure 8.2:
        (d) Direct RL: Store in replay buffer
        (e) Model learning: "Model(S,A) ← R, S'"

        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Episode termination flag
        """
        # Store in replay buffer for direct RL (Step d)
        self.replay_buffer.add(obs, action, reward, next_obs, done)

        # Update world model (Step e from Sutton & Barto, 2018, Figure 8.2)
        self.world_model.update(obs, action, reward, next_obs)

    def update(self) -> Optional[Dict[str, float]]:
        """
        Dyna-Q update combining direct RL and planning.

        Implements Steps (d), (e), (f) from Sutton & Barto (2018), Figure 8.2:
        (d) Direct RL: Q-learning update from real experience
        (e) Model learning: Already done in store_experience()
        (f) Planning: n Q-learning updates from simulated experience

        Returns:
            Dictionary of training metrics, or None if not ready to train
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        # Step (d): Direct RL - Q-learning update from real experience
        q_loss = self._direct_rl_update()

        # Step (f): Planning - Q-learning updates from simulated experience
        # Reference: Sutton & Barto (2018), Figure 8.2, Step (f), p. 164
        planning_loss = self._planning_updates(self.planning_steps)

        # Update target network (soft update)
        self._update_target_network()

        # Decay epsilon
        self._decay_epsilon()

        # Increment step counter
        self.total_steps += 1

        # Collect metrics
        metrics = {
            'q_loss': q_loss,
            'planning_loss': planning_loss if planning_loss is not None else 0.0,
            'epsilon': self.epsilon,
            'replay_buffer_size': len(self.replay_buffer),
            'model_size': len(self.world_model),
            'total_steps': self.total_steps,
        }

        # Add world model statistics
        model_stats = self.world_model.get_statistics()
        metrics.update({f'model_{k}': v for k, v in model_stats.items()})

        return metrics

    def _direct_rl_update(self) -> float:
        """
        Q-learning update from real experience (Step d).

        Implements standard DQN update:
        Q(S,A) ← Q(S,A) + α[R + γ max_a' Q(S',a') - Q(S,A)]

        Reference:
            Sutton & Barto (2018), Equation 6.8, p. 131

        Returns:
            Q-loss value
        """
        # Sample minibatch from replay buffer
        try:
            batch = self.replay_buffer.sample(self.batch_size)
        except ValueError:
            # Not enough samples yet
            return 0.0

        # Extract data from batch dictionary
        obs = batch['obs']
        actions = batch['action']
        rewards = batch['reward']
        next_obs = batch['next_obs']
        dones = batch['done']

        # Convert to tensors (observations are uint8 [0-255], normalize to [0-1])
        obs_t = torch.FloatTensor(obs).permute(0, 3, 1, 2).to(self.device) / 255.0
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).permute(0, 3, 1, 2).to(self.device) / 255.0
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # Target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_t).max(1)[0]
            target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def _planning_updates(self, n_steps: int) -> Optional[float]:
        """
        Perform planning updates using world model.

        Implements Step (f) from Sutton & Barto (2018), Figure 8.2, p. 164:
        "Loop repeat n times:
            S ← random previously observed state
            A ← random action previously taken in S
            R, S' ← Model(S,A)
            Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a') - Q(S,A)]"

        Args:
            n_steps: Number of planning steps to perform

        Returns:
            Average planning loss, or None if model is empty
        """
        if len(self.world_model) == 0:
            return None

        total_planning_loss = 0.0
        successful_updates = 0

        for _ in range(n_steps):
            # Sample random transition from model
            transition = self.world_model.sample_random_transition()
            if transition is None:
                continue

            state_hash, action, next_state, reward = transition

            # Note: We don't have the actual state pixels, only the hash
            # For planning, we use the stored next_state as a starting point
            # and sample actions that were taken from it

            # Create simulated experience for Q-learning
            # We treat next_state as our "current state" for the update
            obs_t = torch.FloatTensor(next_state).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

            # Sample an action that was previously taken in this state
            sampled_transition = self.world_model.sample_from_state(next_state)
            if sampled_transition is None:
                continue

            sampled_action, sampled_next_state, sampled_reward = sampled_transition

            next_obs_t = torch.FloatTensor(sampled_next_state).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

            # Q-learning update using simulated transition
            current_q = self.q_network(obs_t)[0, sampled_action]

            with torch.no_grad():
                next_q = self.target_network(next_obs_t).max(1)[0]
                target_q = sampled_reward + self.gamma * next_q

            # Compute loss
            loss = self.criterion(current_q, target_q)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            self.optimizer.step()

            total_planning_loss += loss.item()
            successful_updates += 1

        return total_planning_loss / successful_updates if successful_updates > 0 else None

    def _update_target_network(self) -> None:
        """
        Soft update of target network.

        θ_target = τ θ_local + (1 - τ) θ_target

        Reference:
            Lillicrap et al. (2015). Continuous control with deep reinforcement learning.
        """
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _decay_epsilon(self) -> None:
        """
        Linearly decay epsilon from start to end over decay_steps.
        """
        if self.total_steps < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                          (self.total_steps / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end

    def end_episode(self, episode_reward: float, episode_length: int) -> None:
        """
        Called at end of episode to update statistics.

        Args:
            episode_reward: Total reward for the episode
            episode_length: Number of steps in the episode
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

    def save(self, path: str) -> None:
        """
        Save agent state to disk.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load agent state from disk.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)
        self.episode_lengths = deque(checkpoint['episode_lengths'], maxlen=100)
        print(f"Model loaded from {path}")
