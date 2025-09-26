"""
DrQ-v2 (Data-Regularized Q-Learning v2) implementation for Crafter.

DrQ-v2 is a model-free RL algorithm designed for visual control tasks.
It combines DQN with data augmentation for improved sample efficiency.

Original paper: https://arxiv.org/abs/2107.09645
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Tuple, Optional


class CNNEncoder(nn.Module):
    """
    Convolutional encoder for processing 64x64 RGB images.
    Based on DrQ-v2 architecture with modifications for discrete actions.
    """

    def __init__(self, observation_shape: Tuple[int, int, int], feature_dim: int = 50):
        super().__init__()
        assert len(observation_shape) == 3  # (H, W, C)

        self.convs = nn.Sequential(
            nn.Conv2d(observation_shape[2], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU()
        )

        # Calculate conv output size
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_shape).permute(0, 3, 1, 2)
            conv_out = self.convs(sample_input)
            conv_out_size = conv_out.numel()

        self.fc = nn.Linear(conv_out_size, feature_dim)

    def forward(self, obs):
        # obs shape: (batch, H, W, C) -> (batch, C, H, W)
        if obs.dim() == 4:
            obs = obs.permute(0, 3, 1, 2)
        elif obs.dim() == 3:
            obs = obs.permute(2, 0, 1).unsqueeze(0)

        h = self.convs(obs / 255.0)  # Normalize to [0, 1]
        h = h.view(h.size(0), -1)
        return self.fc(h)


class QNetwork(nn.Module):
    """
    Q-network for discrete actions in Crafter environment.
    """

    def __init__(self, feature_dim: int, num_actions: int, hidden_dim: int = 1024):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, features):
        return self.q_net(features)


class RandomShiftsAug:
    """
    Data augmentation using random shifts (key component of DrQ-v2).
    """

    def __init__(self, pad: int = 4):
        self.pad = pad

    def __call__(self, x):
        # TODO(human): Implement random shifts augmentation
        # x shape: (batch, H, W, C) or (H, W, C)
        # Apply random crops after padding to simulate camera movement
        # This is a key innovation of DrQ-v2 for visual RL
        pass


class ReplayBuffer:
    """
    Experience replay buffer with efficient storage for image observations.
    """

    def __init__(self, capacity: int, observation_shape: Tuple[int, int, int]):
        self.capacity = capacity
        self.obs_shape = observation_shape

        # Pre-allocate memory for efficiency
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.size = 0
        self.ptr = 0

    def add(self, obs, action, reward, next_obs, done):
        """Add a transition to the buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices]
        }


class DrQv2Agent:
    """
    DrQ-v2 agent adapted for Crafter's discrete action space.
    """

    def __init__(
        self,
        observation_shape: Tuple[int, int, int],
        num_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 100000,
        buffer_size: int = 100000,
        batch_size: int = 128,
        update_freq: int = 2,
        target_update_freq: int = 2000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq

        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Networks
        self.encoder = CNNEncoder(observation_shape).to(self.device)
        self.q_net1 = QNetwork(50, num_actions).to(self.device)  # Twin Q-networks
        self.q_net2 = QNetwork(50, num_actions).to(self.device)

        # Target networks
        self.target_encoder = CNNEncoder(observation_shape).to(self.device)
        self.target_q_net1 = QNetwork(50, num_actions).to(self.device)
        self.target_q_net2 = QNetwork(50, num_actions).to(self.device)

        # Copy weights to target networks
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.target_q_net2.state_dict())

        # Optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.q_net1_opt = torch.optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q_net2_opt = torch.optim.Adam(self.q_net2.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, observation_shape)

        # Data augmentation
        self.aug = RandomShiftsAug()

        # Training counters
        self.step_count = 0
        self.update_count = 0

    def act(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            features = self.encoder(obs_tensor)

            # Use minimum of twin Q-values (conservative)
            q_values1 = self.q_net1(features)
            q_values2 = self.q_net2(features)
            q_values = torch.min(q_values1, q_values2)

            action = q_values.argmax().item()

        return action

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def update(self) -> Optional[dict]:
        """
        Update the agent (called during training).
        """
        if self.replay_buffer.size < self.batch_size:
            return None

        # TODO(human): Implement the DrQ-v2 update step
        # Key components to implement:
        # 1. Sample batch from replay buffer
        # 2. Apply data augmentation to observations
        # 3. Compute Q-targets using twin networks
        # 4. Update Q-networks and encoder
        # 5. Soft update target networks
        # 6. Update exploration epsilon
        pass

    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'q_net1': self.q_net1.state_dict(),
            'q_net2': self.q_net2.state_dict(),
            'encoder_opt': self.encoder_opt.state_dict(),
            'q_net1_opt': self.q_net1_opt.state_dict(),
            'q_net2_opt': self.q_net2_opt.state_dict(),
            'step_count': self.step_count,
            'update_count': self.update_count,
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.q_net1.load_state_dict(checkpoint['q_net1'])
        self.q_net2.load_state_dict(checkpoint['q_net2'])
        self.encoder_opt.load_state_dict(checkpoint['encoder_opt'])
        self.q_net1_opt.load_state_dict(checkpoint['q_net1_opt'])
        self.q_net2_opt.load_state_dict(checkpoint['q_net2_opt'])
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint['update_count']
        self.epsilon = checkpoint['epsilon']