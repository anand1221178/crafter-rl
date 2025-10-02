"""
DrQ-v2 Agent Implementation

This module implements the Data-Regularized Q-Learning v2 algorithm for the
Crafter environment. DrQ-v2 is designed for visual RL tasks and uses data
augmentation to improve sample efficiency.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Dict, Optional, Any
from collections import deque

# Import our custom modules
from .base_agent import BaseAgent
from ..utils.replay_buffer import ReplayBuffer
from ..utils.networks import QNetwork, create_target_network, soft_update


class DrQv2Agent(BaseAgent):

    def __init__(self, 
                observation_shape: tuple = (64, 64, 3),
                num_actions: int = 17,
                device: str = 'cpu',
                # Q-learning hyperparameters
                learning_rate: float = 3e-4,
                gamma: float = 0.99,
                batch_size: int = 32,
                # Exploration parameters
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay_steps: int = 100_000,
                # Network update parameters
                target_update_freq: int = 1,  # Soft update every step
                tau: float = 0.01,  # Soft update coefficient
                # Replay buffer parameters
                replay_buffer_size: int = 100_000,
                min_replay_size: int = 1000):
        """
        Initialize the DrQ-v2 agent.
        
        Args:
            observation_shape: Shape of observations (H, W, C)
            num_actions: Number of discrete actions
            device: Device for PyTorch tensors
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor for future rewards
            batch_size: Size of batches sampled from replay buffer
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon over
            target_update_freq: How often to update target network
            tau: Soft update coefficient (0.01 = 1% new, 99% old)
            replay_buffer_size: Maximum transitions in replay buffer
            min_replay_size: Minimum transitions before training starts
        """
        super().__init__(observation_shape, num_actions, device)

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.min_replay_size = min_replay_size

        # Current exploration rate
        self.epsilon = epsilon_start

        # Set device
        self.device = torch.device(device)

        # Initialize networks
        # Convert observation shape from HWC to CHW for PyTorch
        torch_obs_shape = (observation_shape[2], observation_shape[0],
    observation_shape[1])

        self.q_network = QNetwork(
            observation_shape=torch_obs_shape,
            num_actions=num_actions
        ).to(self.device)

        # Target network for stable Q-learning
        self.target_network = create_target_network(self.q_network).to(self.device)

        # Optimizer for Q-network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Loss function
        self.criterion = nn.MSELoss()

        # Replay buffer for experience storage
        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            observation_shape=observation_shape,
            device=device
        )

        # Training metrics tracking
        self.episode_rewards = deque(maxlen=100)  # Last 100 episodes
        self.episode_lengths = deque(maxlen=100)
        self.q_losses = deque(maxlen=1000)  # Last 1000 updates

        print(f"🚀 DrQ-v2 Agent initialized!")
        print(f"   Device: {self.device}")
        print(f"   Q-network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        print(f"   Replay buffer capacity: {replay_buffer_size:,}")

    def act(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        During training, we explore with probability epsilon.
        During evaluation, we act greedily (no exploration).
        
        Args:
            observation: Current observation (64, 64, 3)
            training: Whether we're in training mode
        
        Returns:
            Action index (0-16 for Crafter)
        """
        # Exploration during training only
        if training and np.random.random() < self.epsilon:
            # Random action for exploration
            return np.random.randint(0, self.num_actions)

        # Greedy action selection
        # Convert observation to PyTorch format
        obs_tensor = self._obs_to_tensor(observation)

        # Get Q-values from network
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)

        # Select action with highest Q-value
        action = q_values.argmax().item()
        return action

    def store_experience(self, 
                        obs: np.ndarray, 
                        action: int, 
                        reward: float,
                        next_obs: np.ndarray, 
                        done: bool) -> None:
        """
        Store a transition in the replay buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
        """
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step if enough data is available.
        
        This is where the Q-learning magic happens:
        1. Sample a batch from replay buffer
        2. Compute target Q-values using target network
        3. Compute current Q-values using main network
        4. Calculate loss and update main network
        5. Soft update target network
        
        Returns:
            Dictionary of training metrics or None if not ready
        """
        # Check if we have enough data to train
        if not self.replay_buffer.is_ready(self.min_replay_size):
            return None

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample_with_augmentation(self.batch_size)

        # Convert to PyTorch tensors
        obs = self._batch_to_tensor(batch['obs'])
        actions = torch.LongTensor(batch['action']).to(self.device)

        # CRITICAL: Clip rewards to prevent Q-value explosion
        # Crafter gives large achievement bonuses that destabilize training
        rewards = torch.FloatTensor(batch['reward']).clamp(-1.0, 1.0).to(self.device)

        next_obs = self._batch_to_tensor(batch['next_obs'])
        dones = torch.BoolTensor(batch['done']).to(self.device)

        # Current Q-values: Q(s, a)
        current_q_values = self.q_network(obs).gather(1,actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ * max_a Q_target(s', a)
        with torch.no_grad():
            next_q_values = self.target_network(next_obs).max(1)[0]
            # Set target to reward if episode ended (no future rewards)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Soft update target network
        if self.training_step % self.target_update_freq == 0:
            soft_update(self.q_network, self.target_network, self.tau)

        # Update exploration rate
        self._update_epsilon()

        # Increment training step
        self.training_step += 1

        # Track metrics
        self.q_losses.append(loss.item())

        # Return training metrics
        return {
            'q_loss': loss.item(),
            'epsilon': self.epsilon,
            'replay_buffer_size': len(self.replay_buffer),
            'training_step': self.training_step,
            'avg_q_value': current_q_values.mean().item(),
            'target_q_value': target_q_values.mean().item()
        }

    def save(self, path: str) -> None:
        """
        Save the agent's state to disk.
        
        Saves Q-network weights, optimizer state, and replay buffer.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'tau': self.tau,
            }
        }

        torch.save(checkpoint, path)
        print(f"💾 Agent saved to {path}")

    def load(self, path: str) -> None:
        """
        Load the agent's state from disk.
        
        Loads Q-network weights and training state.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']

        print(f"📁 Agent loaded from {path}")
        print(f"   Training step: {self.training_step}")
        print(f"   Epsilon: {self.epsilon:.4f}")

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """
        Convert numpy observation to PyTorch tensor.
        
        Handles shape conversion from HWC to CHW and normalization.
        """
        if obs.ndim == 3:
            obs = obs[np.newaxis, ...]  # Add batch dimension

        # Convert to float and normalize
        obs = obs.astype(np.float32) / 255.0

        # Convert from HWC to CHW
        obs = obs.transpose(0, 3, 1, 2)

        return torch.FloatTensor(obs).to(self.device)

    def _batch_to_tensor(self, batch_obs: np.ndarray) -> torch.Tensor:
        """Convert batch of observations to tensor."""
        # Already normalized in replay buffer sampling
        batch_obs = batch_obs.transpose(0, 3, 1, 2)  # HWC to CHW
        return torch.FloatTensor(batch_obs).to(self.device)

    def _update_epsilon(self) -> None:
        """
        Update exploration rate using linear decay.
        
        Epsilon decreases linearly from epsilon_start to epsilon_end
        over epsilon_decay_steps training steps.
        """
        if self.training_step < self.epsilon_decay_steps:
            # Linear decay
            decay_ratio = self.training_step / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_ratio
        else:
            # Keep at minimum value
            self.epsilon = self.epsilon_end

    def get_stats(self) -> Dict[str, float]:
        """
        Get agent statistics for logging.
        
        Returns:
            Dictionary with agent metrics
        """
        stats = {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'replay_buffer_size': len(self.replay_buffer),
        }

        # Add loss statistics if available
        if self.q_losses:
            stats['avg_q_loss'] = np.mean(list(self.q_losses))
            stats['recent_q_loss'] = self.q_losses[-1]

        # Add episode statistics if available
        if self.episode_rewards:
            stats['avg_episode_reward'] = np.mean(list(self.episode_rewards))
            stats['avg_episode_length'] = np.mean(list(self.episode_lengths))

        # Add replay buffer statistics
        buffer_stats = self.replay_buffer.get_stats()
        stats.update({f'buffer_{k}': v for k, v in buffer_stats.items()})

        return stats

    def end_episode(self, episode_reward: float, episode_length: int) -> None:
        """
        Called at the end of each episode to track statistics.
        
        Args:
            episode_reward: Total reward for the episode
            episode_length: Number of steps in the episode
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)