"""
Intrinsic Curiosity Module (ICM) for exploration in sparse reward environments.

Based on: Pathak et al. (2017) - "Curiosity-driven Exploration by Self-supervised Prediction"
Paper: https://arxiv.org/abs/1705.05363

Key Idea:
- Agent receives bonus reward for experiencing novel/surprising state transitions
- Intrinsic reward = prediction error of forward model
- Inverse model learns feature representation that focuses on agent-controllable aspects

Components:
1. Feature Encoder: Extracts compact features from observations (shared CNN)
2. Inverse Model: Predicts action from (state, next_state) features
3. Forward Model: Predicts next_state features from (state, action)
4. Intrinsic Reward: ||predicted_next_features - actual_next_features||²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module for exploration.

    Provides intrinsic reward bonus based on prediction error of state transitions.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_actions: int = 17,
        inverse_lr: float = 1e-3,
        forward_lr: float = 1e-3,
        beta: float = 0.2,  # Weight for intrinsic reward (0.2 = 20% intrinsic, 80% extrinsic)
        lambda_inverse: float = 0.1,  # Weight for inverse model loss
        device: str = 'cpu'
    ):
        """
        Initialize ICM module.

        Args:
            feature_dim: Dimension of encoded features (default 512)
            num_actions: Number of discrete actions (17 for Crafter)
            inverse_lr: Learning rate for inverse model
            forward_lr: Learning rate for forward model
            beta: Weight for intrinsic reward (0 = only extrinsic, 1 = only intrinsic)
            lambda_inverse: Weight for inverse model loss in total ICM loss
            device: Device for tensors
        """
        super(ICMModule, self).__init__()

        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.beta = beta
        self.lambda_inverse = lambda_inverse
        self.device = device

        # Feature encoder: Shared CNN (same architecture as PPO policy network)
        # Encodes 64x64x3 RGB observation into compact feature vector
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten(),  # 64 * 8 * 8 = 4096
            nn.Linear(4096, feature_dim),  # Compress to feature_dim
            nn.ReLU()
        ).to(device)

        # Inverse Model: Predicts action from (state, next_state) features
        # Learns representation that focuses on controllable aspects of environment
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),  # Concatenate state + next_state features
            nn.ReLU(),
            nn.Linear(256, num_actions)  # Predict action logits
        ).to(device)

        # Forward Model: Predicts next_state features from (state, action)
        # Prediction error = intrinsic reward (agent is curious about surprises!)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + num_actions, 256),  # State features + one-hot action
            nn.ReLU(),
            nn.Linear(256, feature_dim)  # Predict next_state features
        ).to(device)

        # Separate optimizers for inverse and forward models
        self.inverse_optimizer = torch.optim.Adam(
            list(self.feature_encoder.parameters()) + list(self.inverse_model.parameters()),
            lr=inverse_lr
        )
        self.forward_optimizer = torch.optim.Adam(
            self.forward_model.parameters(),
            lr=forward_lr
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation to feature vector.

        Args:
            obs: Observation tensor, shape (batch, 3, 64, 64), values in [0, 1]

        Returns:
            features: Feature vector, shape (batch, feature_dim)
        """
        return self.feature_encoder(obs)

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute intrinsic reward based on forward model prediction error.

        Args:
            obs: Current observation (batch, 3, 64, 64)
            next_obs: Next observation (batch, 3, 64, 64)
            action: Action taken (batch,) - discrete action indices

        Returns:
            intrinsic_reward: Prediction error (batch,) - higher = more novel/surprising
        """
        with torch.no_grad():
            # Encode observations to features
            state_features = self.encode_observation(obs)
            next_state_features = self.encode_observation(next_obs)

            # One-hot encode action
            action_onehot = F.one_hot(action.long(), num_classes=self.num_actions).float()

            # Forward model prediction
            forward_input = torch.cat([state_features, action_onehot], dim=1)
            predicted_next_features = self.forward_model(forward_input)

            # Intrinsic reward = prediction error (L2 distance)
            # High error = surprising transition = curiosity reward!
            intrinsic_reward = torch.norm(
                predicted_next_features - next_state_features,
                dim=1,
                p=2
            ) ** 2

            # Normalize by feature dimension for stability
            intrinsic_reward = intrinsic_reward / self.feature_dim

        return intrinsic_reward

    def update(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Update ICM models (inverse and forward).

        Args:
            obs: Batch of observations (batch, 3, 64, 64)
            next_obs: Batch of next observations (batch, 3, 64, 64)
            action: Batch of actions (batch,)

        Returns:
            Tuple of (inverse_loss, forward_loss, total_loss)
        """
        # Encode observations
        state_features = self.encode_observation(obs)
        next_state_features = self.encode_observation(next_obs)

        # Inverse model: Predict action from state transition
        inverse_input = torch.cat([state_features, next_state_features], dim=1)
        action_logits = self.inverse_model(inverse_input)
        inverse_loss = F.cross_entropy(action_logits, action.long())

        # Forward model: Predict next state features
        action_onehot = F.one_hot(action.long(), num_classes=self.num_actions).float()
        forward_input = torch.cat([state_features.detach(), action_onehot], dim=1)
        predicted_next_features = self.forward_model(forward_input)
        forward_loss = F.mse_loss(predicted_next_features, next_state_features.detach())

        # Total ICM loss (weighted combination)
        total_loss = self.lambda_inverse * inverse_loss + (1 - self.lambda_inverse) * forward_loss

        # Update inverse model (includes feature encoder)
        self.inverse_optimizer.zero_grad()
        inverse_loss.backward(retain_graph=True)
        self.inverse_optimizer.step()

        # Update forward model
        self.forward_optimizer.zero_grad()
        forward_loss.backward()
        self.forward_optimizer.step()

        return inverse_loss.item(), forward_loss.item(), total_loss.item()

    def get_combined_reward(
        self,
        extrinsic_reward: torch.Tensor,
        intrinsic_reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine extrinsic (environment) and intrinsic (curiosity) rewards.

        Args:
            extrinsic_reward: Reward from environment (batch,)
            intrinsic_reward: Curiosity-based reward (batch,)

        Returns:
            combined_reward: Weighted sum (batch,)
        """
        return (1 - self.beta) * extrinsic_reward + self.beta * intrinsic_reward


if __name__ == '__main__':
    # Test ICM module
    print("Testing ICM Module...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    icm = ICMModule(feature_dim=512, num_actions=17, device=device)

    # Create dummy batch
    batch_size = 4
    obs = torch.randn(batch_size, 3, 64, 64).to(device)
    next_obs = torch.randn(batch_size, 3, 64, 64).to(device)
    actions = torch.randint(0, 17, (batch_size,)).to(device)

    # Test encoding
    features = icm.encode_observation(obs)
    print(f"✓ Feature encoding: {obs.shape} -> {features.shape}")

    # Test intrinsic reward
    intrinsic_reward = icm.compute_intrinsic_reward(obs, next_obs, actions)
    print(f"✓ Intrinsic reward: {intrinsic_reward.shape}, mean={intrinsic_reward.mean().item():.4f}")

    # Test update
    inv_loss, fwd_loss, total_loss = icm.update(obs, next_obs, actions)
    print(f"✓ ICM update: inverse_loss={inv_loss:.4f}, forward_loss={fwd_loss:.4f}, total_loss={total_loss:.4f}")

    print("\n✅ ICM Module tests passed!")
