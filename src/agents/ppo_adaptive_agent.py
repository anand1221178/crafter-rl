"""
PPO Agent with Adaptive Exploration (Improvement 1)

Enhancements over baseline:
1. Entropy decay: High exploration early → low exploration late
2. Adaptive clipping: Adjust clip_epsilon based on KL divergence
3. Early stopping: Stop updates if KL divergence too high

Expected improvement: 6.41% → 12-18% Crafter Score
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from src.agents.ppo_agent import PPOAgent


class PPOAdaptiveAgent(PPOAgent):
    """
    PPO agent with adaptive exploration mechanisms.

    Inherits from baseline PPOAgent and adds:
    - Entropy coefficient scheduling (decay over training)
    - Adaptive clip epsilon based on KL divergence
    - Early stopping when KL divergence exceeds threshold
    """

    def __init__(self,
                 observation_shape: tuple = (3, 64, 64),
                 num_actions: int = 17,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 # Network hyperparameters
                 hidden_dim: int = 512,
                 lr: float = 3e-4,
                 # PPO hyperparameters
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_clip: float = 0.2,
                 entropy_coef_start: float = 0.05,  # Start high (more exploration)
                 entropy_coef_end: float = 0.001,   # End low (more exploitation)
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 normalize_advantages: bool = True,
                 # Adaptive exploration parameters
                 target_kl: float = 0.015,          # Target KL divergence
                 kl_early_stop: bool = True,        # Stop epoch if KL too high
                 adaptive_clip: bool = True,        # Adjust clip_epsilon based on KL
                 total_timesteps: int = 1_000_000): # For entropy decay schedule
        """
        Initialize adaptive PPO agent.

        Args:
            entropy_coef_start: Initial entropy coefficient (high = more exploration)
            entropy_coef_end: Final entropy coefficient (low = more exploitation)
            target_kl: Target KL divergence for adaptive clipping
            kl_early_stop: If True, stop update epoch if KL exceeds target
            adaptive_clip: If True, adjust clip_epsilon based on KL
            total_timesteps: Total training steps (for entropy decay)
        """
        # Initialize base PPO agent
        super().__init__(
            observation_shape=observation_shape,
            num_actions=num_actions,
            device=device,
            hidden_dim=hidden_dim,
            lr=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            value_clip=value_clip,
            entropy_coef=entropy_coef_start,  # Will be updated dynamically
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            normalize_advantages=normalize_advantages
        )

        # Adaptive exploration parameters
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.target_kl = target_kl
        self.kl_early_stop = kl_early_stop
        self.adaptive_clip = adaptive_clip
        self.total_timesteps = total_timesteps

        # Initial clip epsilon (will be adapted)
        self.initial_clip_epsilon = clip_epsilon
        self.clip_epsilon = clip_epsilon

        # Tracking
        self.kl_divergences = []

    def _update_entropy_coef(self):
        """
        Update entropy coefficient based on training progress.

        Linear decay from entropy_coef_start to entropy_coef_end.
        Early training: High entropy → more exploration
        Late training: Low entropy → more exploitation
        """
        progress = min(1.0, self.training_step / self.total_timesteps)
        self.entropy_coef = self.entropy_coef_start + progress * (
            self.entropy_coef_end - self.entropy_coef_start
        )

    def _compute_kl_divergence(self, old_log_probs: torch.Tensor,
                               new_log_probs: torch.Tensor) -> float:
        """
        Compute KL divergence between old and new policies.

        KL(old || new) = E[log(old) - log(new)]

        Args:
            old_log_probs: Log probabilities from old policy
            new_log_probs: Log probabilities from new policy

        Returns:
            Mean KL divergence
        """
        kl = (old_log_probs - new_log_probs).mean().item()
        return kl

    def _adapt_clip_epsilon(self, kl_div: float):
        """
        Adapt clip_epsilon based on KL divergence.

        If KL too high (policy changing rapidly): reduce clip_epsilon → more conservative
        If KL too low (policy too cautious): increase clip_epsilon → allow larger updates

        Args:
            kl_div: Current KL divergence
        """
        if not self.adaptive_clip:
            return

        if kl_div > self.target_kl * 1.5:
            # Policy changing too fast, be more conservative
            self.clip_epsilon = max(0.05, self.clip_epsilon * 0.9)
        elif kl_div < self.target_kl * 0.5:
            # Policy too cautious, allow larger updates
            self.clip_epsilon = min(self.initial_clip_epsilon, self.clip_epsilon * 1.1)

    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform PPO update with adaptive exploration.

        Adds:
        - Entropy coefficient decay
        - KL divergence tracking
        - Adaptive clipping
        - Early stopping based on KL

        Returns:
            Dictionary of training metrics or None if buffer not full
        """
        if not self.rollout_buffer.full:
            return None

        # Update entropy coefficient based on training progress
        self._update_entropy_coef()

        # Get last value for bootstrapping
        last_obs = self._preprocess_obs(self.rollout_buffer.observations[-1])
        with torch.no_grad():
            last_value = self.policy.get_value(last_obs).item()

        # Compute advantages using GAE
        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        # Get all data from buffer
        for batch in self.rollout_buffer.get():
            obs, actions, old_log_probs, advantages, returns, old_values = batch

            # Normalize advantages
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Multiple epochs of updates
            all_policy_losses = []
            all_value_losses = []
            all_entropies = []
            all_clip_fractions = []
            all_kl_divs = []

            for epoch in range(self.n_epochs):
                # Create minibatches
                from src.utils.rollout_buffer import create_minibatches
                for minibatch in create_minibatches(
                    self.batch_size, obs, actions, old_log_probs,
                    advantages, returns, old_values
                ):
                    mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns, mb_old_values = minibatch

                    # Evaluate actions under current policy
                    new_log_probs, new_values, entropy = self.policy.evaluate_actions(mb_obs, mb_actions)

                    # Reshape values to match returns
                    new_values = new_values.squeeze(-1)

                    # Compute KL divergence
                    kl_div = self._compute_kl_divergence(mb_old_log_probs, new_log_probs)
                    all_kl_divs.append(kl_div)

                    # Early stopping if KL divergence too high
                    if self.kl_early_stop and kl_div > self.target_kl * 2.0:
                        print(f"  Early stopping at epoch {epoch+1}/{self.n_epochs} (KL={kl_div:.4f} > {self.target_kl*2.0:.4f})")
                        break

                    # ===== PPO Loss Components =====

                    # 1. Policy loss (clipped surrogate objective)
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # 2. Value loss (clipped to prevent large updates)
                    if self.value_clip is not None:
                        value_pred_clipped = mb_old_values + torch.clamp(
                            new_values - mb_old_values,
                            -self.value_clip,
                            self.value_clip
                        )
                        value_loss_unclipped = (new_values - mb_returns).pow(2)
                        value_loss_clipped = (value_pred_clipped - mb_returns).pow(2)
                        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    else:
                        value_loss = 0.5 * (new_values - mb_returns).pow(2).mean()

                    # 3. Entropy bonus (encourages exploration, decays over time)
                    entropy_loss = entropy.mean()

                    # Total loss (note: entropy_coef now decays)
                    loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_loss

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    # Track metrics
                    all_policy_losses.append(policy_loss.item())
                    all_value_losses.append(value_loss.item())
                    all_entropies.append(entropy_loss.item())

                    # Clip fraction (how often we clip the ratio)
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    all_clip_fractions.append(clip_fraction)

                # Check if early stopped
                if self.kl_early_stop and kl_div > self.target_kl * 2.0:
                    break

        # Adapt clip epsilon based on mean KL divergence
        if all_kl_divs:
            mean_kl = np.mean(all_kl_divs)
            self._adapt_clip_epsilon(mean_kl)
            self.kl_divergences.append(mean_kl)

        # Reset buffer for next rollout
        self.rollout_buffer.reset()
        self.n_updates += 1

        # Return metrics
        return {
            'policy_loss': np.mean(all_policy_losses),
            'value_loss': np.mean(all_value_losses),
            'entropy': np.mean(all_entropies),
            'clip_fraction': np.mean(all_clip_fractions),
            'kl_divergence': mean_kl if all_kl_divs else 0.0,
            'entropy_coef': self.entropy_coef,  # Track decay
            'clip_epsilon': self.clip_epsilon,    # Track adaptation
            'n_updates': self.n_updates
        }


if __name__ == "__main__":
    """Test the adaptive PPO agent."""
    print("Testing PPOAdaptiveAgent...")

    # Create agent
    agent = PPOAdaptiveAgent(
        observation_shape=(3, 64, 64),
        num_actions=17,
        device='cpu',
        n_steps=8,  # Small for testing
        entropy_coef_start=0.05,
        entropy_coef_end=0.001,
        total_timesteps=1000
    )

    print(f"Agent created successfully!")
    print(f"  Initial entropy coef: {agent.entropy_coef:.4f}")
    print(f"  Initial clip epsilon: {agent.clip_epsilon:.4f}")
    print(f"  Target KL: {agent.target_kl:.4f}")

    # Simulate training progress
    agent.training_step = 500  # Halfway
    agent._update_entropy_coef()
    print(f"\nAfter 500 steps (50% progress):")
    print(f"  Entropy coef: {agent.entropy_coef:.4f}")

    agent.training_step = 1000  # End
    agent._update_entropy_coef()
    print(f"\nAfter 1000 steps (100% progress):")
    print(f"  Entropy coef: {agent.entropy_coef:.4f}")

    print("\nPPOAdaptiveAgent test passed!")
