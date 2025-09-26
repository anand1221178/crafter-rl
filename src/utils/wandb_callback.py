"""
W&B callback for Crafter training with comprehensive metric tracking.
Tracks achievements, survival stats, and learning progress in real-time.
"""

import json
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import pathlib


class WandbCrafterCallback(BaseCallback):
    """
    Custom W&B callback for Crafter experiments.
    Tracks achievements, survival metrics, and training progress.
    """

    def __init__(
        self,
        project_name: str = "crafter-rl",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_freq: int = 1000,
        eval_freq: int = 10000,
        save_model: bool = True,
        verbose: int = 0
    ):
        super().__init__(verbose)

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config or {}
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.save_model = save_model

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.achievement_history = {}
        self.survival_stats = []

        # Crafter achievement names (22 total)
        self.achievement_names = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]

        # Initialize W&B run
        self._init_wandb()

    def _init_wandb(self):
        """Initialize W&B run with comprehensive config."""
        wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=self.config,
            sync_tensorboard=True,  # Also sync tensorboard logs
            monitor_gym=True,       # Monitor gym environments
            save_code=True,         # Save source code
        )

        # Log system info
        wandb.run.log_code(".")

    def _on_training_start(self) -> None:
        """Called when training starts."""
        # Log model architecture and hyperparameters
        model_config = {
            'policy_type': self.model.policy.__class__.__name__,
            'algorithm': self.model.__class__.__name__,
            'n_envs': self.model.get_env().num_envs,
            'observation_space': str(self.model.observation_space),
            'action_space': str(self.model.action_space),
        }

        # Add algorithm-specific hyperparameters
        if hasattr(self.model, 'learning_rate'):
            model_config['learning_rate'] = self.model.learning_rate
        if hasattr(self.model, 'gamma'):
            model_config['gamma'] = self.model.gamma
        if hasattr(self.model, 'batch_size'):
            model_config['batch_size'] = self.model.batch_size

        wandb.config.update(model_config, allow_val_change=True)

    def _on_step(self) -> bool:
        """Called at each training step."""
        # Log basic training metrics every log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            self._log_training_metrics()

        # Check for episode completion and log episode metrics
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    self._log_episode_completion(i)

        return True

    def _log_training_metrics(self):
        """Log basic training progress metrics."""
        metrics = {
            'timesteps': self.num_timesteps,
            'episodes': len(self.episode_rewards) if self.episode_rewards else 0,
        }

        # Add loss metrics if available
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            for key, value in self.model.logger.name_to_value.items():
                if 'loss' in key.lower() or 'entropy' in key.lower():
                    metrics[f'train/{key}'] = value

        wandb.log(metrics, step=self.num_timesteps)

    def _log_episode_completion(self, env_idx: int = 0):
        """Log metrics when an episode completes."""
        infos = self.locals.get('infos', [])
        if env_idx < len(infos) and infos[env_idx]:
            info = infos[env_idx]

            # Episode metrics
            episode_reward = info.get('episode', {}).get('r', 0)
            episode_length = info.get('episode', {}).get('l', 0)

            if episode_reward and episode_length:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                episode_metrics = {
                    'episode/reward': episode_reward,
                    'episode/length': episode_length,
                    'episode/reward_mean': np.mean(self.episode_rewards[-100:]),
                    'episode/length_mean': np.mean(self.episode_lengths[-100:]),
                    'episode/count': len(self.episode_rewards)
                }

                # Extract and log Crafter-specific achievements
                achievements = info.get('achievements', {})
                episode_achievements = {}

                for achievement_name in self.achievement_names:
                    key = f'achievement_{achievement_name}'
                    if key in achievements:
                        # Check if achievement was unlocked (value >= 1)
                        unlocked = achievements[key] >= 1
                        episode_achievements[f'achievements/{achievement_name}'] = int(unlocked)

                        # Track in history for running averages
                        if achievement_name not in self.achievement_history:
                            self.achievement_history[achievement_name] = []
                        self.achievement_history[achievement_name].append(unlocked)

                        # Calculate success rate over last 100 episodes
                        recent_history = self.achievement_history[achievement_name][-100:]
                        success_rate = sum(recent_history) / len(recent_history) * 100
                        episode_achievements[f'achievements/{achievement_name}_rate'] = success_rate

                # Log achievement metrics
                if episode_achievements:
                    episode_metrics.update(episode_achievements)

                    # Calculate total achievements unlocked this episode
                    total_unlocked = sum(1 for key, val in episode_achievements.items()
                                       if key.startswith('achievements/') and not key.endswith('_rate') and val == 1)
                    episode_metrics['achievements/total_unlocked'] = total_unlocked

                wandb.log(episode_metrics, step=self.num_timesteps)

    def _log_achievement_metrics(self, achievements: Dict[str, Any]):
        """
        Log Crafter achievement statistics.

        Args:
            achievements: Dictionary containing achievement data from environment
        """
        # TODO(human): Implement achievement tracking logic
        # This should:
        # 1. Track individual achievement unlock counts
        # 2. Calculate success rates over recent episodes
        # 3. Compute the geometric mean score (official Crafter metric)
        # 4. Log achievement progress charts
        pass

    def _create_achievement_plot(self) -> Figure:
        """Create a plot showing achievement success rates."""
        if not self.achievement_history:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot achievement success rates
        achievement_rates = []
        labels = []

        for achievement in self.achievement_names:
            if achievement in self.achievement_history:
                rate = np.mean(self.achievement_history[achievement][-100:]) * 100
                achievement_rates.append(rate)
                labels.append(achievement.replace('_', ' ').title())

        if achievement_rates:
            y_pos = np.arange(len(labels))
            bars = ax.barh(y_pos, achievement_rates)

            # Color bars by success rate
            for i, (bar, rate) in enumerate(zip(bars, achievement_rates)):
                if rate > 75:
                    bar.set_color('green')
                elif rate > 50:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Success Rate (%)')
            ax.set_title('Crafter Achievement Success Rates (Last 100 Episodes)')
            ax.set_xlim(0, 100)

            # Add value labels on bars
            for i, rate in enumerate(achievement_rates):
                ax.text(rate + 1, i, f'{rate:.1f}%', va='center')

        plt.tight_layout()
        return Figure(fig, close=True)

    def _on_training_end(self) -> None:
        """Called when training ends."""
        # Save final model if requested
        if self.save_model:
            model_path = f"models/{wandb.run.name}_final.zip"
            self.model.save(model_path)
            wandb.save(model_path)

        # Log final summary
        if self.episode_rewards:
            final_metrics = {
                'final/total_episodes': len(self.episode_rewards),
                'final/mean_reward': np.mean(self.episode_rewards),
                'final/mean_length': np.mean(self.episode_lengths),
                'final/best_reward': np.max(self.episode_rewards),
                'final/final_reward': self.episode_rewards[-1],
            }
            wandb.log(final_metrics)

        # Create and log final achievement plot
        achievement_plot = self._create_achievement_plot()
        if achievement_plot:
            wandb.log({"achievements/final_success_rates": achievement_plot})

        wandb.finish()


def setup_wandb_config(
    algorithm: str,
    env_type: str,
    steps: int,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a comprehensive W&B config for Crafter experiments.

    Args:
        algorithm: RL algorithm name
        env_type: Crafter environment variant
        steps: Total training steps
        **kwargs: Additional hyperparameters

    Returns:
        Configuration dictionary for W&B
    """
    config = {
        # Experiment details
        'algorithm': algorithm,
        'environment': f'Crafter{env_type.title()}-v1',
        'total_timesteps': steps,
        'framework': 'stable-baselines3',

        # Environment settings
        'obs_space': '64x64x3 RGB',
        'action_space': '17 discrete actions',
        'achievements': 22,

        # Evaluation metrics
        'primary_metric': 'crafter_score',
        'secondary_metrics': ['achievement_unlock_rate', 'survival_time', 'cumulative_reward'],
    }

    # Add any additional hyperparameters
    config.update(kwargs)

    return config