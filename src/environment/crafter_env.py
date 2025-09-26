import warnings
import gym as old_gym
import gymnasium
import numpy as np
import crafter
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register
from typing import Optional, Dict, Any, Tuple

# Suppress the Gym deprecation warning since we're using Gymnasium
warnings.filterwarnings("ignore", category=UserWarning, message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overriding environment.*")


def create_crafter_env(
    env_type: str = 'partial',
    logdir: Optional[str] = None,
    save_stats: bool = True,
    save_video: bool = False,
    save_episode: bool = False
) -> gymnasium.Env:
    """
    Create a Crafter environment compatible with Gymnasium.

    Args:
        env_type: Type of environment ('partial', 'reward', 'noreward')
        logdir: Directory to save logs (if None, no logging)
        save_stats: Whether to save statistics
        save_video: Whether to save videos
        save_episode: Whether to save full episodes

    Returns:
        Gymnasium-compatible Crafter environment
    """

    # Register Crafter environments
    try:
        register(id='CrafterPartial-v1', entry_point=crafter.Env)
        register(id='CrafterReward-v1', entry_point=crafter.Env,
                kwargs=dict(reward=True))
        register(id='CrafterNoReward-v1', entry_point=crafter.Env,
                kwargs=dict(reward=False))
    except:
        pass  # Already registered

    # Select environment name
    env_map = {
        'partial': 'CrafterPartial-v1',
        'reward': 'CrafterReward-v1',
        'noreward': 'CrafterNoReward-v1'
    }
    env_name = env_map.get(env_type, 'CrafterPartial-v1')

    # Create the old gym environment
    env = old_gym.make(env_name)

    # Add recorder if logdir is specified
    if logdir:
        env = crafter.Recorder(
            env, logdir,
            save_stats=save_stats,
            save_video=save_video,
            save_episode=save_episode,
        )

    # Convert to Gymnasium using shimmy
    env = GymV21CompatibilityV0(env=env)

    return env


class CrafterEnvWrapper(gymnasium.Wrapper):
    """
    Additional wrapper for Crafter environment with preprocessing options.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        preprocess_obs: bool = False,
        grayscale: bool = False,
        frame_stack: int = 1,
        normalize: bool = False
    ):
        """
        Initialize wrapper.

        Args:
            env: Base environment
            preprocess_obs: Whether to preprocess observations
            grayscale: Convert to grayscale
            frame_stack: Number of frames to stack
            normalize: Normalize observations to [0, 1]
        """
        super().__init__(env)
        self.preprocess_obs = preprocess_obs
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        self.normalize = normalize

        # Update observation space if preprocessing
        if self.preprocess_obs:
            obs_shape = env.observation_space.shape

            if self.grayscale:
                # Grayscale reduces channels to 1
                new_shape = (obs_shape[0], obs_shape[1], frame_stack)
            else:
                # RGB with frame stacking
                new_shape = (obs_shape[0], obs_shape[1], obs_shape[2] * frame_stack)

            if self.normalize:
                self.observation_space = gymnasium.spaces.Box(
                    low=0.0, high=1.0, shape=new_shape, dtype=np.float32
                )
            else:
                self.observation_space = gymnasium.spaces.Box(
                    low=0, high=255, shape=new_shape, dtype=np.uint8
                )

        # Frame buffer for stacking
        self.frame_buffer = []

    def reset(self, **kwargs):
        """Reset environment and frame buffer."""
        obs, info = self.env.reset(**kwargs)

        # Clear frame buffer
        self.frame_buffer = []

        # Initialize frame buffer
        for _ in range(self.frame_stack):
            self.frame_buffer.append(self._preprocess_frame(obs))

        return self._get_stacked_obs(), info

    def step(self, action):
        """Step environment and update frame buffer."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update frame buffer
        self.frame_buffer.append(self._preprocess_frame(obs))
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer.pop(0)

        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _preprocess_frame(self, frame):
        """Preprocess a single frame."""
        if not self.preprocess_obs:
            return frame

        if self.grayscale:
            # Convert RGB to grayscale
            frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
            frame = frame.astype(np.uint8)

        if self.normalize:
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0

        return frame

    def _get_stacked_obs(self):
        """Get stacked observation from frame buffer."""
        if not self.preprocess_obs or self.frame_stack == 1:
            return self.frame_buffer[-1] if self.frame_buffer else None

        # Stack frames
        if self.grayscale:
            # Stack grayscale frames along channel dimension
            return np.stack(self.frame_buffer, axis=-1)
        else:
            # Concatenate RGB frames along channel dimension
            return np.concatenate(self.frame_buffer, axis=-1)