import argparse
import os
from datetime import datetime
import gym as old_gym
import stable_baselines3
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import crafter
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register

# Import custom agents
# from src.agents.ppo_agent import PPOAgent  # Partner's work
from src.agents.drqv2_agent import DrQv2Agent  # Anand's work

# Additional imports for DrQ-v2 training
import torch
import numpy as np
import time
from collections import defaultdict

parser = argparse.ArgumentParser(description='Train RL agents on Crafter environment')
parser.add_argument('--algorithm', type=str, choices=['ppo', 'drqv2'],
                   default='ppo', help='Algorithm to train (ppo or drqv2)')
parser.add_argument('--outdir', default='logdir/crafter')
parser.add_argument('--steps', type=float, default=1e6, help='Training steps (default: 1M)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--eval_freq', type=int, default=50000, help='Evaluation frequency')
args = parser.parse_args()

# Simple wrapper to handle Gym API differences
class CrafterWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        obs = self.env.reset()
        # Always return just observation (not tuple)
        if isinstance(obs, tuple):
            return obs[0]
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Ensure done is a Python bool (handle different types)
        if hasattr(done, 'item'):  # NumPy scalar
            done = bool(done.item())
        else:
            done = bool(done)
        return obs, reward, done, info

    def close(self):
        self.env.close()

# Setup environment - bypass Gym entirely for cleaner setup
# Create output directory with algorithm and timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"{args.outdir}_{args.algorithm}_{timestamp}"
os.makedirs(outdir, exist_ok=True)

# Create Crafter environment directly
base_env = crafter.Env()

# Add recording wrapper for Crafter metrics
recorded_env = crafter.Recorder(base_env, outdir, save_stats=True, save_video=False, save_episode=False)

# Apply our simple wrapper to handle API differences
env = CrafterWrapper(recorded_env)

print(f"Training {args.algorithm.upper()} for {int(args.steps):,} steps")
print(f"Output directory: {outdir}")

# TODO(human): Choose algorithm based on argument
if args.algorithm == 'ppo':
    # Partner's work: Stable-baselines PPO
    print("🚀 Starting PPO training...")

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create PPO model with device specification
    model = stable_baselines3.PPO(
        'CnnPolicy',
        env,
        verbose=1,
        device=device,
        # Optimized hyperparameters for Crafter
        learning_rate=3e-4,
        n_steps=2048,  # Steps per env before update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=outdir
    )

    print(f"\n📊 PPO Configuration:")
    print(f"  Total steps: {int(args.steps):,}")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Batch size: {model.batch_size}")
    print(f"  Device: {device}")
    print(f"\n" + "="*50)
    print("Starting training loop...")
    print("="*50 + "\n")

    # Train the model
    model.learn(total_timesteps=int(args.steps))

    # Save the final model
    model_path = os.path.join(outdir, 'ppo_final.zip')
    model.save(model_path)

    print(f"\n" + "="*50)
    print("PPO Training Complete!")
    print(f"  Model saved: {model_path}")
    print(f"  Results saved: {outdir}")
    print("="*50)
elif args.algorithm == 'drqv2':
    # Anand's work: Custom DrQ-v2 implementation
    print("🚀 Starting DrQ-v2 training...")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Detect device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get environment information
    obs = env.reset()  # Our wrapper handles API differences
    observation_shape = obs.shape  # Should be (64, 64, 3)
    num_actions = env.action_space.n  # Should be 17

    print(f"Environment setup:")
    print(f"  Observation shape: {observation_shape}")
    print(f"  Number of actions: {num_actions}")

    # Initialize DrQ-v2 agent
    agent = DrQv2Agent(
        observation_shape=observation_shape,
        num_actions=num_actions,
        device=device,
        # Hyperparameters optimized for Crafter
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.05,  # Higher final epsilon for continued exploration
        epsilon_decay_steps=750_000,  # Explore for 75% of training (was 10%!)
        tau=0.01,
        replay_buffer_size=100_000,
        min_replay_size=1000
    )

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    episode_count = 0
    training_start_time = time.time()

    # Logging frequency
    log_freq = 1000  # Log every 1000 steps
    save_freq = 50000  # Save model every 50k steps

    print(f"\n📊 Training Configuration:")
    print(f"  Total steps: {int(args.steps):,}")
    print(f"  Log frequency: {log_freq:,} steps")
    print(f"  Save frequency: {save_freq:,} steps")
    print(f"  Min replay size: {agent.min_replay_size:,}")
    print(f"\n" + "="*50)
    print("Starting training loop...")

    # Training loop
    obs = env.reset()
    episode_reward = 0
    episode_step = 0

    while total_steps < args.steps:
        # Select action using epsilon-greedy policy
        action = agent.act(obs, training=True)

        # Take step in environment
        next_obs, reward, done, info = env.step(action)

        # Store experience in replay buffer
        agent.store_experience(obs, action, reward, next_obs, done)

        # Update metrics
        episode_reward += reward
        episode_step += 1
        total_steps += 1

        # Update the agent (if enough data in replay buffer)
        update_metrics = agent.update()

        # Skip if agent couldn't update (not enough data yet)
        if update_metrics is None:
            update_metrics = {}

        # Handle episode end
        if done:
            # Track episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_step)
            agent.end_episode(episode_reward, episode_step)
            episode_count += 1

            # Reset for next episode
            obs = env.reset()
            episode_reward = 0
            episode_step = 0

            # Log episode results
            if episode_count % 10 == 0:  # Every 10 episodes
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"Episode {episode_count:4d} | "
                      f"Steps: {total_steps:7,} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Length: {avg_length:6.1f} | "
                      f"ε: {agent.epsilon:.3f}")
        else:
            obs = next_obs

        # Periodic logging
        if total_steps % log_freq == 0 and update_metrics is not None:
            elapsed_time = time.time() - training_start_time
            steps_per_sec = total_steps / elapsed_time

            print(f"\n📊 Training Progress (Step {total_steps:,}):")
            print(f"  Time elapsed: {elapsed_time/60:.1f} min")
            print(f"  Steps/sec: {steps_per_sec:.1f}")
            print(f"  Episodes: {episode_count}")
            print(f"  Q-loss: {update_metrics.get('q_loss', 0):.4f}")
            print(f"  Epsilon: {update_metrics.get('epsilon', 0):.3f}")
            print(f"  Replay buffer: {update_metrics.get('replay_buffer_size', 0):,}")

            if episode_rewards:
                print(f"  Recent avg reward: {np.mean(episode_rewards[-10:]):.2f}")
                print(f"  Recent avg length: {np.mean(episode_lengths[-10:]):.1f}")

        # Periodic model saving
        if total_steps % save_freq == 0:
            model_path = os.path.join(outdir, f'drqv2_step_{total_steps}.pt')
            agent.save(model_path)
            print(f"💾 Model saved at step {total_steps:,}")

    # Final save
    final_model_path = os.path.join(outdir, 'drqv2_final.pt')
    agent.save(final_model_path)

    # Training summary
    total_time = time.time() - training_start_time
    print(f"\n" + "="*50)
    print(f" DrQ-v2 Training Complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Average reward: {np.mean(episode_rewards):.2f}")
    print(f"  Final epsilon: {agent.epsilon:.3f}")
    print(f"  Final model: {final_model_path}")
    print(f"  Results saved to: {outdir}")

    # Close environment
    env.close()
else:
    raise ValueError(f"Unknown algorithm: {args.algorithm}")