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
from src.agents.dynaq_agent import DynaQAgent  # Dyna-Q (external algorithm)

# Additional imports for training
import torch
import numpy as np
import time
from collections import defaultdict

parser = argparse.ArgumentParser(description='Train RL agents on Crafter environment')
parser.add_argument('--algorithm', type=str, choices=['ppo', 'dqn', 'dynaq'],
                   default='ppo', help='Algorithm to train (ppo, dqn, or dynaq)')
parser.add_argument('--outdir', default='logdir/crafter')
parser.add_argument('--steps', type=float, default=1e6, help='Training steps (default: 1M)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--eval_freq', type=int, default=50000, help='Evaluation frequency')
# Dyna-Q specific arguments
parser.add_argument('--planning_steps', type=int, default=5, help='Planning steps per real step (Dyna-Q only)')
parser.add_argument('--prioritized', action='store_true', help='Use prioritized sweeping (Dyna-Q only)')
parser.add_argument('--exploration_bonus', type=float, default=0.0, help='Exploration bonus κ for Dyna-Q+')
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
elif args.algorithm == 'dqn':
    # Partner's work: Vanilla DQN using Stable-Baselines3
    print("🚀 Starting DQN training...")

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create DQN model
    model = stable_baselines3.DQN(
        'CnnPolicy',
        env,
        verbose=1,
        device=device,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.75,  # Explore for 75% of training
        exploration_final_eps=0.05,
        tensorboard_log=outdir
    )

    print(f"\n📊 DQN Configuration:")
    print(f"  Total steps: {int(args.steps):,}")
    print(f"  Device: {device}")
    print(f"\n" + "="*50)
    print("Starting training loop...")
    print("="*50 + "\n")

    # Train the model
    model.learn(total_timesteps=int(args.steps))

    # Save the final model
    model_path = os.path.join(outdir, 'dqn_final.zip')
    model.save(model_path)

    print(f"\n" + "="*50)
    print("DQN Training Complete!")
    print(f"  Model saved: {model_path}")
    print(f"  Results saved: {outdir}")
    print("="*50)

elif args.algorithm == 'dynaq':
    # Anand's work: Dyna-Q (model-based RL)
    print("🚀 Starting Dyna-Q training...")
    print(f"📚 Algorithm: Dyna-Q (Sutton & Barto, 2018, Ch. 8)")
    print(f"🎯 Planning steps: {args.planning_steps} per real step")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Detect device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get environment information
    obs = env.reset()
    observation_shape = obs.shape  # Should be (64, 64, 3)
    num_actions = env.action_space.n  # Should be 17

    print(f"Environment setup:")
    print(f"  Observation shape: {observation_shape}")
    print(f"  Number of actions: {num_actions}")

    # Initialize Dyna-Q agent with improved baseline hyperparameters
    agent = DynaQAgent(
        observation_shape=observation_shape,
        num_actions=num_actions,
        device=device,
        # Q-learning hyperparameters (improved baseline)
        learning_rate=3e-4,  # 3x higher for faster learning
        gamma=0.99,
        batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.1,  # Keep 10% exploration at end
        epsilon_decay_steps=900_000,  # Explore for 90% of training
        tau=0.005,  # Slower target updates
        replay_buffer_size=100_000,
        min_replay_size=5000,  # More initial exploration
        # Dyna-Q specific
        planning_steps=args.planning_steps,
        model_capacity=50_000,
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

    print(f"\n📊 Dyna-Q Training Configuration:")
    print(f"  Total steps: {int(args.steps):,}")
    print(f"  Planning steps per real step: {args.planning_steps}")
    print(f"  Model capacity: 50,000 transitions")
    print(f"  Prioritized sweeping: {args.prioritized}")
    print(f"  Exploration bonus: {args.exploration_bonus}")
    print(f"  Log frequency: {log_freq:,} steps")
    print(f"  Save frequency: {save_freq:,} steps")
    print(f"\n" + "="*50)
    print("Starting Dyna-Q training loop...")
    print("="*50 + "\n")

    # Training loop
    obs = env.reset()
    episode_reward = 0
    episode_step = 0

    while total_steps < args.steps:
        # Select action using epsilon-greedy policy
        action = agent.act(obs, training=True)

        # Take step in environment
        next_obs, reward, done, info = env.step(action)

        # Store experience in replay buffer AND world model
        agent.store_experience(obs, action, reward, next_obs, done)

        # Update metrics
        episode_reward += reward
        episode_step += 1
        total_steps += 1

        # Update the agent (direct RL + planning)
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
                      f"ε: {agent.epsilon:.3f} | "
                      f"Model: {len(agent.world_model):5,}")
        else:
            obs = next_obs

        # Periodic logging
        if total_steps % log_freq == 0 and update_metrics:
            elapsed_time = time.time() - training_start_time
            steps_per_sec = total_steps / elapsed_time

            print(f"\n📊 Training Progress (Step {total_steps:,}):")
            print(f"  Time elapsed: {elapsed_time/60:.1f} min")
            print(f"  Steps/sec: {steps_per_sec:.1f}")
            print(f"  Episodes: {episode_count}")
            print(f"  Q-loss: {update_metrics.get('q_loss', 0):.4f}")
            print(f"  Planning loss: {update_metrics.get('planning_loss', 0):.4f}")
            print(f"  Epsilon: {update_metrics.get('epsilon', 0):.3f}")
            print(f"  Replay buffer: {update_metrics.get('replay_buffer_size', 0):,}")
            print(f"  World model transitions: {update_metrics.get('model_size', 0):,}")
            print(f"  World model states: {update_metrics.get('model_num_states', 0):,}")

            if episode_rewards:
                print(f"  Recent avg reward: {np.mean(episode_rewards[-10:]):.2f}")
                print(f"  Recent avg length: {np.mean(episode_lengths[-10:]):.1f}")

        # Periodic model saving
        if total_steps % save_freq == 0:
            model_path = os.path.join(outdir, f'dynaq_step_{total_steps}.pt')
            agent.save(model_path)
            print(f"💾 Model saved at step {total_steps:,}")

    # Final save
    final_model_path = os.path.join(outdir, 'dynaq_final.pt')
    agent.save(final_model_path)

    # Training summary
    total_time = time.time() - training_start_time
    print(f"\n" + "="*50)
    print(f"✅ Dyna-Q Training Complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Average reward: {np.mean(episode_rewards):.2f}")
    print(f"  Final epsilon: {agent.epsilon:.3f}")
    print(f"  World model size: {len(agent.world_model):,} transitions")
    print(f"  Final model: {final_model_path}")
    print(f"  Results saved to: {outdir}")
    print("="*50)

    # Close environment
    env.close()
else:
    raise ValueError(f"Unknown algorithm: {args.algorithm}")