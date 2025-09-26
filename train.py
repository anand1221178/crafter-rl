import argparse
import os
from datetime import datetime
import gym as old_gym
import gymnasium
import stable_baselines3
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import crafter
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register
import wandb

from src.environment.crafter_env import create_crafter_env
from src.utils.wandb_callback import WandbCrafterCallback, setup_wandb_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent on Crafter')
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['ppo', 'dqn', 'sac'],
                       help='RL algorithm to use')
    parser.add_argument('--env_type', type=str, default='partial',
                       choices=['partial', 'reward', 'noreward'],
                       help='Crafter environment type')
    parser.add_argument('--outdir', type=str, default=None,
                       help='Output directory for logs')
    parser.add_argument('--steps', type=float, default=1e6,
                       help='Total training steps')
    parser.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluation frequency')
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Model checkpoint save frequency')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for training')
    parser.add_argument('--wandb_project', type=str, default='crafter-rl',
                       help='W&B project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for W&B')
    parser.add_argument('--wandb_tags', nargs='+', default=None,
                       help='Tags for W&B experiment')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable W&B logging')
    return parser.parse_args()


def create_log_dir(algorithm, env_type):
    """Create a unique log directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"results/logs/{algorithm}_{env_type}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def train():
    args = parse_args()

    # Set up logging directory
    if args.outdir is None:
        log_dir = create_log_dir(args.algorithm, args.env_type)
    else:
        log_dir = args.outdir
        os.makedirs(log_dir, exist_ok=True)

    print(f"Logging to: {log_dir}")

    # Set up experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.experiment_name = f"{args.algorithm}_{args.env_type}_{timestamp}"

    # Create training environment
    train_env = create_crafter_env(
        env_type=args.env_type,
        logdir=os.path.join(log_dir, 'train'),
        save_stats=True,
        save_video=False,
        save_episode=False
    )

    # Create evaluation environment (no video recording to avoid metadata issues)
    eval_env = create_crafter_env(
        env_type=args.env_type,
        logdir=None,  # Disable recording for eval to avoid render issues
        save_stats=False,
        save_video=False,
        save_episode=False
    )

    # Wrap environments with Monitor for logging
    train_env = Monitor(train_env, os.path.join(log_dir, 'train_monitor'))
    eval_env = Monitor(eval_env, os.path.join(log_dir, 'eval_monitor'))

    # Select algorithm
    if args.algorithm == 'ppo':
        model = PPO(
            'CnnPolicy',
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=os.path.join(log_dir, 'tensorboard'),
            device=args.device,
            seed=args.seed
        )
    elif args.algorithm == 'dqn':
        model = DQN(
            'CnnPolicy',
            train_env,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=10000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=os.path.join(log_dir, 'tensorboard'),
            device=args.device,
            seed=args.seed
        )
    elif args.algorithm == 'sac':
        # Note: SAC typically works with continuous actions
        # For discrete actions, we might need to use a different approach
        # or modify the action space
        print("Warning: SAC is designed for continuous actions.")
        print("Using SAC with discrete actions may require modifications.")
        model = SAC(
            'CnnPolicy',
            train_env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=os.path.join(log_dir, 'tensorboard'),
            device=args.device,
            seed=args.seed
        )

    # Set up W&B configuration
    wandb_config = setup_wandb_config(
        algorithm=args.algorithm,
        env_type=args.env_type,
        steps=int(args.steps),
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        device=args.device,
        seed=args.seed,
        log_dir=log_dir
    )

    # Set up callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix=f'{args.algorithm}_model',
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback (reduced frequency for testing)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, 'best_model'),
        log_path=os.path.join(log_dir, 'eval_results'),
        eval_freq=max(args.eval_freq, 5000),  # At least 5000 steps between evals
        deterministic=True,
        render=False,
        n_eval_episodes=5  # Fewer episodes for faster eval
    )
    callbacks.append(eval_callback)

    # W&B callback (if enabled)
    if not args.no_wandb:
        wandb_callback = WandbCrafterCallback(
            project_name=args.wandb_project,
            experiment_name=args.experiment_name,
            config=wandb_config,
            log_freq=1000,
            eval_freq=args.eval_freq,
            save_model=True
        )
        callbacks.append(wandb_callback)
        print(f"W&B tracking enabled: {args.wandb_project}/{args.experiment_name}")
    else:
        print("W&B tracking disabled")

    # Combine all callbacks
    callback_list = CallbackList(callbacks)

    # Train the model
    print(f"Starting training with {args.algorithm} for {args.steps} steps...")
    model.learn(
        total_timesteps=int(args.steps),
        callback=callback_list,
        progress_bar=True
    )

    # Save final model
    final_model_path = os.path.join(log_dir, f'{args.algorithm}_final_model')
    model.save(final_model_path)
    print(f"Training complete! Final model saved to: {final_model_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()