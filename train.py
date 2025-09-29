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

# TODO: Import custom agents once implemented
# from src.agents.ppo_agent import PPOAgent
# from src.agents.drqv2_agent import DrQv2Agent

parser = argparse.ArgumentParser(description='Train RL agents on Crafter environment')
parser.add_argument('--algorithm', type=str, choices=['ppo', 'drqv2'],
                   default='ppo', help='Algorithm to train (ppo or drqv2)')
parser.add_argument('--outdir', default='logdir/crafter')
parser.add_argument('--steps', type=float, default=1e6, help='Training steps (default: 1M)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--eval_freq', type=int, default=50000, help='Evaluation frequency')
args = parser.parse_args()

# Setup environment (using CrafterPartial-v1 as per assignment requirements)
register(id='CrafterPartial-v1', entry_point=crafter.Env)
env = old_gym.make('CrafterPartial-v1')

# Create output directory with algorithm and timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"{args.outdir}_{args.algorithm}_{timestamp}"
os.makedirs(outdir, exist_ok=True)

# Add recording wrapper for Crafter metrics
env = crafter.Recorder(env, outdir, save_stats=True, save_video=False, save_episode=False)
env = GymV21CompatibilityV0(env=env)

print(f"Training {args.algorithm.upper()} for {int(args.steps):,} steps")
print(f"Output directory: {outdir}")

# TODO(human): Choose algorithm based on argument
if args.algorithm == 'ppo':
    # Partner's work: Stable-baselines PPO
    model = stable_baselines3.PPO('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=int(args.steps))
elif args.algorithm == 'drqv2':
    # Anand's work: Custom DrQ-v2 implementation
    print("DrQ-v2 implementation coming soon...")
    # TODO: Initialize DrQv2Agent and train
else:
    raise ValueError(f"Unknown algorithm: {args.algorithm}")