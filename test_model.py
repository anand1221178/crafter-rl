#!/usr/bin/env python3
"""
Simple Model Testing Script

Quick way to test your trained models with minimal setup.
This is a simplified version of evaluate.py for quick testing.

Usage:
    python test_model.py models/ppo_model.zip ppo
    python test_model.py models/dqn_model.zip dqn
    python test_model.py models/dynaq_model.pt dynaq
"""

import sys
import os
import argparse
from datetime import datetime

# Import the full evaluator
from evaluate import CrafterEvaluator

def quick_test(model_path, algorithm, episodes=10):
    """Quick model test with minimal episodes."""
    print(f"🚀 Quick testing {algorithm.upper()} model...")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")

    # Create evaluator
    evaluator = CrafterEvaluator(
        algorithm=algorithm,
        episodes=episodes,
        budget=1e6
    )

    # Create quick test output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"quick_test_{algorithm}_{timestamp}"

    try:
        # Run evaluation
        evaluation_dir = evaluator.evaluate_model(model_path, outdir)
        results = evaluator.analyze_results(evaluation_dir)

        print(f"\n✅ Quick test complete!")
        print(f"🏆 Crafter Score: {results['score']:.2f}%")
        print(f"📊 Average Reward: {results['reward']:.2f}")
        print(f"⏱️  Average Length: {results['length']:.2f}")

        # Show top 5 achievements
        achievements = sorted(results['achievements'].items(),
                            key=lambda x: x[1], reverse=True)[:5]
        print(f"\n🏆 Top 5 Achievements:")
        for task, rate in achievements:
            name = task[len('achievement_'):].replace('_', ' ').title()
            print(f"  {name:<15} {rate:6.2f}%")

        return results

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_model.py <model_path> <algorithm> [episodes]")
        print("Example: python test_model.py models/ppo_model.zip ppo 10")
        sys.exit(1)

    model_path = sys.argv[1]
    algorithm = sys.argv[2]
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    if algorithm not in ['ppo', 'dqn', 'dynaq']:
        print(f"❌ Unknown algorithm: {algorithm}")
        print("Supported: ppo, dqn, dynaq")
        sys.exit(1)

    quick_test(model_path, algorithm, episodes)

if __name__ == '__main__':
    main()