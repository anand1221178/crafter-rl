#!/usr/bin/env python3
"""
Evaluation script for trained Crafter agents.
Uses the official Crafter analysis tools.
"""

import argparse
import pathlib
from src.evaluation.evaluator import CrafterEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Crafter agent performance')
    parser.add_argument('--log_dirs', nargs='+', required=True,
                       help='Directories containing training logs (with stats.jsonl)')
    parser.add_argument('--methods', nargs='+', required=True,
                       help='Method names for each log directory')
    parser.add_argument('--budget', type=int, default=int(1e6),
                       help='Maximum steps to evaluate (default: 1M)')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--report_name', type=str, default='evaluation_report.md',
                       help='Name of the evaluation report file')
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.log_dirs) != len(args.methods):
        raise ValueError("Number of log directories must match number of method names")

    # Create evaluator
    evaluator = CrafterEvaluator(args.output_dir)

    # Evaluate each agent
    agent_results = []
    print("Evaluating agents...")
    print("=" * 50)

    for log_dir, method in zip(args.log_dirs, args.methods):
        print(f"\nEvaluating {method} from {log_dir}...")

        if not pathlib.Path(log_dir).exists():
            print(f"Warning: Directory {log_dir} does not exist, skipping...")
            continue

        result = evaluator.evaluate_single_agent(log_dir, method, args.budget)

        if result:
            agent_results.append(result)
            print(f"✓ {method}: Score = {result['crafter_score']:.2f}, "
                  f"Avg Reward = {result['avg_reward']:.2f}")
        else:
            print(f"✗ {method}: Failed to evaluate")

    if not agent_results:
        print("No agents could be evaluated. Please check your log directories.")
        return

    print(f"\n{len(agent_results)} agents evaluated successfully!")

    # Generate comparison
    if len(agent_results) > 1:
        print("\nGenerating comparison plots...")
        evaluator.plot_achievement_comparison(agent_results)
        evaluator.plot_training_curves(args.log_dirs, args.methods)
        print("✓ Plots saved to results directory")

    # Generate report
    print(f"\nGenerating evaluation report...")
    report_path = evaluator.generate_report(agent_results, args.report_name)
    print(f"✓ Report saved to {report_path}")

    # Print summary to console
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    for result in sorted(agent_results, key=lambda x: x['crafter_score'], reverse=True):
        print(f"{result['method']:<15} | Score: {result['crafter_score']:6.2f} | "
              f"Reward: {result['avg_reward']:6.2f} | "
              f"Episodes: {result['total_episodes']:4d}")

    if len(agent_results) > 1:
        best = max(agent_results, key=lambda x: x['crafter_score'])
        print(f"\n🏆 Best performing agent: {best['method']} "
              f"(Score: {best['crafter_score']:.2f})")


if __name__ == "__main__":
    main()