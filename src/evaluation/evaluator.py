import json
import pathlib
import collections
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from . import common


class CrafterEvaluator:
    """
    Evaluation framework for Crafter agents using the official analysis tools.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = pathlib.Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

    def load_agent_stats(self, log_dir: str, method_name: str, budget: int = int(1e6)) -> Optional[Dict]:
        """
        Load stats from a single agent's training run.

        Args:
            log_dir: Directory containing stats.jsonl file
            method_name: Name of the method/algorithm
            budget: Maximum number of steps to consider

        Returns:
            Dictionary containing processed stats or None if incomplete
        """
        log_path = pathlib.Path(log_dir)
        stats_file = log_path / "stats.jsonl"

        if not stats_file.exists():
            print(f"Warning: No stats.jsonl found in {log_dir}")
            return None

        rewards, lengths, achievements = self._load_stats_file(stats_file, budget)

        if sum(lengths) < budget - 1e4:
            print(f"Warning: Incomplete run in {log_dir} ({sum(lengths)} < {budget} steps)")
            return None

        return dict(
            method=method_name,
            seed="0",  # Single run for now
            xs=np.cumsum(lengths).tolist(),
            reward=rewards,
            length=lengths,
            **achievements,
        )

    def _load_stats_file(self, filename: pathlib.Path, budget: int) -> Tuple[List, List, Dict]:
        """Load and process a stats.jsonl file."""
        steps = 0
        rewards = []
        lengths = []
        achievements = collections.defaultdict(list)

        for line in filename.read_text().split('\n'):
            if not line.strip():
                continue
            episode = json.loads(line)
            steps += episode['length']
            if steps > budget:
                break

            lengths.append(episode['length'])
            for key, value in episode.items():
                if key.startswith('achievement_'):
                    achievements[key].append(value)

            # Compute reward (unlocks + health penalty)
            unlocks = int(np.sum([(v[-1] >= 1) for v in achievements.values()]))
            health = -0.9  # Health penalty as in original
            rewards.append(unlocks + health)

        return rewards, lengths, dict(achievements)

    def evaluate_single_agent(self, log_dir: str, method_name: str, budget: int = int(1e6)) -> Dict:
        """
        Evaluate a single agent and return metrics.

        Args:
            log_dir: Directory containing training logs
            method_name: Name of the algorithm
            budget: Maximum steps to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        run_data = self.load_agent_stats(log_dir, method_name, budget)
        if run_data is None:
            return {}

        # Compute success rates and scores
        percents, methods, seeds, tasks = common.compute_success_rates([run_data], budget)
        scores = common.compute_scores(percents)

        # Extract individual metrics
        episodes = len(run_data['length'])
        avg_reward = np.mean(run_data['reward'])
        avg_length = np.mean(run_data['length'])
        crafter_score = float(np.squeeze(scores))

        # Achievement success rates
        achievement_rates = {}
        for i, task in enumerate(tasks):
            task_name = task[len('achievement_'):].replace('_', ' ').title()
            achievement_rates[task_name] = float(np.squeeze(percents[0, 0, i]))

        return {
            'method': method_name,
            'crafter_score': crafter_score,
            'avg_reward': avg_reward,
            'avg_episode_length': avg_length,
            'total_episodes': episodes,
            'achievement_rates': achievement_rates,
            'raw_data': run_data
        }

    def compare_agents(self, agent_results: List[Dict]) -> Dict:
        """
        Compare multiple agents and generate comparison metrics.

        Args:
            agent_results: List of results from evaluate_single_agent

        Returns:
            Comparison statistics
        """
        if len(agent_results) < 2:
            return {}

        comparison = {
            'methods': [r['method'] for r in agent_results],
            'scores': [r['crafter_score'] for r in agent_results],
            'rewards': [r['avg_reward'] for r in agent_results],
            'episode_lengths': [r['avg_episode_length'] for r in agent_results],
        }

        # Find best performing agent
        best_idx = np.argmax(comparison['scores'])
        comparison['best_agent'] = {
            'method': comparison['methods'][best_idx],
            'score': comparison['scores'][best_idx]
        }

        return comparison

    def plot_achievement_comparison(self, agent_results: List[Dict], save_path: Optional[str] = None):
        """Plot achievement success rates for multiple agents."""
        if not agent_results:
            return

        # Get all unique achievements
        all_achievements = set()
        for result in agent_results:
            all_achievements.update(result['achievement_rates'].keys())
        all_achievements = sorted(all_achievements)

        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(all_achievements))
        width = 0.8 / len(agent_results)

        for i, result in enumerate(agent_results):
            rates = [result['achievement_rates'].get(ach, 0) for ach in all_achievements]
            ax.bar(x + i * width, rates, width, label=result['method'])

        ax.set_xlabel('Achievements')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Achievement Success Rates Comparison')
        ax.set_xticks(x + width * (len(agent_results) - 1) / 2)
        ax.set_xticklabels(all_achievements, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.results_dir / 'achievement_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_curves(self, log_dirs: List[str], method_names: List[str],
                           save_path: Optional[str] = None):
        """Plot training curves for multiple agents."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for log_dir, method_name in zip(log_dirs, method_names):
            run_data = self.load_agent_stats(log_dir, method_name)
            if run_data is None:
                continue

            steps = run_data['xs']
            rewards = run_data['reward']
            lengths = run_data['length']

            # Plot rewards
            ax1.plot(steps, np.cumsum(rewards), label=method_name, alpha=0.8)

            # Plot episode lengths (smoothed)
            smoothed_lengths = np.convolve(lengths, np.ones(50)/50, mode='valid')
            smooth_steps = steps[:len(smoothed_lengths)]
            ax2.plot(smooth_steps, smoothed_lengths, label=method_name, alpha=0.8)

        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('Learning Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Episode Length (smoothed)')
        ax2.set_title('Survival Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, agent_results: List[Dict], output_file: str = "evaluation_report.md"):
        """Generate a markdown report of evaluation results."""
        report_path = self.results_dir / output_file

        with open(report_path, 'w') as f:
            f.write("# Crafter Agent Evaluation Report\n\n")

            # Summary table
            f.write("## Summary\n\n")
            f.write("| Method | Crafter Score | Avg Reward | Avg Episode Length | Episodes |\n")
            f.write("|--------|---------------|------------|-------------------|----------|\n")

            for result in sorted(agent_results, key=lambda x: x['crafter_score'], reverse=True):
                f.write(f"| {result['method']} | {result['crafter_score']:.2f} | "
                       f"{result['avg_reward']:.2f} | {result['avg_episode_length']:.1f} | "
                       f"{result['total_episodes']} |\n")

            # Detailed achievement breakdown
            f.write("\n## Achievement Success Rates\n\n")
            for result in agent_results:
                f.write(f"### {result['method']}\n\n")
                for achievement, rate in sorted(result['achievement_rates'].items()):
                    f.write(f"- **{achievement}**: {rate:.1f}%\n")
                f.write("\n")

        print(f"Report saved to {report_path}")
        return report_path