"""
Analyze hyperparameter sweep results and find the best configuration.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_sweep_results(sweep_dir):
    """Load all evaluation results from sweep directory."""
    sweep_dir = Path(sweep_dir)
    results = []

    # Load configuration file
    config_file = sweep_dir / 'sweep_configs.json'
    with open(config_file, 'r') as f:
        configs = json.load(f)

    # Load evaluation results
    for i, config in enumerate(configs):
        # Find evaluation directory
        eval_dirs = list(sweep_dir.glob(f'eval_{i}_*'))
        if not eval_dirs:
            print(f"Warning: No evaluation results for config {i}")
            continue

        eval_dir = eval_dirs[0]

        # Find evaluation report
        reports = list(eval_dir.glob('evaluation_report_*.json'))
        if not reports:
            print(f"Warning: No evaluation report in {eval_dir}")
            continue

        report_file = reports[0]

        # Load report
        with open(report_file, 'r') as f:
            report = json.load(f)

        # Combine config and results
        result = {
            'config_id': i,
            'lr': config['lr'],
            'entropy_coef': config['entropy_coef'],
            'icm_beta': config['icm_beta'],
            'hidden_dim': config['hidden_dim'],
            'clip_epsilon': config['clip_epsilon'],
            'score': report['results']['score'],
            'reward': report['results']['reward'],
            'length': report['results']['length'],
        }

        # Add key achievements
        achievements = report['results']['achievements']
        result['wood'] = achievements.get('achievement_collect_wood', 0)
        result['pickaxe'] = achievements.get('achievement_make_wood_pickaxe', 0)
        result['sword'] = achievements.get('achievement_make_wood_sword', 0)
        result['coal'] = achievements.get('achievement_collect_coal', 0)
        result['zombie'] = achievements.get('achievement_defeat_zombie', 0)

        results.append(result)

    return pd.DataFrame(results)


def print_top_results(df, n=10):
    """Print top N configurations by score."""
    print(f"\n{'='*80}")
    print(f"TOP {n} CONFIGURATIONS (by Crafter Score)")
    print(f"{'='*80}")

    df_sorted = df.sort_values('score', ascending=False)

    for i, row in df_sorted.head(n).iterrows():
        print(f"\nRank {df_sorted.index.get_loc(i) + 1}: Config {row['config_id']}")
        print(f"  Score: {row['score']:.2f}%")
        print(f"  Hyperparameters:")
        print(f"    - lr:           {row['lr']:.0e}")
        print(f"    - entropy_coef: {row['entropy_coef']:.0e}")
        print(f"    - icm_beta:     {row['icm_beta']:.2f}")
        print(f"    - hidden_dim:   {row['hidden_dim']}")
        print(f"    - clip_epsilon: {row['clip_epsilon']:.2f}")
        print(f"  Metrics:")
        print(f"    - Reward: {row['reward']:.2f}")
        print(f"    - Length: {row['length']:.1f}")
        print(f"  Key Achievements:")
        print(f"    - Wood: {row['wood']:.0f}%, Pickaxe: {row['pickaxe']:.0f}%, Sword: {row['sword']:.0f}%")
        print(f"    - Coal: {row['coal']:.0f}%, Zombie: {row['zombie']:.0f}%")


def plot_parameter_importance(df, output_file=None):
    """Plot how each parameter affects score."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Hyperparameter Impact on Crafter Score', fontsize=16, fontweight='bold')

    params = ['lr', 'entropy_coef', 'icm_beta', 'hidden_dim', 'clip_epsilon']

    for idx, param in enumerate(params):
        ax = axes[idx // 3, idx % 3]

        # Group by parameter and compute mean/std
        grouped = df.groupby(param)['score'].agg(['mean', 'std', 'count'])

        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   fmt='o-', linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel('Crafter Score (%)', fontsize=12)
        ax.set_title(f'{param} vs Score', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add count labels
        for x, (mean, count) in enumerate(zip(grouped['mean'], grouped['count'])):
            ax.text(grouped.index[x], mean + 0.3, f'n={count}',
                   ha='center', fontsize=9)

    # Hide last subplot if odd number of params
    if len(params) % 3 != 0:
        axes[-1, -1].axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_file}")

    plt.show()


def plot_correlation_matrix(df, output_file=None):
    """Plot correlation between hyperparameters and score."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Select numeric columns
    cols = ['lr', 'entropy_coef', 'icm_beta', 'hidden_dim', 'clip_epsilon', 'score', 'reward', 'length']
    corr = df[cols].corr()

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Hyperparameter Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_file}")

    plt.show()


def plot_score_distribution(df, output_file=None):
    """Plot distribution of scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(df['score'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(df['score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["score"].mean():.2f}%')
    ax.axvline(df['score'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["score"].median():.2f}%')
    ax.set_xlabel('Crafter Score (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Score Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot by hidden_dim
    ax = axes[1]
    df.boxplot(column='score', by='hidden_dim', ax=ax)
    ax.set_xlabel('Hidden Dimension', fontsize=12)
    ax.set_ylabel('Crafter Score (%)', fontsize=12)
    ax.set_title('Score by Network Size', fontsize=12, fontweight='bold')
    plt.suptitle('')  # Remove default title

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_file}")

    plt.show()


def generate_summary_report(df, output_file):
    """Generate a text summary report."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER SWEEP SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total configurations tested: {len(df)}\n\n")

        f.write("Score Statistics:\n")
        f.write(f"  - Best:   {df['score'].max():.2f}%\n")
        f.write(f"  - Worst:  {df['score'].min():.2f}%\n")
        f.write(f"  - Mean:   {df['score'].mean():.2f}%\n")
        f.write(f"  - Median: {df['score'].median():.2f}%\n")
        f.write(f"  - Std:    {df['score'].std():.2f}%\n\n")

        # Best configuration
        best_idx = df['score'].idxmax()
        best = df.loc[best_idx]

        f.write("BEST CONFIGURATION:\n")
        f.write(f"  Config ID: {best['config_id']}\n")
        f.write(f"  Score:     {best['score']:.2f}%\n\n")
        f.write("  Hyperparameters:\n")
        f.write(f"    - lr:           {best['lr']:.0e}\n")
        f.write(f"    - entropy_coef: {best['entropy_coef']:.0e}\n")
        f.write(f"    - icm_beta:     {best['icm_beta']:.2f}\n")
        f.write(f"    - hidden_dim:   {best['hidden_dim']}\n")
        f.write(f"    - clip_epsilon: {best['clip_epsilon']:.2f}\n\n")

        # Parameter importance (variance in mean scores)
        f.write("PARAMETER IMPORTANCE (variance of mean scores per value):\n")
        params = ['lr', 'entropy_coef', 'icm_beta', 'hidden_dim', 'clip_epsilon']
        importances = []
        for param in params:
            means = df.groupby(param)['score'].mean()
            variance = means.var()
            importances.append((param, variance))

        importances.sort(key=lambda x: x[1], reverse=True)
        for param, var in importances:
            f.write(f"  - {param:15s}: {var:.3f}\n")

    print(f"✓ Saved summary: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter sweep results')
    parser.add_argument('sweep_dir', type=str, help='Directory containing sweep results')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top configs to show')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for plots')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(args.sweep_dir) / 'analysis'

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading sweep results...")
    df = load_sweep_results(args.sweep_dir)
    print(f"✓ Loaded {len(df)} configurations")

    if len(df) == 0:
        print("No results found!")
        return

    # Print top results
    print_top_results(df, n=args.top_n)

    # Generate plots
    print("\nGenerating analysis plots...")
    plot_parameter_importance(df, Path(args.output_dir) / 'parameter_importance.png')
    plot_correlation_matrix(df, Path(args.output_dir) / 'correlation_matrix.png')
    plot_score_distribution(df, Path(args.output_dir) / 'score_distribution.png')

    # Generate summary report
    generate_summary_report(df, Path(args.output_dir) / 'summary_report.txt')

    # Save results to CSV
    csv_file = Path(args.output_dir) / 'sweep_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved results CSV: {csv_file}")

    print(f"\n✓ Analysis complete! Results in {args.output_dir}")


if __name__ == '__main__':
    main()
