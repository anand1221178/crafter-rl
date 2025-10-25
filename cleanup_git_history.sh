#!/bin/bash
# Remove large files from git history to make repo smaller

echo "This will remove all .pt checkpoints and .jsonl files from git history"
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Remove large files from all commits
git filter-repo --invert-paths --path-glob 'logs/**/*.pt' --force
git filter-repo --invert-paths --path-glob 'logdir/**/*.pt' --force
git filter-repo --invert-paths --path-glob '**/*.jsonl' --force

echo "✓ Done! Large files removed from git history"
echo ""
echo "Next steps:"
echo "1. Force push: git push origin master --force"
echo "2. Warn collaborators to re-clone the repo"
