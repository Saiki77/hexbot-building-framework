"""Quick training demo using the Orca pipeline.

Runs 5 iterations of self-play + training to demonstrate the pipeline.
For real training, use more iterations:

    python -m orca.train --iterations 100

Usage:
    python examples/train_orca.py
    python examples/train_orca.py --iterations 10
    python examples/train_orca.py --fresh   # start from scratch
"""
import sys; sys.path.insert(0, '..')
import argparse

parser = argparse.ArgumentParser(description='Train Orca bot (quick demo)')
parser.add_argument('--iterations', type=int, default=5, help='Training iterations')
parser.add_argument('--games', type=int, default=10, help='Games per iteration')
parser.add_argument('--fresh', action='store_true', help='Start from scratch (no resume)')
args = parser.parse_args()

from orca.train import OrcaTrainer

trainer = OrcaTrainer(
    iterations=args.iterations,
    games_per_iter=args.games,
    train_steps=50,  # fewer steps for demo
    resume=not args.fresh,
    config='standard',
)

print(f'Training Orca: {args.iterations} iterations, {args.games} games/iter')
print('Press Ctrl+C to stop early\n')
trainer.run()
print('\nDone! Resume training with: python -m orca.train')
