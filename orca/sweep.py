"""Hyperparameter sweep adapter for Orca training using Optuna.

Wraps `OrcaTrainer` as an Optuna objective so you can run a sweep over
learning rate, batch size, MCTS sims, and train-steps in a few lines.

Usage:
    pip install 'hexbot[sweep]'
    python -m orca.sweep --trials 20 --iterations-per-trial 5
    python -m orca.sweep --trials 50 --storage sqlite:///sweep.db

Each trial runs a short training (default 5 iterations) and the
objective is the final `current_elo`. Other expensive features
(curriculum, auto-tuner) are disabled per trial so trials are cheap
and comparable.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna  # type: ignore


def objective(trial: "optuna.Trial", iterations_per_trial: int) -> float:
    """Optuna objective: train briefly with sampled hyperparams, return final ELO."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    mcts_sims = trial.suggest_int("mcts_sims", 50, 300, step=50)
    train_steps = trial.suggest_int("train_steps", 50, 300, step=50)

    from orca.train import OrcaTrainer
    trainer = OrcaTrainer(
        iterations=iterations_per_trial,
        lr=lr,
        batch_size=batch_size,
        mcts_sims=mcts_sims,
        train_steps=train_steps,
        # Keep trials fast and comparable
        use_curriculum=False,
        use_auto_tuner=False,
        elo_every=iterations_per_trial,  # one ELO eval at the end is enough
    )
    trainer.run()
    return float(trainer.metrics.get("current_elo", 0.0))


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Number of trials to run (default: 20)",
    )
    parser.add_argument(
        "--iterations-per-trial", type=int, default=5,
        help="Training iterations per trial (default: 5)",
    )
    parser.add_argument(
        "--study-name", default="hexbot-sweep",
        help="Optuna study name (default: hexbot-sweep)",
    )
    parser.add_argument(
        "--storage", default=None,
        help="Optuna storage URL (e.g. sqlite:///sweep.db) for resumable sweeps",
    )
    args = parser.parse_args(argv)

    try:
        import optuna
    except ImportError:
        print("optuna not installed. pip install 'hexbot[sweep]'",
              file=sys.stderr)
        return 1

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, args.iterations_per_trial),
        n_trials=args.trials,
    )

    print("\n" + "=" * 60)
    print(f"  Best ELO: {study.best_value:.1f}")
    print("  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
