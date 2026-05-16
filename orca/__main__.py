"""Entry point for `python -m orca [subcommand]`.

Subcommands:
    train      Run the AlphaZero-style training loop (default).
    init NAME  Scaffold a new bot project from templates.

If no subcommand is given, falls through to `train` for backwards
compatibility with the original `python -m orca` invocation.
"""

import sys


def _main():
    argv = sys.argv[1:]
    if argv and argv[0] in {"init", "train"}:
        sub = argv.pop(0)
        sys.argv = [f"python -m orca {sub}"] + argv
        if sub == "init":
            from orca.init import main as init_main
            sys.exit(init_main())
        # else: train (fall through)
    from orca.train import main as train_main
    train_main()


if __name__ == "__main__":
    _main()
