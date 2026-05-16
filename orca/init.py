"""Project scaffolder: `python -m orca init <name>`.

Creates a templated bot project directory with a config, train/play
shell scripts, a plugins.py stub, and a starter README. Lowers the
barrier from "read the wiki" to "run a command."

Usage:
    python -m orca init my-bot                 # create ./my-bot/
    python -m orca init my-bot --profile=cuda-single
    python -m orca init my-bot --force         # overwrite existing dir
"""

from __future__ import annotations

import argparse
import os
import sys


_README_TEMPLATE = """# {name}

A Hex Connect-6 bot scaffolded with `python -m orca init`.

## Train

```bash
bash train.sh
```

This runs `python -m orca.train --profile={profile} --iterations 20`.
Edit `train.sh` to change iterations, sims, network architecture, or
to add the `--tensorboard` flag for run-to-run metric comparison.

## Play

```bash
bash play.sh
```

Opens the live dashboard at http://localhost:5000. Click the **Play**
tab to play against the latest checkpoint.

## Custom bots

Define a custom bot in `plugins.py`, then refer to it by name in
training or play scripts. See the [Plugin System wiki page](
https://github.com/Saiki77/hexbot-building-framework/wiki/Plugin-System)
for full details.

## Documentation

- [Quickstart](https://github.com/Saiki77/hexbot-building-framework/wiki/Quickstart)
- [Bot Approaches](https://github.com/Saiki77/hexbot-building-framework/wiki/Bot-Approaches)
- [Training Guide](https://github.com/Saiki77/hexbot-building-framework/wiki/Training-Guide)
"""


_TRAIN_SH = """#!/usr/bin/env bash
# Train this bot. Edit the flags below to taste.
set -euo pipefail

python -m orca.train \\
    --profile={profile} \\
    --iterations 20 \\
    --tensorboard
"""


_PLAY_SH = """#!/usr/bin/env bash
# Open the live training/play dashboard at http://localhost:5000
set -euo pipefail

python -m orca.train --profile={profile} --iterations 0 &  # warm up checkpoints
python ../train_dashboard.py
"""


_CONFIG_PY = '''"""Per-project config overrides.

Anything imported from `orca.config` can be overridden here by simply
reassigning the value. Imported once at training start, so changes are
picked up on the next `bash train.sh`.
"""

# Example overrides (uncomment to use):
#
# import orca.config
# orca.config.NUM_SIMULATIONS = 150     # MCTS sims per move
# orca.config.BATCH_SIZE = 512          # training batch size
# orca.config.NUM_FILTERS = 256         # bigger network
'''


_PLUGINS_PY = '''"""Custom bots and networks for this project.

Import this module before using `Bot.from_name(...)` to make registered
names visible to the framework, dashboards, and CLI flags.

See: https://github.com/Saiki77/hexbot-building-framework/wiki/Plugin-System
"""

from hexbot import BotProtocol, register_bot, HexGame


class MyBot(BotProtocol):
    """Template: replace this with your bot logic."""

    def __init__(self, depth: int = 4):
        self.depth = depth

    def best_move(self, game: HexGame):
        # Default behaviour: short alpha-beta search via the C engine.
        # Replace this with your own move selection.
        return game.search(depth=self.depth)["best_move"]


register_bot("my-bot", MyBot)
'''


_GITIGNORE = """# Training artifacts
runs/
hex_checkpoint_*.pt
hex_best.pt
replay_buffer.pkl*
*.onnx

# Python
__pycache__/
*.py[cod]
.venv/
"""


_TEMPLATES = {
    "README.md": _README_TEMPLATE,
    "train.sh": _TRAIN_SH,
    "play.sh": _PLAY_SH,
    "config.py": _CONFIG_PY,
    "plugins.py": _PLUGINS_PY,
    ".gitignore": _GITIGNORE,
}

_EXECUTABLE = {"train.sh", "play.sh"}


def scaffold(name: str, profile: str = "cpu-only", force: bool = False) -> int:
    """Create a templated project at ./<name>/.

    Returns 0 on success, non-zero on error.
    """
    if not name or "/" in name or name.startswith("."):
        print(f"error: invalid project name {name!r}")
        return 2

    target = os.path.abspath(name)
    if os.path.exists(target) and not force:
        print(f"error: {target} already exists (use --force to overwrite)")
        return 1

    os.makedirs(target, exist_ok=True)
    for filename, template in _TEMPLATES.items():
        path = os.path.join(target, filename)
        content = template.format(name=name, profile=profile)
        with open(path, "w") as f:
            f.write(content)
        if filename in _EXECUTABLE:
            os.chmod(path, 0o755)
        print(f"  created {os.path.relpath(path)}")

    print()
    print(f"Project scaffolded at {target}")
    print(f"  cd {name}")
    print(f"  bash train.sh")
    return 0


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m orca init",
        description="Scaffold a new Hex bot project from templates.",
    )
    parser.add_argument("name", help="Project directory name (e.g., 'my-bot')")
    parser.add_argument(
        "--profile", default="cpu-only",
        help="Hardware profile baked into train.sh/play.sh (default: cpu-only)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite files if the project directory already exists",
    )
    args = parser.parse_args(argv)
    return scaffold(args.name, profile=args.profile, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
