# hexbot

A Python framework for building AI bots for [Hexagonal Tic-Tac-Toe](https://hexo.did.science) (Connect-6 on an infinite hex grid). Fast C game engine, multiple neural network architectures, AlphaZero training pipeline, and a live training dashboard.

Home of the **Orca** bot.

[![PyPI](https://img.shields.io/pypi/v/hexbot)](https://pypi.org/project/hexbot/)
[![Wiki](https://img.shields.io/badge/docs-wiki-blue?logo=github)](https://github.com/Saiki77/hexbot-building-framework/wiki)

> Full guides, learning path, and feature reference live on the **[Wiki](https://github.com/Saiki77/hexbot-building-framework/wiki)**. Start with [Quickstart](https://github.com/Saiki77/hexbot-building-framework/wiki/Quickstart) or [Concepts](https://github.com/Saiki77/hexbot-building-framework/wiki/Concepts). Try it now in **[Colab](https://colab.research.google.com/github/Saiki77/hexbot-building-framework/blob/main/notebooks/colab_quickstart.ipynb)** (free T4 GPU, ~5 minutes).

> **Built a great bot with this?** Open an issue or get in touch, and I'll feature it here.

## Model Zoo

Share bots with the community. Download by name, upload your trained checkpoints. Built-in metadata (architecture, ELO, author, description) keeps everything interoperable.

```python
from orca.zoo import Zoo

# See what others have shared
Zoo.list()

# Try someone else's bot
bot = Zoo.download('orca-v3')

# Share yours
Zoo.package('hex_checkpoint_500.pt',
            name='my-bot-v1', author='You', elo=1450,
            description='Trained 500 iterations with hex-masked')
Zoo.upload('my-bot-v1.pt', name='my-bot-v1')
```

Trained something interesting? Publish it and the next person can pull it with one line. See the [Evaluation and Sharing](https://github.com/Saiki77/hexbot-building-framework/wiki/Evaluation-and-Sharing) wiki page for the full lifecycle (Arena, Leaderboard, Zoo).

## Featured Community Bots

<!-- BOTS:START -->
_No community bots rated yet. Train one, then `Leaderboard().rate(bot, name='my-bot-v1')` to land here. See [Evaluation and Sharing](https://github.com/Saiki77/hexbot-building-framework/wiki/Evaluation-and-Sharing)._
<!-- BOTS:END -->

This table is regenerated from `leaderboard.json` by `scripts/update_featured_bots.py` (runs in CI on every push to main).

## Installation

```bash
pip install hexbot
```

Or from source (for development):

```bash
git clone https://github.com/Saiki77/hexbot-building-framework.git
cd hexbot-building-framework
pip install -e .               # core (game engine + torch)
pip install -e '.[dashboard]'  # + dashboard UI
pip install -e '.[all]'        # everything
```

The C engine auto-compiles on first import (`cc`/`gcc`/`clang` required).

## Quick Start

```python
from hexbot import HexGame, Bot, Arena

game = HexGame()
game.place(0, 0)         # P0: 1 stone first turn
game.place(1, 0)         # P1: first of 2
game.place(1, -1)        # P1: second stone

# Any function is a bot
def my_bot(game):
    return game.legal_moves()[0]

result = Arena(my_bot, Bot.heuristic(), num_games=20).play()
```

## Orca Bot

Pre-trained AlphaZero-style bot. The included checkpoint is early-stage (65 iterations) - gets significantly stronger with more training.

```python
from hexbot import Bot
orca = Bot.orca()
move = orca.best_move(game)
```

### Train

```bash
# From game collections (fast bootstrap)
python -m orca.scrape --games 5000 --output games.jsonl
python -m orca.sft --data games.jsonl --then-selfplay 50

# Pure self-play
python -m orca.train --iterations 100

# With web dashboard
python train_dashboard.py
```

### Network Architectures

| Config | Params | Description |
|--------|--------|-------------|
| `fast` | 656K | Quick experiments |
| `standard` | 3.9M | Default Orca (128 filters, 12 ResBlocks) |
| `hex-masked` | 3.9M | Hex-neighbor masked CNN (recommended for hex) |
| `large` | 14.5M | Maximum strength (256 filters) |
| `orca-transformer` | 4.4M | CNN + transformer attention |
| `hex-gnn` | 432K | Graph neural network on hex topology |
| `multiscale` | 1.1M | Local CNN + global attention two-tower |

```bash
python -m orca.train --config hex-gnn --iterations 50
```

## Features

| Feature | Description |
|---------|-------------|
| **SFT Pipeline** | Train from game collections (JSONL/CSV/text), chain with self-play |
| **Endgame Solver** | Deep alpha-beta with transposition cache, integrates with MCTS |
| **Opening Book** | Trie-based lookup from winning games, blends with MCTS policy |
| **6 Architectures** | CNN, transformer, GNN, multiscale, hybrid, fast |
| **Mixed Precision** | FP16 on CUDA for 2x training speed |
| **Distributed** | Multi-GPU (DDP) and multi-machine (Ray) |
| **Skill Curriculum** | 6-level auto-progression: basics through endgame |
| **Ensemble** | Average N checkpoints with uncertainty estimation |
| **Model Zoo** | Share and download community models |
| **Leaderboard** | Rate bots against references, compare head-to-head |
| **Plugin System** | Register custom bots and network architectures |
| **Dashboard** | REST + WebSocket API, live game replay, charts |
| **30+ API Functions** | NN eval, MCTS search, threats, solver, augmentation |

## Bot Approaches

| # | Approach | Complexity |
|---|----------|-----------|
| 1 | [Hand-tuned evaluation](docs/bot-approaches.md#approach-1-hand-tuned-evaluation) | Low |
| 2 | [Evolutionary weights](docs/bot-approaches.md#approach-2-evolutionary-weights) | Low |
| 3 | [Minimax with alpha-beta](docs/bot-approaches.md#approach-3-minimax-with-alpha-beta) | Medium |
| 4 | [Monte Carlo playouts](docs/bot-approaches.md#approach-4-monte-carlo-random-playouts) | Medium |
| 5 | [Neural network (AlphaZero)](docs/bot-approaches.md#approach-5-neural-network-alphazero-style) | High |
| 6 | [Hybrid strategies](docs/bot-approaches.md#approach-6-combine-multiple-strategies) | High |

## Architecture

```
hexbot.py              Framework API (Bot, Arena, 30+ analysis functions)
hexgame.py             Game engine (wraps C engine via ctypes)
engine.c               2,300 lines optimized C (bitboard win detection)
bot.py                 Neural networks, MCTS, training pipeline
dashboard.py           Dashboard API (REST + WebSocket)
train_dashboard.py     Training dashboard with web UI
pyproject.toml         pip install config
orca/
  __init__.py          Orca bot loader
  config.py            All tunable parameters
  train.py             Training pipeline (CLI + library)
  sft.py               Supervised fine-tuning from games
  scrape.py            Game downloader
  solver.py            Endgame solver
  openings.py          Opening book
  curriculum.py        Skill-based curriculum
  ensemble.py          Multi-checkpoint ensemble
  distributed.py       Multi-GPU / Ray training
  zoo.py               Model zoo
  leaderboard.py       Bot rating system
  transformer_net.py   Transformer architecture
  hex_gnn.py           Graph neural network
  multiscale_net.py    Multi-scale architecture
  benchmark.py         Platform benchmark (compare Mac/CUDA/CPU)
  checkpoint.pt        Pre-trained weights
docs/                  Detailed documentation
examples/              Example bot scripts
```

## Benchmark

Compare performance across platforms:

```bash
python -m orca.benchmark --quick    # ~30 seconds
```

Measures: C engine speed, NN inference, MCTS search, self-play, training steps, augmentation.

## Device Support

| Priority | Device | Platform |
|----------|--------|----------|
| 1 | CUDA | NVIDIA GPU (Linux/Windows) |
| 2 | MPS | Apple Silicon (macOS) |
| 3 | CPU | Any platform |

## Documentation

Full docs live on the **[Wiki](https://github.com/Saiki77/hexbot-building-framework/wiki)** (16 pages organized by Learn / Build / Train / Evaluate / Reference).

Good starting points:

| Page | What it covers |
|------|----------------|
| [Quickstart](https://github.com/Saiki77/hexbot-building-framework/wiki/Quickstart) | 5 minutes from `pip install` to a bot playing |
| [Game Rules](https://github.com/Saiki77/hexbot-building-framework/wiki/Game-Rules) | Hex Connect-6 rules in 60 seconds |
| [Concepts](https://github.com/Saiki77/hexbot-building-framework/wiki/Concepts) | MCTS, AlphaZero, policy/value, ELO explained |
| [Bot Approaches](https://github.com/Saiki77/hexbot-building-framework/wiki/Bot-Approaches) | Six complete bot designs with full code |
| [Training Guide](https://github.com/Saiki77/hexbot-building-framework/wiki/Training-Guide) | Self-play, SFT, curriculum, distributed |
| [Playing Online](https://github.com/Saiki77/hexbot-building-framework/wiki/Playing-Online) | Run your bot on hexo.did.science |
| [API Reference](https://github.com/Saiki77/hexbot-building-framework/wiki/API-Reference) | All public functions and classes |
| [FAQ](https://github.com/Saiki77/hexbot-building-framework/wiki/FAQ) | Common pitfalls and questions |

## Contributing

Built for the Hexagonal Tic-Tac-Toe community. Contributions welcome.

## License

MIT
