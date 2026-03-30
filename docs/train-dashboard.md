# Training Dashboard

The training dashboard (`train_dashboard.py`) is a web-based UI that wraps the full Orca training pipeline with live game visualization, charts, and controls.

This is the visual counterpart to `python -m orca.train` (CLI). Same training logic, but with a browser UI.

See also: [Training Guide](training-guide.md) | [Configuration](configuration.md) | [Dashboard Guide](dashboard-guide.md) (API-only dashboard)

---

## Quick Start

```bash
python train_dashboard.py
```

Then open http://localhost:5001. Click the play button to start training.

---

## How It Differs From `orca.train`

| Feature | `python -m orca.train` | `python train_dashboard.py` |
|---------|----------------------|---------------------------|
| UI | Terminal output only | Web browser with charts |
| Live game replay | No | Yes (hex board animation) |
| Charts | No | ELO, loss, win rates, game length, speed, resources |
| Start/stop | Ctrl+C | Browser buttons |
| Progress bar | No | Yes (self-play + training) |
| ONNX export | For workers | For workers |
| Game streaming | No | Latest games replay in browser |
| Background scraper | No | Yes (optional, hexo.did.science) |
| Human game loading | Via CLI | Auto on first iteration |

Both use the same underlying training logic from `bot.py`: same network, same MCTS, same self-play, same checkpoints. The train dashboard adds visualization on top.

---

## UI Overview

### Header
- Play/stop buttons to start and stop training
- Status indicator (IDLE, TRAINING, iteration count)
- Connection dot (green = WebSocket connected)
- Settings gear icon

### Left Panel
- **Live stats**: iteration, workers, games, avg length, win%, loss
- **Progress bar**: self-play and training step progress
- **Game replay**: animated hex board showing training games
- **Game history**: clickable list of recent games to re-watch

### Right Panel (collapsible charts)
- **ELO Progression**: rating over iterations
- **Loss Curves**: total, value, policy loss
- **Win Rates**: P0 vs P1 percentages
- **Game Length**: average moves per game
- **Training Speed**: games per second
- **Resources**: CPU and RAM usage

### Footer
- Compact stats: iteration, games, ELO, win%, CPU, RAM

---

## Training Pipeline

The dashboard wraps the same components as `orca.train.OrcaTrainer`:

1. **Self-play**: parallel workers generate games via `self_play_game_v2()`
2. **Augmentation**: hex-valid symmetry transforms (3x data)
3. **Training**: gradient steps with `train_step()`
4. **ELO evaluation**: generational arena every 2 iterations
5. **Checkpoint**: saves every 5 iterations

### Curriculum

Adaptive sim/game scaling (same as orca.train):
- 0-30 min: 50 sims, 60 games/iter
- 30min-1.5h: 100 sims, 50 games
- 1.5h-3h: 150 sims, 40 games
- 3h+: 200 sims, 30 games

With plateau detection: if ELO stalls for 10+ iterations, sims boost by +50.

### Workers

Uses `ProcessPoolExecutor` with 2 games per future for streaming results. Default 5 workers (auto-scaled by CPU count).

---

## Configuration

The train dashboard reads constants from `bot.py` (which imports from `orca/config.py`). To change training parameters, edit `orca/config.py` before starting.

Key settings to tune:
- `BATCH_SIZE`: training batch size (default 1024)
- `NUM_SIMULATIONS`: MCTS sims per move (curriculum overrides)
- `LEARNING_RATE`: Adam optimizer LR (default 0.001)
- Games per iteration, train steps, ELO frequency

See [Configuration Reference](configuration.md) for the complete list.

---

## API Routes

The train dashboard exposes these HTTP endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard HTML page |
| `GET` | `/api/stats` | Current training stats |
| `GET` | `/api/elo` | ELO history (returns `{elo_history: [...]}`) |
| `GET` | `/api/losses` | Loss curve data |
| `GET` | `/api/resources` | CPU/RAM history |
| `GET` | `/api/winrates` | Win rate history |
| `GET` | `/api/speed` | Training speed history |
| `GET` | `/api/gamelength` | Game length history |
| `GET` | `/api/autotuner` | AutoTuner status and decisions |
| `POST` | `/api/train/start` | Start training (JSON body: `{iterations, games_per_iter, train_steps}`) |
| `POST` | `/api/train/stop` | Stop training |

### WebSocket Events (Socket.IO)

Server emits:
- `iteration_start` - `{iteration, total}`
- `game_complete` - `{game_idx, total_games, result, num_moves, moves}`
- `iteration_complete` - full iteration metrics
- `training_complete` - training finished
- `train_progress` - `{step, total, loss, pct}`

---

## Checkpoints

Saved every 5 iterations to `hex_checkpoint_N.pt`. Includes:
- `model_state_dict`: network weights
- `optimizer_state_dict`: Adam optimizer state
- `scheduler_state_dict`: LR scheduler state
- `metrics`: iteration count, ELO history, total games
- `auto_tuner`: hyperparameter tuning state

Replay buffer saved separately as `replay_buffer.pkl`.

On restart, the dashboard auto-resumes from the latest checkpoint.

---

## Differences From API Dashboard (`dashboard.py`)

The API dashboard (`dashboard.py`) is a generic, clean dashboard for any bot. The train dashboard (`train_dashboard.py`) is specifically for the Orca training pipeline.

| Feature | `dashboard.py` | `train_dashboard.py` |
|---------|---------------|---------------------|
| Purpose | Generic bot visualization | Orca training management |
| Training | None (API only) | Full pipeline built-in |
| Dependencies | Flask, psutil | Flask, psutil, PyTorch, bot.py |
| Bot imports | None | Full bot.py stack |
| Size | ~1000 lines | ~2400 lines |
| Start training | External (REST/WebSocket) | Built-in (play button) |

---

## Next Steps

- [Training Guide](training-guide.md) - detailed training pipeline docs
- [Configuration](configuration.md) - all tunable parameters
- [Orca Bot](orca.md) - network architecture and loading
