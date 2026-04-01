# Dashboard Guide

The hexbot training dashboard provides live visualization of games, ELO tracking, loss curves, and more. It works with any bot -- the framework's built-in bots or your own.

See also: [Getting Started](getting-started.md) | [Bot Approaches](bot-approaches.md) | [API Reference](api-reference.md)

---

## Quick Start

```bash
python test_dashboard.py       # real bot games on port 5002
```

Open the URL printed in the terminal to see the dashboard.

---

## Python API

### Arena Mode (2 lines)

Pit any two bots against each other. ELO is computed automatically from results.

```python
from hexbot import Bot
from dashboard import Dashboard

dash = Dashboard(port=5001)
dash.start()

# One line - dashboard handles everything: games, ELO, charts
dash.run_arena(Bot.heuristic(), Bot.random(), games=100)
```

### Training Mode (auto-ELO via snapshots)

The dashboard stores snapshots of your bot over time and automatically plays the current version against past versions to compute ELO.

```python
from hexbot import Bot
from dashboard import Dashboard

dash = Dashboard(port=5001)
dash.start()

# Dashboard runs self-play, snapshots the bot, auto-computes ELO
dash.train(Bot.heuristic(), iterations=50, games_per_iter=20)
```

#### train() Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bot` | required | `function(game)->(q,r)` or object with `best_move(game)` |
| `iterations` | `100` | Number of training iterations |
| `games_per_iter` | `20` | Self-play games per iteration |
| `opponent` | `None` | Opponent for self-play (default: bot plays itself) |
| `eval_every` | `5` | Run ELO evaluation every N iterations |
| `eval_games` | `10` | Games per ELO evaluation |
| `snapshot_every` | `3` | Snapshot bot every N iterations for ELO |

### Using Your Own Bot

Any function that takes a game and returns a move works:

```python
from dashboard import Dashboard

dash = Dashboard(port=5001)
dash.start()

def my_bot(game):
    # your logic here
    return (0, 0)

dash.train(my_bot, iterations=50)  # auto-ELO, auto-charts, auto-everything
```

### Manual Control

For full control, push games and metrics yourself:

```python
dash.add_game(moves=[[0,0],[1,0]], result=1.0)
dash.add_metric(iteration=1, loss=0.5, elo=1050)
dash.update_progress(step=50, total=100)
```

---

## REST API

The dashboard is a standard HTTP server. Any language that can send JSON over HTTP has full access to every feature.

Start the dashboard with `python dashboard.py`, then send data from your bot process.

### Submit a Game

```
POST http://localhost:5001/api/game
Content-Type: application/json

{ "moves": [[0,0],[1,0],[0,1],[2,0]], "result": 1.0 }
```

### Submit Metrics

```
POST http://localhost:5001/api/metric
Content-Type: application/json

{ "iteration": 5, "loss": {"total": 0.82}, "elo": 1100, "wins": [8,2,0], "games": 10 }
```

### Read Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stats` | Current aggregate stats |
| `GET` | `/api/elo` | ELO history array |
| `GET` | `/api/losses` | Loss curve data |
| `GET` | `/api/games` | Recent 50 game move histories |
| `GET` | `/api/resources` | CPU/RAM history |
| `GET` | `/api/winrates` | Win rate history |
| `GET` | `/api/gamelength` | Game length history |
| `GET` | `/api/speed` | Training speed history |

### Write Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/game` | Submit a completed game |
| `POST` | `/api/metric` | Submit training metrics |

---

## WebSocket Events

For real-time streaming, connect via Socket.IO protocol.

### Client-to-Server

| Event | Data | Description |
|-------|------|-------------|
| `game_result` | `{ moves: [...], result: 1.0 }` | Submit a completed game |
| `metric` | `{ iteration: 5, loss: {...}, elo: 1100 }` | Submit training metrics |

### Server-to-Client

| Event | Data | Description |
|-------|------|-------------|
| `game_complete` | `{ game_idx, moves, result, ... }` | A game finished |
| `stats_update` | `{ iteration, total_games, elo, ... }` | Stats changed |
| `train_progress` | `{ step, total, pct, phase, ... }` | Training progress update |

The REST API is simplest to integrate. The WebSocket gives you real-time push updates (useful if your bot needs to react to dashboard state). Both give full access to all dashboard features.

---

## Game Viewer

The left panel shows an animated replay of training games with numbered moves. Black hexagons represent Player 0 and white hatched hexagons represent Player 1. Gray dots show empty hex positions around the stones.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause / resume auto-advance |
| `Right arrow` | Step forward one move (pauses auto-advance) |
| `Left arrow` | Step backward one move |
| `R` | Restart current game from the beginning |

When you use arrow keys to step through moves, auto-advance pauses so you can analyze the position. Press Space to resume.

The game history bar below the board shows recent games. Click any game to re-watch it.

---

## Settings

Click the gear icon in the header to adjust dashboard settings.

| Setting | Default | Description |
|---------|---------|-------------|
| Replay speed | 120ms | Speed of game replay animation |
| Dot size | 2 | Size of empty hex grid dots |
| Grid radius | 2 | How many empty hexes shown around stones |
| Move numbers | On | Show move order numbers on stones |
| Auto-refresh | On | Periodically refresh charts |

All settings are saved to your browser's localStorage and persist across sessions.

---

## Charts

The right panel has 6 collapsible chart panels. Click any header to collapse or expand.

| Chart | Description |
|-------|-------------|
| **ELO Progression** | Rating over training iterations |
| **Loss Curves** | Total, value, and policy loss |
| **Win Rates** | Player 0 vs Player 1 win percentages |
| **Game Length** | Average moves per game |
| **Training Speed** | Games per second |
| **Resources** | CPU and RAM usage |

---

## KaTrain-Style Analysis View (v4.1)

The dashboard now includes analysis overlays inspired by KaTrain, showing move
quality and threats directly on the board during game replay.

### Keyboard Toggles

| Key | Overlay | Description |
|-----|---------|-------------|
| `V` | Value / quality | Color each move by quality: green = good, red = blunder (value drop > 0.2) |
| `T` | Threats | Highlight cells with 4+ in-a-row threats |

### Move Quality Coloring

When the V overlay is active, each stone is tinted based on the value head's
assessment of the move:

- **Green** -- move maintained or improved the position
- **Red** -- blunder, value dropped by more than 0.2
- **Circle size** -- proportional to MCTS visit count (top moves overlay)

### Win Rate Chart

A line chart below the board tracks the value head output per move, showing
momentum swings throughout the game. Hover over the chart to jump to that move.

### Best ELO Badge

The dashboard footer now displays `Best: XXXX @ iter N`, tracking the highest
ELO achieved across all training iterations. Stored in the `DataStore`.

---

## Next Steps

- [Getting Started](getting-started.md) -- learn the HexGame API
- [Bot Approaches](bot-approaches.md) -- six strategies to try with the dashboard
- [API Reference](api-reference.md) -- full method signatures
