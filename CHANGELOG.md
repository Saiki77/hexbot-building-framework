# Changelog

## v4.1.0 - Training Quality + CUDA Performance + Analysis View

### CUDA Performance (2x speedup on NVIDIA GPUs)
- **Mixed precision training** - FP16 via GradScaler, was implemented but never activated. Now wired up and working. Includes gradient clipping and pin_memory for faster CPU-to-GPU transfers.
- **torch.compile** - automatically applied on CUDA after checkpoint restore for reduced kernel overhead.
- **GPU inference server** (`orca/gpu_server.py`) - centralized batched inference for self-play workers. Workers play on CPU (C engine), send positions to GPU for NN eval. Eliminates per-worker network loading.

### Training Data Quality
- **Progressive short-game penalties** - replaces flat 0.5x penalty:
  - Games < 10 moves: discarded entirely (junk)
  - 10-19 moves: 0.2x priority
  - 20-29 moves: 0.4x priority
  - 30-39 moves: 0.7x priority
  - 40+ moves: 1.0x (no penalty)
- **Long game bonus** - 45+ moves: 1.3x, 60+ moves: 1.8x priority
- **Blocking reward** - moves that successfully block opponent threats get 3.0x priority boost. Surviving a threat gets 2.0x. Previously only offensive moves (forks) got priority boosts.

### Checkpoint Gating
- Only saves `hex_best.pt` when ELO actually improves
- Safety checkpoints every 5 iterations unchanged
- Tracks best_elo and best_iteration in metrics
- Logs clearly whether each iteration improved: "NEW BEST" or "(best: X @ iter N)"

### ELO Evaluation
- **Baseline matches** - now plays against random bot (anchored ~500 ELO) and heuristic bot (anchored ~1000 ELO) alongside generational arena
- **Blended ELO** - 60% generational + 20% random-anchored + 20% heuristic-anchored for more stable, meaningful ratings
- Configurable: `ELO_BASELINE_GAMES` in config.py (default 4, set 0 to disable)

### Full Hex Rotation Augmentation (4x -> 8x data)
- **4 new axial rotations** (60, 120, 240, 300 degrees) via coordinate re-encoding
- Total: 7 transforms (3 grid-safe + 4 axial) = up to 8x training data per game
- Axial rotations get 0.7x priority (grid-safe get 0.8x)
- Rotations where all mass falls off-grid are automatically filtered

### New Network Architectures
- **hex-native** (3.1M params) - true 7-weight hex convolution kernel, no masking. Only learns from actual hex neighbors.
- **hex-native-circular** - same with toroidal padding for cleaner boundary handling
- Total: 8 architectures now available via `create_network()`

### KaTrain-style Analysis View (Dashboard)
- **Move quality coloring** - green (good), red (blunder) based on value drop > 0.2
- **Top moves overlay** - sized circles showing MCTS visit distribution
- **Win rate chart** - line chart below board tracking value head output per move
- **Threat overlay** - highlights cells with 4+ in a row threats
- **Keyboard toggles** - V (value/quality), T (threats)
- **Per-move analysis data** stored during self-play: value_estimate, top_moves, threat_count

### Configurable AB Hybrid
- `USE_AB_HYBRID` / `--no-ab-hybrid` - disable the depth-4 alpha-beta pre-check in MCTS
- `AB_HYBRID_DEPTH` - configurable search depth (default 4)
- Disabling lets MCTS see full game trees including blocking positions, producing richer training data

### Dashboard Polish
- Best ELO badge in footer: "Best: XXXX @ iter N"
- best_elo tracking in DataStore

### Code Maintainability
- New focused modules: `orca/samples.py`, `orca/replay.py`, `orca/augment.py`, `orca/gpu_server.py`
- All backwards compatible (orca/data.py still works as before)
- 81 unit tests + 23 pip install tests, all passing

### Configuration Additions
```python
# Defensive training
BLOCKING_PRIORITY_BOOST = 3.0
SURVIVAL_PRIORITY_BOOST = 2.0
USE_AB_HYBRID = True
AB_HYBRID_DEPTH = 4

# ELO baselines
ELO_BASELINE_GAMES = 4

# Mixed precision
USE_MIXED_PRECISION = True
GRAD_CLIP_NORM = 1.0
```

### Backwards Compatible
All v4.0.0 API unchanged. All existing imports work.
