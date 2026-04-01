"""
HEX BOT - Training Dashboard

Black-and-white minimalist dashboard with live game visualization,
ELO progression, loss curves, and auto-scaling parallel training.

Usage: python dashboard.py
Then open http://localhost:5001
"""

from __future__ import annotations
import sys
import multiprocessing
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from flask import Flask, jsonify, request, Response
from flask_socketio import SocketIO

from main import HexGame
from bot import (
    HexNet, MCTS, self_play_game,
    get_device, NUM_SIMULATIONS, BATCH_SIZE, LEARNING_RATE, L2_REG,
    OnnxPredictor, encode_state,
)

# ---------------------------------------------------------------------------
# Timestamped print helper
# ---------------------------------------------------------------------------
from datetime import datetime as _dt
_print = print
def print(*args, **kwargs):
    """Override print to prepend timestamp."""
    ts = _dt.now().strftime('%H:%M:%S')
    _print(f'[{ts}]', *args, **kwargs)


# Resource monitor
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Monitors CPU, GPU (MPS), and RAM usage."""

    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        # Scale workers: C engine releases GIL, so more workers = more throughput
        # Use most cores, leaving 2 for training + system
        self.num_threads = min(12, max(2, self.cpu_count - 2))
        self._lock = threading.Lock()
        self._history: List[dict] = []

    def snapshot(self) -> dict:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        snap = {
            'cpu_pct': round(cpu_pct, 1),
            'ram_pct': round(mem.percent, 1),
            'ram_used_gb': round(mem.used / (1024 ** 3), 1),
            'ram_total_gb': round(mem.total / (1024 ** 3), 1),
            'cpu_count': self.cpu_count,
            'workers': self.num_threads,
            'gpu_available': torch.backends.mps.is_available(),
        }
        with self._lock:
            self._history.append(snap)
            if len(self._history) > 300:
                self._history = self._history[-300:]
        return snap

    def get_history(self) -> List[dict]:
        with self._lock:
            return list(self._history[-60:])

    @property
    def current_workers(self) -> int:
        return self.num_threads


# ---------------------------------------------------------------------------
# Curriculum: MCTS sim count by iteration
# ---------------------------------------------------------------------------

_curriculum_start_time = None
_curriculum_last_elo = None
_curriculum_stall_iters = 0

def get_curriculum_sims(iteration: int) -> int:
    """Adaptive sim curriculum: start low, scale up over hours.

    - First 10 iters: 50 sims (fast exploration, many games)
    - Iter 10-30: 100 sims (better quality moves)
    - Iter 30-60: 150 sims (deeper search)
    - Iter 60+: 200 sims (full depth)

    Also scales up faster if ELO stalls (plateau detection).
    """
    global _curriculum_start_time
    if _curriculum_start_time is None:
        _curriculum_start_time = time.perf_counter()

    hours_elapsed = (time.perf_counter() - _curriculum_start_time) / 3600

    # Time-based scaling: ramp up over 4 hours
    if hours_elapsed < 0.5:
        base_sims = 50
    elif hours_elapsed < 1.5:
        base_sims = 100
    elif hours_elapsed < 3.0:
        base_sims = 150
    else:
        base_sims = 200

    # Also consider iteration count (whichever gives more sims)
    if iteration < 10:
        iter_sims = 50
    elif iteration < 30:
        iter_sims = 100
    elif iteration < 60:
        iter_sims = 150
    else:
        iter_sims = 200

    # Plateau boost: if stalled for 10+ iters, bump sims
    sims = max(base_sims, iter_sims)
    if _curriculum_stall_iters >= 10:
        sims = min(400, sims + 50)

    return sims


def update_curriculum_plateau(current_elo: float) -> None:
    """Call after each ELO evaluation to track plateaus."""
    global _curriculum_last_elo, _curriculum_stall_iters
    if _curriculum_last_elo is not None:
        if abs(current_elo - _curriculum_last_elo) < 15:
            _curriculum_stall_iters += 1
        else:
            _curriculum_stall_iters = 0
    _curriculum_last_elo = current_elo


def get_curriculum_games(iteration: int, base: int) -> int:
    """More games when sims are low, fewer when sims are high."""
    sims = get_curriculum_sims(iteration)
    if sims <= 50:
        return 60   # fast games, do more of them
    if sims <= 100:
        return 50
    if sims <= 150:
        return 40
    return 30       # deep search, fewer games needed


# ---------------------------------------------------------------------------
# Parallel self-play worker (subprocess, CPU inference)
# ---------------------------------------------------------------------------

def _self_play_worker(onnx_path: str, num_sims: int,
                      games: int, positions: Optional[list] = None) -> list:
    """Run self-play games in a subprocess using ONNX Runtime.
    positions: list of (position_dict, hint_moves) tuples or plain dicts."""
    predictor = OnnxPredictor(onnx_path)
    mcts = MCTS(predictor, num_simulations=num_sims)

    results = []
    for i in range(games):
        pos = None
        hints = None
        if positions and i < len(positions):
            entry = positions[i]
            if isinstance(entry, tuple) and len(entry) == 2:
                pos, hints = entry
            elif isinstance(entry, dict):
                pos = entry
        samples, move_history = self_play_game(predictor, mcts, start_position=pos,
                                                hint_moves=hints)
        serialized = []
        for s in samples:
            serialized.append({
                'state': s.encoded_state.numpy(),
                'policy': s.policy_target,
                'player': s.player,
                'result': s.result,
                'threat': s.threat_label,
                'priority': s.priority,
            })
        result_val = samples[0].result if samples else 0.0
        results.append((serialized, [list(m) for m in move_history],
                        result_val, len(samples)))
    return results


def _self_play_worker_v2(net_state_dict: dict, net_config: str, num_sims: int,
                          games: int, positions: Optional[list] = None,
                          use_alphabeta: bool = True, ab_depth: int = 8) -> list:
    """V2 worker: CGameState + NNAlphaBeta or BatchedMCTS."""
    try:
        from bot import (CGameState, BatchedMCTS, BatchedNNAlphaBeta,
                         self_play_game_v2, create_network,
                         migrate_checkpoint_5to7, migrate_checkpoint_filters)
    except ImportError:
        return []

    import time as _time, os as _os

    t_load = _time.perf_counter()
    net = create_network(net_config)
    migrated = migrate_checkpoint_5to7(dict(net_state_dict))
    migrated = migrate_checkpoint_filters(migrated)
    net.load_state_dict(migrated, strict=False)
    net.eval()
    t_load = _time.perf_counter() - t_load

    if use_alphabeta:
        searcher = BatchedNNAlphaBeta(net, depth=ab_depth, nn_depth=5)
    else:
        searcher = BatchedMCTS(net, num_simulations=num_sims, batch_size=64)

    pid = _os.getpid()
    print(f'  │  [Worker {pid}] loaded in {t_load:.1f}s, playing {games} games ({num_sims} sims)')

    results = []
    for i in range(games):
        t_game = _time.perf_counter()
        pos = None
        hints = None
        if positions and i < len(positions):
            entry = positions[i]
            if isinstance(entry, tuple) and len(entry) == 2:
                pos, hints = entry
            elif isinstance(entry, dict):
                pos = entry
        samples, move_history, _analysis = self_play_game_v2(net, searcher, start_position=pos,
                                                              hint_moves=hints)
        t_game = _time.perf_counter() - t_game
        winner = 'P0' if (samples and samples[0].result > 0) else 'P1'
        print(f'  │  [W{pid}] game {i+1}/{games}: {winner} {len(move_history)}mv {t_game:.1f}s ({t_game/max(len(move_history),1):.2f}s/mv)')
        serialized = []
        for s in samples:
            serialized.append({
                'state': s.encoded_state.numpy(),
                'policy': s.policy_target,
                'player': s.player,
                'result': s.result,
                'threat': s.threat_label,
                'priority': s.priority,
            })
        result_val = samples[0].result if samples else 0.0
        results.append((serialized, [list(m) for m in move_history],
                        result_val, len(samples)))
    return results


# ---------------------------------------------------------------------------
# Metrics store (thread-safe)
# ---------------------------------------------------------------------------

class MetricsStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.iterations: List[dict] = []
        self.elo_history: List[dict] = [{'iteration': 0, 'elo': 1000.0}]
        self.current_elo: float = 1000.0
        self.total_games: int = 0
        self.current_iteration: int = 0
        self.total_iterations: int = 0
        self.is_training: bool = False

    def add_iteration(self, metrics: dict) -> None:
        with self._lock:
            self.iterations.append(metrics)
            self.total_games += metrics.get('games', 0)

    def update_elo(self, iteration: int, elo: float) -> None:
        with self._lock:
            self.current_elo = elo
            self.elo_history.append({'iteration': iteration, 'elo': round(elo, 1)})

    def get_stats(self) -> dict:
        with self._lock:
            latest = self.iterations[-1] if self.iterations else {}
            return {
                'iteration': self.current_iteration,
                'total_iterations': self.total_iterations,
                'total_games': self.total_games,
                'current_elo': round(self.current_elo, 1),
                'is_training': self.is_training,
                'buffer_size': latest.get('buffer_size', 0),
                'latest_loss': latest.get('loss', {}),
                'latest_wins': latest.get('wins', [0, 0, 0]),
                'self_play_time': latest.get('self_play_time', 0),
                'train_time': latest.get('train_time', 0),
                'avg_game_length': latest.get('avg_game_length', 0),
            }

    def get_elo_history(self) -> List[dict]:
        with self._lock:
            return list(self.elo_history)

    def get_loss_history(self) -> List[dict]:
        with self._lock:
            return [
                {
                    'iteration': m['iteration'],
                    'total': m['loss']['total'],
                    'value': m['loss']['value'],
                    'policy': m['loss']['policy'],
                }
                for m in self.iterations
                if m.get('loss') and isinstance(m['loss'], dict) and 'total' in m['loss']
            ]


# ---------------------------------------------------------------------------
# ELO evaluator
# ---------------------------------------------------------------------------

class ModelVault:
    """Stores compressed weights for every evaluated generation."""

    def __init__(self, max_models: int = 200):
        self.models: list = []  # [(iteration, state_dict_cpu_fp16), ...]
        self.max_models = max_models

    def add(self, iteration: int, state_dict: dict):
        compressed = {k: v.detach().cpu().half() for k, v in state_dict.items()}
        self.models.append((iteration, compressed))
        # Thin old models if too many (keep first, last 20, evenly spaced)
        if len(self.models) > self.max_models:
            n = len(self.models)
            keep = {0, n - 1}  # first + last
            keep.update(range(max(0, n - 20), n))  # last 20
            step = max(1, n // 50)
            keep.update(range(0, n, step))  # evenly spaced
            self.models = [self.models[i] for i in sorted(keep)]

    def get_net(self, idx: int, device) -> HexNet:
        """Load a stored model by index, return on device."""
        _, state = self.models[idx]
        state_fp32 = {k: v.float() for k, v in state.items()}
        # Detect network size from stored weights
        init_w = state_fp32.get('conv_init.weight')
        if init_w is not None:
            nf = init_w.shape[0]
            # Count res blocks
            nb = 0
            while f'res_blocks.{nb}.conv1.weight' in state_fp32:
                nb += 1
            net = HexNet(num_filters=nf, num_res_blocks=max(nb, 4))
        else:
            net = HexNet(num_filters=64, num_res_blocks=4)
        net.load_state_dict(state_fp32, strict=False)
        net.to(device)
        net.eval()
        return net

    def __len__(self):
        return len(self.models)


class GenerationalArena:
    """Evaluates current model via mini round-robin against past generations.
    Much more stable than 10-game single-opponent ELO."""

    def __init__(self, device: torch.device, games_per_opponent: int = 4,
                 num_sims: int = 30, max_opponents: int = 6):
        self.device = device
        self.games_per_opponent = games_per_opponent
        self.num_sims = num_sims
        self.max_opponents = max_opponents
        self.matchup_history: list = []  # [{gen_i: {gen_j: {w,l,d}}}]

    def evaluate(self, current_net: HexNet, vault: ModelVault,
                 current_elo: float) -> float:
        """Play mini round-robin against selected past generations."""
        current_net.eval()
        n = len(vault)
        if n == 0:
            return current_elo

        # Select opponents: first + last + evenly spaced
        opponent_indices = self._select_opponents(n)
        mcts_cur = MCTS(current_net, num_simulations=self.num_sims)

        total_wins = total_losses = total_draws = 0
        matchups = {}

        for opp_idx in opponent_indices:
            opp_net = vault.get_net(opp_idx, self.device)
            mcts_opp = MCTS(opp_net, num_simulations=self.num_sims)
            opp_iter = vault.models[opp_idx][0]

            w = l = d = 0
            for g in range(self.games_per_opponent):
                if g % 2 == 0:
                    r = self._play(mcts_cur, mcts_opp)
                else:
                    r = -self._play(mcts_opp, mcts_cur)
                if r > 0:
                    w += 1
                elif r < 0:
                    l += 1
                else:
                    d += 1

            matchups[opp_iter] = {'w': w, 'l': l, 'd': d}
            total_wins += w
            total_losses += l
            total_draws += d

            # Free GPU memory
            del opp_net, mcts_opp

        self.matchup_history.append(matchups)

        # Compute ELO delta from aggregate score
        total_games = total_wins + total_losses + total_draws
        if total_games == 0:
            return current_elo
        score = (total_wins + 0.5 * total_draws) / total_games
        # Use K=16 (lower than before) since we have more games
        new_elo = current_elo + 16 * (score - 0.5) * len(opponent_indices)
        return new_elo

    def _select_opponents(self, n: int) -> list:
        """Pick diverse opponents from vault."""
        if n <= self.max_opponents:
            return list(range(n))
        selected = {0, n - 1}  # always include first and last
        step = max(1, n // (self.max_opponents - 2))
        for i in range(1, self.max_opponents - 1):
            selected.add(min(i * step, n - 1))
        return sorted(selected)

    def _play(self, mcts_p0: MCTS, mcts_p1: MCTS) -> float:
        game = HexGame(candidate_radius=2, max_total_stones=200)
        while not game.is_terminal:
            mcts = mcts_p0 if game.current_player == 0 else mcts_p1
            policy = mcts.search(game, temperature=0.1, add_noise=False)
            if not policy:
                break
            best = max(policy, key=policy.get)
            game.place_stone(*best)
        return game.result()

    def get_matchup_summary(self) -> dict:
        """Return matchup data for dashboard display."""
        return {
            'history': self.matchup_history[-20:],
            'total_evaluations': len(self.matchup_history),
        }


# ---------------------------------------------------------------------------
# AutoTuner - self-improving hyperparameter controller
# ---------------------------------------------------------------------------

class AutoTuner:
    """Observes metrics after each iteration and adjusts hyperparams for the next.
    Rule-based + trend detection. No external ML needed."""

    def __init__(self):
        self.loss_history: list = []
        self.elo_history: list = []
        self.decisions: list = []  # log of (iteration, decision_str)

        # Current tunable params (conservative - AlphaZero-style)
        self.params = {
            'lr': 0.002,           # was 0.01 - too high, caused forgetting
            'sims': 10,
            'mix_normal': 1.00,     # pure self-play - bot generates its own positions
            'mix_catalog': 0.00,
            'mix_endgame': 0.00,
            'mix_formation': 0.00,
            'mix_sequence': 0.00,
            'hint_blend': 0.3,
            'temp_threshold': 20,
            'train_steps': 200,    # was 400 - overfitting early
        }
        self.param_history: list = []  # list of params dicts per iteration

    def _log(self, iteration: int, msg: str):
        self.decisions.append((iteration, msg))
        print(f'  │   AutoTuner: {msg}')

    def update(self, metrics: dict, iteration: int) -> dict:
        """Observe metrics, return updated params for next iteration."""
        p = self.params.copy()

        # Track histories
        total_loss = metrics.get('total_loss', 0)
        self.loss_history.append(total_loss)

        elo = metrics.get('elo')
        if elo is not None:
            self.elo_history.append(elo)

        changes = []

        # --- Learning Rate: managed by CosineAnnealingWarmRestarts (not AutoTuner) ---
        # LR is set by the scheduler, AutoTuner just reports it
        # No manual decay - cosine annealing handles everything

        # --- MCTS Sims: fixed for training, only ramp slowly ---
        # Keep sims at 50 for training (good balance of quality vs speed)
        # Sims 200 is only for online play
        if p['sims'] > 50:
            p['sims'] = 50
            changes.append(f'sims->50 (training cap)')

        # --- Game Mix: PURE SELF-PLAY (locked, no adjustments) ---
        vloss = metrics.get('value_loss', 0)
        ploss = metrics.get('policy_loss', 0)
        p['mix_normal'] = 1.0
        p['mix_endgame'] = 0.0
        p['mix_catalog'] = 0.0
        p['mix_formation'] = 0.0
        p['mix_sequence'] = 0.0

        # --- Hint Blend: decay over time ---
        p['hint_blend'] = max(0.0, 0.3 - iteration * 0.015)

        # --- Train Steps: only increase if loss is decreasing AND buffer full ---
        buf_fill = metrics.get('buffer_fill', 0)
        loss_decreasing = (len(self.loss_history) >= 3 and
                           self.loss_history[-1] < self.loss_history[-3] * 0.95)
        if buf_fill > 0.9 and loss_decreasing and p['train_steps'] < 600:
            p['train_steps'] = min(600, p['train_steps'] + 50)
            changes.append(f'train_steps->{p["train_steps"]}')

        if changes:
            self._log(iteration, ' | '.join(changes))
        else:
            self._log(iteration, 'no changes')

        # Normalize mix ratios
        total_mix = (p['mix_normal'] + p['mix_catalog'] + p['mix_endgame']
                     + p['mix_formation'] + p['mix_sequence'])
        if abs(total_mix - 1.0) > 0.01:
            for k in ['mix_normal', 'mix_catalog', 'mix_endgame',
                       'mix_formation', 'mix_sequence']:
                p[k] /= total_mix

        self.params = p
        self.param_history.append(p.copy())
        return p

    def get_status(self) -> dict:
        return {
            'params': self.params,
            'history': self.param_history[-20:],
            'decisions': self.decisions[-20:],
        }


# ---------------------------------------------------------------------------
# Dashboard observer
# ---------------------------------------------------------------------------

class DashboardObserver:
    def __init__(self, metrics: MetricsStore, sio: SocketIO):
        self.metrics = metrics
        self.sio = sio
        self._stop_flag = threading.Event()

    def on_iteration_start(self, iteration: int, total: int) -> None:
        self.metrics.current_iteration = iteration + 1
        self.metrics.total_iterations = total
        self.sio.emit('iteration_start', {'iteration': iteration + 1, 'total': total})

    def on_game_complete(self, game_idx: int, total_games: int,
                         move_history: list, result: float,
                         num_samples: int,
                         analysis_data: list = None) -> None:
        data = {
            'game_idx': game_idx,
            'total_games': total_games,
            'result': result,
            'num_moves': len(move_history),
            'moves': move_history,
        }
        if analysis_data:
            data['analysis'] = analysis_data
        print(f'  │   Emitting game_complete #{game_idx} with {len(move_history)} moves')
        self.sio.emit('game_complete', data)

    def on_iteration_complete(self, metrics: dict) -> None:
        self.metrics.add_iteration(metrics)
        self.sio.emit('iteration_complete', metrics)

    def on_training_complete(self) -> None:
        self.metrics.is_training = False
        self.sio.emit('training_complete', {})

    def should_stop(self) -> bool:
        return self._stop_flag.is_set()

    def request_stop(self):
        self._stop_flag.set()

    def reset_stop(self):
        self._stop_flag.clear()


# ---------------------------------------------------------------------------
# Training manager (parallel self-play with auto-scaling)
# ---------------------------------------------------------------------------

class TrainingManager:
    """Wraps OrcaTrainer with dashboard observer for web UI training.

    Delegates all training logic to orca/train.py's OrcaTrainer.
    The dashboard only provides the observer (for SocketIO events)
    and the start/stop controls.
    """

    def __init__(self, metrics: MetricsStore, observer: DashboardObserver,
                 sio: SocketIO, resource_monitor: ResourceMonitor):
        self.metrics = metrics
        self.observer = observer
        self.sio = sio
        self.resource = resource_monitor
        self.device = get_device()
        self.net: Optional[HexNet] = None
        self.model_vault = None
        self.arena = None
        self._thread: Optional[threading.Thread] = None
        self._trainer = None

    def start(self, num_iterations=999999, games_per_iter=100,
              train_steps=200) -> bool:
        if self.metrics.is_training:
            return False
        self.observer.reset_stop()
        self.metrics.is_training = True
        self._thread = threading.Thread(
            target=self._run,
            args=(num_iterations, games_per_iter, train_steps),
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self):
        self.observer.request_stop()

    def _run(self, num_iterations, games_per_iter, train_steps):
        """Delegate to OrcaTrainer with the dashboard observer."""
        from orca.train import OrcaTrainer

        self._trainer = OrcaTrainer(
            iterations=num_iterations,
            games_per_iter=games_per_iter,
            train_steps=train_steps,
            observer=self.observer,
            resume=True,
        )
        self._trainer.run()

        # Sync state back for dashboard access
        self.net = self._trainer.net
        self.model_vault = self._trainer.model_vault
        self.arena = self._trainer.arena
        self.metrics.is_training = False
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Background game scraper (gentle - 1 game every 10 seconds)
# ---------------------------------------------------------------------------

class BackgroundScraper:
    """Slowly downloads human games in background without overwhelming the server."""

    def __init__(self, output_path: str = "human_games.jsonl"):
        self.output_path = output_path
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        import requests as _req
        base_ts = int(time.time() * 1000)

        # Load existing IDs
        existing = set()
        if os.path.exists(self.output_path):
            with open(self.output_path, 'r') as f:
                for line in f:
                    try:
                        g = json.loads(line)
                        existing.add(g.get("id", ""))
                    except Exception:
                        pass

        page = 1
        while not self._stop.is_set():
            try:
                resp = _req.get(f"https://hexo.did.science/api/finished-games",
                                params={"page": page, "pageSize": 5, "baseTimestamp": base_ts},
                                timeout=10)
                if resp.status_code != 200:
                    self._stop.wait(30)
                    continue

                data = resp.json()
                games = data.get("games", [])
                if not games:
                    page = 1  # restart from beginning
                    self._stop.wait(60)
                    continue

                for g in games:
                    if self._stop.is_set():
                        return
                    gid = g.get("id", "")
                    if gid in existing:
                        continue

                    # Filter: only six-in-a-row wins, 10+ moves
                    result = g.get("gameResult", {})
                    if result.get("reason") != "six-in-a-row":
                        continue
                    if g.get("moveCount", 0) < 10:
                        continue

                    # Fetch detail
                    try:
                        detail = _req.get(f"https://hexo.did.science/api/finished-games/{gid}",
                                          timeout=10)
                        if detail.status_code == 200:
                            with open(self.output_path, 'a') as f:
                                f.write(json.dumps(detail.json()) + "\n")
                            existing.add(gid)
                    except Exception:
                        pass

                    # Wait 10 seconds between games - very gentle
                    self._stop.wait(10)

                page += 1
            except Exception:
                self._stop.wait(30)


# ---------------------------------------------------------------------------
# Live game broadcaster
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hex'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

resource_monitor = ResourceMonitor()  # auto-detects CPU count, sets threads
metrics_store = MetricsStore()
observer = DashboardObserver(metrics_store, socketio)
training_mgr = TrainingManager(metrics_store, observer, socketio, resource_monitor)

# Start gentle background scraper
bg_scraper = BackgroundScraper(os.path.join(os.path.dirname(__file__), 'human_games.jsonl'))
bg_scraper.start()


@app.route('/')
def index():
    return Response(DASHBOARD_HTML, mimetype='text/html')


@app.route('/api/stats')
def api_stats():
    return jsonify(metrics_store.get_stats())


@app.route('/api/elo')
def api_elo():
    result = {
        'elo_history': metrics_store.get_elo_history(),
        'vault_size': len(training_mgr.model_vault) if training_mgr and training_mgr.model_vault else 0,
        'matchups': training_mgr.arena.get_matchup_summary() if training_mgr and training_mgr.arena else {},
    }
    return jsonify(result)


@app.route('/api/losses')
def api_losses():
    return jsonify(metrics_store.get_loss_history())


@app.route('/api/resources')
def api_resources():
    snap = resource_monitor.snapshot()
    result = {
        'cpu_pct': snap['cpu_pct'],
        'ram_pct': snap['ram_pct'],
        'ram_used_gb': snap['ram_used_gb'],
        'ram_total_gb': snap['ram_total_gb'],
        'cpu_count': snap['cpu_count'],
        'workers': snap['workers'],
        'gpu_available': snap['gpu_available'],
        'history': resource_monitor.get_history(),
    }
    return jsonify(result)


@app.route('/api/autotuner')
def api_autotuner():
    if training_mgr and hasattr(training_mgr, '_trainer') and training_mgr._trainer:
        tuner = getattr(training_mgr._trainer, '_auto_tuner', None)
        if tuner:
            return jsonify(tuner.get_status())
    return jsonify({'params': {}, 'history': [], 'decisions': []})


@app.route('/api/winrates')
def api_winrates():
    data = []
    for entry in metrics_store.iterations:
        wins = entry.get('wins', [0, 0, 0])
        total = sum(wins) or 1
        data.append({
            'iteration': entry['iteration'],
            'p0': round(wins[0] / total * 100, 1),
            'p1': round(wins[1] / total * 100, 1),
            'draw': round(wins[2] / total * 100, 1),
        })
    return jsonify(data)


@app.route('/api/speed')
def api_speed():
    data = []
    for entry in metrics_store.iterations:
        games = entry.get('games', 0)
        sp_time = entry.get('self_play_time', 1)
        data.append({
            'iteration': entry['iteration'],
            'games_per_sec': round(games / max(sp_time, 0.1), 2),
            'samples': entry.get('samples', 0),
            'avg_game_length': round(entry.get('avg_game_length', 0), 1),
        })
    return jsonify(data)


@app.route('/api/gamelength')
def api_gamelength():
    data = []
    for entry in metrics_store.iterations:
        avg = entry.get('avg_game_length', 0)
        if avg > 0:
            data.append({
                'iteration': entry['iteration'],
                'avg_game_length': round(avg, 1),
            })
    return jsonify(data)


@app.route('/api/train/start', methods=['POST'])
def train_start():
    data = request.json or {}
    ok = training_mgr.start(
        num_iterations=data.get('iterations', 999999),
        games_per_iter=data.get('games_per_iter', 100),
        train_steps=data.get('train_steps', 200),
    )
    if ok:
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'}), 409


@app.route('/api/train/stop', methods=['POST'])
def train_stop():
    training_mgr.stop()
    return jsonify({'status': 'stopping'})


@app.route('/api/config')
def api_config():
    """Get current training configuration (all orca/config.py parameters)."""
    import orca.config as cfg
    return jsonify({
        'network': {
            'board_size': cfg.BOARD_SIZE,
            'num_channels': cfg.NUM_CHANNELS,
            'filters': cfg.NUM_FILTERS,
            'res_blocks': cfg.NUM_RES_BLOCKS,
        },
        'search': {
            'c_puct': cfg.C_PUCT,
            'sims': cfg.NUM_SIMULATIONS,
            'mcts_batch': cfg.MCTS_BATCH_SIZE,
            'dirichlet_alpha': cfg.DIRICHLET_ALPHA,
            'dirichlet_epsilon': cfg.DIRICHLET_EPSILON,
            'temp_threshold': cfg.TEMP_THRESHOLD,
        },
        'play_style': {
            'style': cfg.PLAY_STYLE,
            'c_blend_adjacent': cfg.C_BLEND_ADJACENT,
            'c_blend_distant': cfg.C_BLEND_DISTANT,
            'distant_explore_prob': cfg.DISTANT_EXPLORE_PROB,
        },
        'training': {
            'batch_size': cfg.BATCH_SIZE,
            'lr': cfg.LEARNING_RATE,
            'l2_reg': cfg.L2_REG,
            'buffer_size': cfg.REPLAY_BUFFER_SIZE,
        },
        'pipeline': {
            'train_steps': cfg.DEFAULT_TRAIN_STEPS,
            'games_per_iter': cfg.DEFAULT_GAMES_PER_ITER,
            'checkpoint_every': cfg.CHECKPOINT_EVERY,
            'max_workers': cfg.MAX_WORKERS,
            'games_per_future': cfg.GAMES_PER_FUTURE,
        },
        'elo': {
            'eval_every': cfg.ELO_EVAL_EVERY,
            'eval_games': cfg.ELO_EVAL_GAMES,
            'eval_sims': cfg.ELO_EVAL_SIMS,
            'max_opponents': cfg.ELO_MAX_OPPONENTS,
            'baseline_games': cfg.ELO_BASELINE_GAMES,
            'vault_max_models': cfg.VAULT_MAX_MODELS,
        },
        'curriculum': {
            'plateau_threshold': cfg.PLATEAU_THRESHOLD,
            'plateau_iters': cfg.PLATEAU_ITERS,
            'plateau_sim_boost': cfg.PLATEAU_SIM_BOOST,
        },
        'lr_schedule': {
            'cosine_t0': cfg.COSINE_T0,
            'cosine_t_mult': cfg.COSINE_T_MULT,
            'cosine_eta_min': cfg.COSINE_ETA_MIN,
        },
        'defensive': {
            'blocking_priority_boost': cfg.BLOCKING_PRIORITY_BOOST,
            'survival_priority_boost': cfg.SURVIVAL_PRIORITY_BOOST,
            'use_ab_hybrid': cfg.USE_AB_HYBRID,
            'ab_hybrid_depth': cfg.AB_HYBRID_DEPTH,
            'threat_policy_blend': cfg.THREAT_POLICY_BLEND,
        },
        'mixed_precision': {
            'enabled': cfg.USE_MIXED_PRECISION,
            'grad_clip_norm': cfg.GRAD_CLIP_NORM,
        },
    })


@app.route('/api/config', methods=['POST'])
def api_config_update():
    """Update training config (applies to next OrcaTrainer run)."""
    import orca.config as cfg
    data = request.json or {}
    updated = []
    # Mapping of JSON key -> (config attr, type, optional validator)
    _map = {
        # Search
        'c_puct':              ('C_PUCT', float),
        'sims':                ('NUM_SIMULATIONS', int),
        'mcts_batch':          ('MCTS_BATCH_SIZE', int),
        'dirichlet_alpha':     ('DIRICHLET_ALPHA', float),
        'dirichlet_epsilon':   ('DIRICHLET_EPSILON', float),
        'temp_threshold':      ('TEMP_THRESHOLD', int),
        # Play style
        'play_style':          ('PLAY_STYLE', str),
        'c_blend_adjacent':    ('C_BLEND_ADJACENT', float),
        'c_blend_distant':     ('C_BLEND_DISTANT', float),
        'distant_explore_prob':('DISTANT_EXPLORE_PROB', float),
        # Training
        'batch_size':          ('BATCH_SIZE', int),
        'lr':                  ('LEARNING_RATE', float),
        'l2_reg':              ('L2_REG', float),
        'buffer_size':         ('REPLAY_BUFFER_SIZE', int),
        # Pipeline
        'train_steps':         ('DEFAULT_TRAIN_STEPS', int),
        'games_per_iter':      ('DEFAULT_GAMES_PER_ITER', int),
        'checkpoint_every':    ('CHECKPOINT_EVERY', int),
        'max_workers':         ('MAX_WORKERS', int),
        'games_per_future':    ('GAMES_PER_FUTURE', int),
        # ELO
        'eval_every':          ('ELO_EVAL_EVERY', int),
        'eval_games':          ('ELO_EVAL_GAMES', int),
        'eval_sims':           ('ELO_EVAL_SIMS', int),
        'max_opponents':       ('ELO_MAX_OPPONENTS', int),
        'baseline_games':      ('ELO_BASELINE_GAMES', int),
        'vault_max_models':    ('VAULT_MAX_MODELS', int),
        # Curriculum / plateau
        'plateau_threshold':   ('PLATEAU_THRESHOLD', int),
        'plateau_iters':       ('PLATEAU_ITERS', int),
        'plateau_sim_boost':   ('PLATEAU_SIM_BOOST', int),
        # LR schedule
        'cosine_t0':           ('COSINE_T0', int),
        'cosine_t_mult':       ('COSINE_T_MULT', int),
        'cosine_eta_min':      ('COSINE_ETA_MIN', float),
        # Defensive
        'blocking_priority_boost': ('BLOCKING_PRIORITY_BOOST', float),
        'survival_priority_boost': ('SURVIVAL_PRIORITY_BOOST', float),
        'use_ab_hybrid':       ('USE_AB_HYBRID', bool),
        'ab_hybrid_depth':     ('AB_HYBRID_DEPTH', int),
        'threat_policy_blend': ('THREAT_POLICY_BLEND', float),
        # Mixed precision
        'mixed_precision':     ('USE_MIXED_PRECISION', bool),
        'grad_clip_norm':      ('GRAD_CLIP_NORM', float),
    }
    for key, (attr, typ) in _map.items():
        if key in data:
            val = data[key]
            if typ is bool:
                val = val if isinstance(val, bool) else str(val).lower() in ('true', '1')
            else:
                val = typ(val)
            setattr(cfg, attr, val)
            updated.append(key)
    return jsonify({'updated': updated, 'status': 'ok'})


# ---------------------------------------------------------------------------
# Online Play Manager (runs browser_player.py as subprocess)
# ---------------------------------------------------------------------------

class OnlinePlayManager:
    """Manages the browser bot subprocess from the dashboard."""

    def __init__(self):
        self._proc = None
        self._log_lines: list = []
        self._max_log = 200
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.is_running = False
        self._reader_thread = None

    def start(self, games: int = 9999, fast: bool = False):
        if self._proc and self._proc.poll() is None:
            return False
        import subprocess
        cmd = [
            sys.executable, '-u', 'browser_player.py',
            '--auto', '--games', str(games),
        ]
        if fast:
            cmd.append('--fast')
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=os.path.dirname(__file__),
            env=env,
        )
        self.is_running = True
        self._log_lines = []
        # Reader thread to capture output
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()
        return True

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self.is_running = False

    def _read_output(self):
        try:
            for line in self._proc.stdout:
                line = line.rstrip()
                if line:
                    self._log_lines.append(line)
                    if len(self._log_lines) > self._max_log:
                        self._log_lines = self._log_lines[-self._max_log:]
                    # Parse wins/losses
                    if '🏆' in line or 'WIN' in line:
                        self.wins += 1
                        self.games_played += 1
                    elif '✗' in line or 'Lost' in line:
                        self.losses += 1
                        self.games_played += 1
        except Exception:
            pass
        self.is_running = False

    def get_status(self):
        return {
            'running': self.is_running,
            'games': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'winrate': round(self.wins / max(self.games_played, 1) * 100, 1),
            'log': self._log_lines[-30:],
        }


online_mgr = OnlinePlayManager()


@app.route('/api/online/start', methods=['POST'])
def online_start():
    data = request.json or {}
    ok = online_mgr.start(
        games=data.get('games', 9999),
        fast=data.get('fast', False),
    )
    return jsonify({'status': 'started' if ok else 'already_running'})


@app.route('/api/online/stop', methods=['POST'])
def online_stop():
    online_mgr.stop()
    return jsonify({'status': 'stopped'})


@app.route('/api/online/status')
def online_status():
    return jsonify(online_mgr.get_status())


@app.route('/api/network_vis')
def network_vis():
    """Generate hex-grid network visualization on the fly."""
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon

    try:
        if training_manager.net is None:
            return 'No network loaded', 404

        net = training_manager.net
        net.eval()

        game = HexGame(candidate_radius=2, max_total_stones=200)
        for q, r in [(0,0), (2,0), (2,-1), (1,0), (0,1)]:
            game.place_stone(q, r)

        encoded, oq, orr = encode_state(game)
        x = encoded.unsqueeze(0).to(next(net.parameters()).device)

        with torch.no_grad():
            p_logits, value, threats = net(x)
            policy = torch.softmax(p_logits, dim=1)[0].cpu().numpy().reshape(19, 19)

        def draw_hex(ax, data, title, cmap='hot', vmin=None, vmax=None):
            if vmin is None: vmin = data.min()
            if vmax is None: vmax = data.max()
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cm = plt.get_cmap(cmap)
            sz = 0.55
            for r in range(19):
                for q in range(19):
                    px = sz * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
                    py = sz * (3/2 * r)
                    h = RegularPolygon((px, -py), 6, radius=sz*0.58, orientation=0,
                                       facecolor=cm(norm(data[r, q])),
                                       edgecolor='#555', linewidth=0.2)
                    ax.add_patch(h)
            ax.set_xlim(-1, sz * np.sqrt(3) * 19 + 1)
            ax.set_ylim(-sz * 1.5 * 19 - 1, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(title, fontsize=9, fontweight='bold')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        board = encoded[0].numpy() - encoded[1].numpy()
        draw_hex(ax1, board, 'Board (Red=P0, Blue=P1)', cmap='RdBu_r', vmin=-1, vmax=1)
        draw_hex(ax2, policy, f'Policy (Value: {value[0].item():.3f})', cmap='YlOrRd', vmin=0)
        fig.suptitle('Network Insight', fontsize=12, fontweight='bold')
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), 200, {'Content-Type': 'image/png'}
    except Exception as e:
        return f'Error: {e}', 500


@socketio.on('connect')
def on_connect():
    pass


# ---------------------------------------------------------------------------
# Frontend HTML (high-res, responsive canvases, resource monitoring)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HEX BOT</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#fff;color:#000;font-family:'SF Mono','Courier New',monospace;font-size:13px;line-height:1.5;
  display:flex;flex-direction:column;height:100vh;overflow:hidden}
header{border-bottom:1px solid #000;padding:14px 24px;display:flex;align-items:center;gap:20px}
.title{font-size:15px;font-weight:700;letter-spacing:4px;text-transform:uppercase}
.status{margin-left:auto;letter-spacing:3px;font-size:11px;font-weight:700}
main{display:flex;flex:1;overflow:hidden}
.left{width:50%;border-right:1px solid #000;padding:20px;display:flex;flex-direction:column;align-items:center;overflow:hidden}
.right{width:50%;display:flex;flex-direction:column;overflow:hidden}
.chart-box{padding:16px 20px;border-bottom:1px solid #000;display:flex;flex-direction:column}
.chart-box:last-child{border-bottom:none}
.chart-header{font-weight:700;letter-spacing:2px;font-size:10px;text-transform:uppercase;
  cursor:pointer;user-select:none;display:flex;align-items:center;gap:6px}
.chart-header .toggle{font-size:8px;transition:transform .15s}
.chart-header .toggle.collapsed{transform:rotate(-90deg)}
.chart-body{overflow:hidden;transition:max-height .25s ease,opacity .2s ease}
.chart-body.collapsed{max-height:0!important;opacity:0;padding:0}
.canvas-wrap{position:relative;min-height:180px;height:180px;margin-top:8px}
.canvas-wrap canvas{position:absolute;top:0;left:0;width:100%;height:100%}
#hex-canvas-wrap{flex:1;position:relative;width:100%;min-height:0}
#hex-canvas{position:absolute;top:0;left:0;width:100%;height:100%;border:1px solid #eee}
.game-info{font-size:11px;letter-spacing:1px;min-height:18px;margin-top:8px;flex-shrink:0}
footer{padding:12px 24px;display:flex;gap:20px;flex-wrap:wrap;font-size:11px;letter-spacing:.5px;border-top:1px solid #000}
footer b{font-weight:700}
footer span{white-space:nowrap}
.sep{color:#ccc}
.res-bar{display:flex;gap:4px;align-items:center}
.res-meter{width:40px;height:8px;border:1px solid #000;display:inline-block;position:relative;vertical-align:middle}
.res-meter-fill{height:100%;background:#000;transition:width .3s}
#live-stats{width:100%;padding:0 0 12px 0;font:11px 'SF Mono','Courier New',monospace;
  border-bottom:1px solid #eee;margin-bottom:12px}
#live-stats .row{display:flex;justify-content:space-between;margin-bottom:4px}
#progress-wrap{width:100%;margin-bottom:8px;display:none}
.progress-bar{display:flex;align-items:center;gap:8px}
.progress-track{flex:1;height:6px;background:#eee;border-radius:3px;overflow:hidden}
.progress-fill{height:100%;background:#000;width:0%;transition:width .2s}
.progress-label{font:700 10px 'Courier New';min-width:120px;text-align:right}
.gh-item{cursor:pointer;padding:1px 3px;border-radius:2px;display:inline-block;margin:0 1px;font-size:8px}
.gh-item:hover{background:#eee}
.gh-item.active{background:#000;color:#fff}
.conn-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-left:8px;vertical-align:middle}
.overlay-toggles{display:flex;gap:6px;margin-top:6px;align-items:center}
.overlay-toggles .obtn{cursor:pointer;padding:2px 8px;border:1px solid #000;font:bold 10px 'SF Mono',monospace;
  letter-spacing:1px;user-select:none;background:#fff;color:#000;transition:all .15s}
.overlay-toggles .obtn.active{background:#000;color:#fff}
#value-chart-wrap{width:100%;height:60px;position:relative;margin-top:6px;border:1px solid #eee}
#value-chart{position:absolute;top:0;left:0;width:100%;height:100%}
.tab-bar{display:flex;border-bottom:1px solid #000;flex-shrink:0}
.tab{padding:8px 20px;font:bold 10px 'SF Mono','Courier New',monospace;letter-spacing:2px;text-transform:uppercase;
  cursor:pointer;user-select:none;border-right:1px solid #000;transition:background .15s}
.tab:hover{background:#f5f5f5}
.tab.active{background:#000;color:#fff}
.cfg-section{margin-bottom:2px}
.cfg-section-header{font:bold 10px 'SF Mono','Courier New',monospace;letter-spacing:2px;text-transform:uppercase;
  cursor:pointer;user-select:none;padding:8px 0;display:flex;align-items:center;gap:6px;border-bottom:1px solid #eee}
.cfg-section-header .toggle{font-size:8px;transition:transform .15s}
.cfg-section-header .toggle.collapsed{transform:rotate(-90deg)}
.cfg-section-body{padding:8px 0 4px 0;transition:max-height .25s ease,opacity .2s ease;overflow:hidden}
.cfg-section-body.collapsed{max-height:0!important;opacity:0;padding:0}
.cfg-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.cfg-row label{width:130px;font:bold 11px 'SF Mono','Courier New',monospace;flex-shrink:0;cursor:help;
  border-bottom:1px dotted #ccc}
.cfg-row input[type=number]{width:100px;font:12px 'SF Mono','Courier New',monospace;border:1px solid #ccc;
  padding:3px 6px;background:#fafafa;outline:none}
.cfg-row input[type=number]:focus{border-color:#000;background:#fff}
.cfg-row input[type=checkbox]{accent-color:#000;width:16px;height:16px}
.cfg-row select{font:12px 'SF Mono','Courier New',monospace;border:1px solid #ccc;padding:3px 6px;background:#fafafa;outline:none}
.cfg-row select:focus{border-color:#000;background:#fff}
.cfg-row input[type=range]{flex:1;accent-color:#000}
.cfg-row span.val{width:50px;text-align:right;font-size:10px}
.cfg-ro{font:12px 'SF Mono','Courier New',monospace;color:#666}
.cfg-hint{font:10px 'SF Mono','Courier New',monospace;color:#999;margin:-2px 0 8px 0}
body.dark{background:#111;color:#ddd;filter:invert(1) hue-rotate(180deg)}
body.dark canvas,body.dark img{filter:invert(1) hue-rotate(180deg)}
body.dark header{border-color:#444}
body.dark .left{border-color:#444}
body.dark footer{border-color:#444}
body.dark .chart-box{border-color:#333}
body.dark .tab.active{background:#ddd;color:#111}
body.dark .cfg-section-header{border-color:#333}
#lights-out-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:#000;z-index:9999;
  align-items:center;justify-content:center;flex-direction:column;gap:20px}
#lights-out-overlay.active{display:flex}
#lights-out-btn{font:bold 12px 'SF Mono','Courier New',monospace;padding:12px 32px;border:2px solid #333;
  background:#111;color:#555;cursor:pointer;letter-spacing:3px;text-transform:uppercase;transition:all .2s}
#lights-out-btn:hover{border-color:#fff;color:#fff}
#lights-out-status{font:10px 'SF Mono','Courier New',monospace;color:#333;letter-spacing:2px}
</style>
</head>
<body>
<header>
  <span class="title">Hex Bot Training</span>
  <button id="btn-start" onclick="startTraining()" style="background:#000;color:#fff;border:none;padding:4px 10px;font:13px 'Courier New';cursor:pointer">&#9654;</button><button id="btn-stop" onclick="stopTraining()" style="background:#fff;color:#000;border:1px solid #000;padding:4px 10px;font:13px 'Courier New';cursor:pointer">&#9632;</button><span class="status" id="status">IDLE</span>
  <span class="conn-dot" id="conn-dot" style="background:#ccc" title="Disconnected"></span>

</header>
<main>
  <div class="left">
    <div id="live-stats">
      <div class="row">
        <span><b>Iteration</b> <span id="ls-iter">&mdash;</span></span>
        <span><b>Workers</b> <span id="ls-workers">&mdash;</span></span>
        <span><b>Games</b> <span id="ls-games">&mdash;</span></span>
      </div>
      <div class="row">
        <span><b>Avg Length</b> <span id="ls-len">&mdash;</span></span>
        <span><b>Win%</b> P0:<span id="ls-w0">&mdash;</span> P1:<span id="ls-w1">&mdash;</span></span>
        <span><b>Loss</b> <span id="ls-loss">&mdash;</span></span>
      </div>
    </div>
    <div id="progress-wrap">
      <div class="progress-bar">
        <div class="progress-track">
          <div class="progress-fill" id="progress-fill"></div>
        </div>
        <span class="progress-label" id="progress-label"></span>
      </div>
    </div>
    <div style="font-weight:700;letter-spacing:2px;font-size:10px;text-transform:uppercase;margin-bottom:4px">
      Training Game #<span id="game-num">&mdash;</span>
    </div>
    <div id="hex-canvas-wrap"><canvas id="hex-canvas"></canvas></div>
    <div class="game-info" id="game-info">Waiting for games...</div>
    <div class="overlay-toggles">
      <span class="obtn" id="btn-val" onclick="toggleOverlay('value')" title="Value/quality overlay (V)">V</span>
      <span class="obtn" id="btn-threat" onclick="toggleOverlay('threat')" title="Threat overlay (T)">T</span>
      <span style="font:8px 'SF Mono',monospace;color:#999;margin-left:4px" id="analysis-label"></span>
    </div>
    <div id="value-chart-wrap" style="display:none"><canvas id="value-chart"></canvas></div>
    <div id="game-history" style="width:100%;margin-top:6px;max-height:48px;overflow-y:auto;overflow-x:hidden;
      font:9px 'SF Mono',monospace;border-top:1px solid #eee;padding-top:4px;line-height:1.6">
    </div>
    <div style="font:8px 'SF Mono',monospace;color:#bbb;margin-top:2px">
      Space: pause &middot; R: restart &middot; V: value &middot; T: threats
    </div>
  </div>
  <div class="right">
    <div class="tab-bar">
      <span class="tab active" data-tab="charts" onclick="switchTab('charts')">Charts</span>
      <span class="tab" data-tab="settings" onclick="switchTab('settings')">Settings</span>
    </div>
    <div id="tab-charts" class="tab-content" style="display:flex;flex-direction:column;overflow-y:auto;flex:1">
      <div class="chart-box" id="box-elo">
        <div class="chart-header" onclick="toggleChart('elo')">
          <span class="toggle" id="tog-elo">&#9660;</span> Elo Progression
        </div>
        <div class="chart-body" id="body-elo">
          <div class="canvas-wrap"><canvas id="elo-chart"></canvas></div>
        </div>
      </div>
      <div class="chart-box" id="box-loss">
        <div class="chart-header" onclick="toggleChart('loss')">
          <span class="toggle" id="tog-loss">&#9660;</span> Loss Curves
        </div>
        <div class="chart-body" id="body-loss">
          <div class="canvas-wrap"><canvas id="loss-chart"></canvas></div>
        </div>
      </div>
      <div class="chart-box" id="box-winrate">
        <div class="chart-header" onclick="toggleChart('winrate')">
          <span class="toggle" id="tog-winrate">&#9660;</span> Win Rates
        </div>
        <div class="chart-body" id="body-winrate">
          <div class="canvas-wrap"><canvas id="winrate-chart"></canvas></div>
        </div>
      </div>
      <div class="chart-box" id="box-gamelength">
        <div class="chart-header" onclick="toggleChart('gamelength')">
          <span class="toggle" id="tog-gamelength">&#9660;</span> Game Length
        </div>
        <div class="chart-body" id="body-gamelength">
          <div class="canvas-wrap"><canvas id="gamelength-chart"></canvas></div>
        </div>
      </div>
      <div class="chart-box" id="box-speed">
        <div class="chart-header" onclick="toggleChart('speed')">
          <span class="toggle" id="tog-speed">&#9660;</span> Training Speed
        </div>
        <div class="chart-body" id="body-speed">
          <div class="canvas-wrap"><canvas id="speed-chart"></canvas></div>
        </div>
      </div>
      <div class="chart-box" id="box-resources">
        <div class="chart-header" onclick="toggleChart('resources')">
          <span class="toggle" id="tog-resources">&#9660;</span> Resources (CPU / RAM)
        </div>
        <div class="chart-body" id="body-resources">
          <div class="canvas-wrap" style="min-height:140px;height:140px"><canvas id="res-chart"></canvas></div>
        </div>
      </div>
    </div>
    <div id="tab-settings" class="tab-content" style="display:none;overflow-y:auto;flex:1;padding:16px 20px">
      <div id="cfg-status" style="font:bold 10px 'SF Mono',monospace;color:#090;min-height:16px;margin-bottom:8px"></div>

      <div class="cfg-section">
        <div class="cfg-section-header" onclick="toggleCfgSection(this)">
          <span class="toggle">&#9660;</span> Network
        </div>
        <div class="cfg-section-body">
          <div class="cfg-row"><label title="Encoding grid size (19x19 captures all relevant hexes)">Board size</label><span class="cfg-ro" id="cfg-board_size">19</span></div>
          <div class="cfg-row"><label title="Input feature planes: 5 board planes + 2 threat planes">Input channels</label><span class="cfg-ro" id="cfg-num_channels">7</span></div>
          <div class="cfg-row"><label title="Conv2d filter width. 128 = fast, 256 = stronger but slower">Filters</label><span class="cfg-ro" id="cfg-filters">128</span></div>
          <div class="cfg-row"><label title="Depth of the residual tower. More blocks = stronger but slower">Res blocks</label><span class="cfg-ro" id="cfg-res_blocks">12</span></div>
          <div class="cfg-hint">Network architecture is read-only (set via --config at launch)</div>
        </div>
      </div>

      <div class="cfg-section">
        <div class="cfg-section-header" onclick="toggleCfgSection(this)">
          <span class="toggle">&#9660;</span> Search (MCTS)
        </div>
        <div class="cfg-section-body">
          <div class="cfg-row"><label title="Exploration constant balancing prior vs visit count. Higher = more exploration, lower = more exploitation">C PUCT</label><input type="number" id="cfg-c_puct" step="0.1" min="0.1" max="10" onchange="cfgUpdate('c_puct',+this.value)"></div>
          <div class="cfg-row"><label title="MCTS simulations per move. Target value that curriculum scales toward (25% early, 100% at iter 60+). More sims = stronger but slower">Simulations</label><input type="number" id="cfg-sims" step="10" min="10" max="1600" onchange="cfgUpdate('sims',+this.value)"></div>
          <div class="cfg-row"><label title="Positions batched per NN forward pass during MCTS. Higher = better GPU utilization">MCTS batch</label><input type="number" id="cfg-mcts_batch" step="8" min="1" max="512" onchange="cfgUpdate('mcts_batch',+this.value)"></div>
          <div class="cfg-row"><label title="Root noise parameter. Lower = more concentrated noise (less diverse), higher = more uniform. 0.03 for Go, 0.3 for hex">Dirichlet alpha</label><input type="number" id="cfg-dirichlet_alpha" step="0.01" min="0.01" max="1" onchange="cfgUpdate('dirichlet_alpha',+this.value)"></div>
          <div class="cfg-row"><label title="Fraction of root prior replaced by Dirichlet noise. 0 = no noise, 0.25 = standard AlphaZero">Dirichlet epsilon</label><input type="number" id="cfg-dirichlet_epsilon" step="0.05" min="0" max="1" onchange="cfgUpdate('dirichlet_epsilon',+this.value)"></div>
          <div class="cfg-row"><label title="Move number after which MCTS switches from sampling to greedy (argmax). Lower = less randomness in mid/endgame">Temp threshold</label><input type="number" id="cfg-temp_threshold" step="1" min="0" max="100" onchange="cfgUpdate('temp_threshold',+this.value)"></div>
        </div>
      </div>

      <div class="cfg-section">
        <div class="cfg-section-header" onclick="toggleCfgSection(this)">
          <span class="toggle">&#9660;</span> Training
        </div>
        <div class="cfg-section-body">
          <div class="cfg-row"><label title="Samples per gradient step. Larger = more stable but uses more memory. 1024 is good for most GPUs">Batch size</label><input type="number" id="cfg-batch_size" step="64" min="32" max="4096" onchange="cfgUpdate('batch_size',+this.value)"></div>
          <div class="cfg-row"><label title="Adam optimizer learning rate. Cosine schedule cycles between this and eta_min">Learning rate</label><input type="number" id="cfg-lr" step="0.0001" min="0.00001" max="0.1" onchange="cfgUpdate('lr',+this.value)"></div>
          <div class="cfg-row"><label title="Weight decay (L2 regularization). Prevents overfitting by penalizing large weights">L2 reg</label><input type="number" id="cfg-l2_reg" step="0.00001" min="0" max="0.01" onchange="cfgUpdate('l2_reg',+this.value)"></div>
          <div class="cfg-row"><label title="Experience replay capacity. Older samples are evicted when full. Larger = more diverse training data">Buffer size</label><input type="number" id="cfg-buffer_size" step="50000" min="10000" max="2000000" onchange="cfgUpdate('buffer_size',+this.value)"></div>
        </div>
      </div>

      <div class="cfg-section">
        <div class="cfg-section-header" onclick="toggleCfgSection(this)">
          <span class="toggle">&#9660;</span> Pipeline
        </div>
        <div class="cfg-section-body">
          <div class="cfg-row"><label title="Gradient update steps per iteration. More = better learning per iteration but slower">Train steps</label><input type="number" id="cfg-train_steps" step="10" min="10" max="2000" onchange="cfgUpdate('train_steps',+this.value)"></div>
          <div class="cfg-row"><label title="Base self-play games per iteration. Curriculum scales this (1.5x early, 1x at full sims)">Games / iter</label><input type="number" id="cfg-games_per_iter" step="5" min="1" max="500" onchange="cfgUpdate('games_per_iter',+this.value)"></div>
          <div class="cfg-row"><label title="Save model checkpoint every N iterations">Checkpoint every</label><input type="number" id="cfg-checkpoint_every" step="1" min="1" max="100" onchange="cfgUpdate('checkpoint_every',+this.value)"></div>
          <div class="cfg-row"><label title="Parallel self-play workers. Threads on CUDA (shared GPU), processes on MPS/CPU">Max workers</label><input type="number" id="cfg-max_workers" step="1" min="1" max="32" onchange="cfgUpdate('max_workers',+this.value)"></div>
          <div class="cfg-row"><label title="Games assigned per subprocess future. Higher = less overhead, lower = better load balancing">Games / future</label><input type="number" id="cfg-games_per_future" step="1" min="1" max="10" onchange="cfgUpdate('games_per_future',+this.value)"></div>
        </div>
      </div>

      <div class="cfg-section">
        <div class="cfg-section-header" onclick="toggleCfgSection(this)">
          <span class="toggle">&#9660;</span> ELO Evaluation
        </div>
        <div class="cfg-section-body">
          <div class="cfg-row"><label title="Run ELO evaluation every N training iterations">Eval every</label><input type="number" id="cfg-eval_every" step="1" min="1" max="50" onchange="cfgUpdate('eval_every',+this.value)"></div>
          <div class="cfg-row"><label title="Games played against each past opponent per evaluation round">Eval games</label><input type="number" id="cfg-eval_games" step="1" min="1" max="50" onchange="cfgUpdate('eval_games',+this.value)"></div>
          <div class="cfg-row"><label title="MCTS simulations during ELO games. Lower = faster eval but noisier ratings">Eval sims</label><input type="number" id="cfg-eval_sims" step="5" min="5" max="400" onchange="cfgUpdate('eval_sims',+this.value)"></div>
          <div class="cfg-row"><label title="Max past model versions to play against per evaluation. More = better ELO estimate but slower">Max opponents</label><input type="number" id="cfg-max_opponents" step="1" min="1" max="20" onchange="cfgUpdate('max_opponents',+this.value)"></div>
          <div class="cfg-row"><label title="Games vs random (~500 ELO) + heuristic (~1000 ELO) baselines. 0 = disable baseline anchoring">Baseline games</label><input type="number" id="cfg-baseline_games" step="1" min="0" max="20" onchange="cfgUpdate('baseline_games',+this.value)"></div>
          <div class="cfg-row"><label title="Maximum stored model snapshots in the vault. Old models are evicted when full">Vault max models</label><input type="number" id="cfg-vault_max_models" step="10" min="10" max="1000" onchange="cfgUpdate('vault_max_models',+this.value)"></div>
        </div>
      </div>

      <div class="cfg-section">
        <div class="cfg-section-header" onclick="toggleCfgSection(this)">
          <span class="toggle">&#9660;</span> Play Style
        </div>
        <div class="cfg-section-body">
          <div class="cfg-row"><label title="Colony strategy: 'distant' places stones far apart for multi-front pressure, 'close' keeps groups connected">Style</label>
            <select id="cfg-play_style" onchange="cfgUpdate('play_style',this.value)">
              <option value="distant">distant</option>
              <option value="close">close</option>
            </select>
          </div>
          <div class="cfg-row"><label title="Heuristic weight blended into MCTS prior for moves adjacent to existing stones">C blend adjacent</label><input type="number" id="cfg-c_blend_adjacent" step="0.01" min="0" max="1" onchange="cfgUpdate('c_blend_adjacent',+this.value)"></div>
          <div class="cfg-row"><label title="Heuristic weight blended into MCTS prior for distant moves (colony placements)">C blend distant</label><input type="number" id="cfg-c_blend_distant" step="0.01" min="0" max="1" onchange="cfgUpdate('c_blend_distant',+this.value)"></div>
          <div class="cfg-row"><label title="Probability of injecting distant candidate moves into the policy. Helps explore multi-front strategies">Explore prob</label><input type="number" id="cfg-distant_explore_prob" step="0.05" min="0" max="1" onchange="cfgUpdate('distant_explore_prob',+this.value)"></div>
        </div>
      </div>

      <div class="cfg-section">
        <div class="cfg-section-header" onclick="toggleCfgSection(this)">
          <span class="toggle">&#9660;</span> Curriculum / Plateau
        </div>
        <div class="cfg-section-body">
          <div class="cfg-row"><label title="ELO change below this threshold counts as stalled. Triggers sim boost after plateau_iters of stall">Plateau threshold</label><input type="number" id="cfg-plateau_threshold" step="1" min="1" max="100" onchange="cfgUpdate('plateau_threshold',+this.value)"></div>
          <div class="cfg-row"><label title="Consecutive stalled iterations before boosting sims to break through plateau">Plateau iters</label><input type="number" id="cfg-plateau_iters" step="1" min="1" max="50" onchange="cfgUpdate('plateau_iters',+this.value)"></div>
          <div class="cfg-row"><label title="Extra simulations added when plateau is detected (capped at 2x configured sims)">Plateau sim boost</label><input type="number" id="cfg-plateau_sim_boost" step="10" min="0" max="200" onchange="cfgUpdate('plateau_sim_boost',+this.value)"></div>
        </div>
      </div>

      <div class="cfg-section">
        <div class="cfg-section-header" onclick="toggleCfgSection(this)">
          <span class="toggle">&#9660;</span> LR Schedule
        </div>
        <div class="cfg-section-body">
          <div class="cfg-row"><label title="First cosine cycle length in iterations. LR decays from max to eta_min over T0 iters, then restarts">Cosine T0</label><input type="number" id="cfg-cosine_t0" step="5" min="5" max="500" onchange="cfgUpdate('cosine_t0',+this.value)"></div>
          <div class="cfg-row"><label title="Each restart multiplies the cycle length by this factor. 2 = first cycle 50 iters, next 100, then 200...">Cosine T mult</label><input type="number" id="cfg-cosine_t_mult" step="1" min="1" max="10" onchange="cfgUpdate('cosine_t_mult',+this.value)"></div>
          <div class="cfg-row"><label title="Minimum learning rate at the bottom of each cosine cycle">Cosine eta min</label><input type="number" id="cfg-cosine_eta_min" step="0.00001" min="0" max="0.01" onchange="cfgUpdate('cosine_eta_min',+this.value)"></div>
        </div>
      </div>

      <div class="cfg-section">
        <div class="cfg-section-header" onclick="toggleCfgSection(this)">
          <span class="toggle">&#9660;</span> Defensive Training
        </div>
        <div class="cfg-section-body">
          <div class="cfg-row"><label title="Replay buffer priority multiplier for moves that block opponent threats. Higher = train more on blocking positions">Blocking boost</label><input type="number" id="cfg-blocking_priority_boost" step="0.5" min="0" max="10" onchange="cfgUpdate('blocking_priority_boost',+this.value)"></div>
          <div class="cfg-row"><label title="Priority boost for positions where the player survived an opponent's threat sequence">Survival boost</label><input type="number" id="cfg-survival_priority_boost" step="0.5" min="0" max="10" onchange="cfgUpdate('survival_priority_boost',+this.value)"></div>
          <div class="cfg-row"><label title="Run alpha-beta pre-check before MCTS to detect forced wins/blocks. Catches tactical shots MCTS might miss">AB hybrid</label><input type="checkbox" id="cfg-use_ab_hybrid" onchange="cfgUpdate('use_ab_hybrid',this.checked)"></div>
          <div class="cfg-row"><label title="Alpha-beta search depth for the hybrid pre-check. 4 is fast, 6+ catches deeper tactics but slows self-play">AB depth</label><input type="number" id="cfg-ab_hybrid_depth" step="1" min="0" max="12" onchange="cfgUpdate('ab_hybrid_depth',+this.value)"></div>
          <div class="cfg-row"><label title="Blend threat head spatial map into policy logits. The threat conv learns WHERE threats are and boosts those cells in MCTS policy. 0 = disabled, 0.3 = standard, higher = more threat-driven play">Threat blend</label><input type="number" id="cfg-threat_policy_blend" step="0.05" min="0" max="2" onchange="cfgUpdate('threat_policy_blend',+this.value)"></div>
        </div>
      </div>

      <div class="cfg-section">
        <div class="cfg-section-header" onclick="toggleCfgSection(this)">
          <span class="toggle">&#9660;</span> Mixed Precision
        </div>
        <div class="cfg-section-body">
          <div class="cfg-row"><label title="FP16 mixed precision training. ~2x faster on CUDA GPUs with Tensor Cores. Ignored on MPS/CPU">Enabled</label><input type="checkbox" id="cfg-mixed_precision" onchange="cfgUpdate('mixed_precision',this.checked)"></div>
          <div class="cfg-row"><label title="Max gradient norm for clipping. Prevents exploding gradients. 0 = disabled, 1.0 = standard">Grad clip norm</label><input type="number" id="cfg-grad_clip_norm" step="0.1" min="0" max="10" onchange="cfgUpdate('grad_clip_norm',+this.value)"></div>
          <div class="cfg-hint">Mixed precision only effective on CUDA with Tensor Cores</div>
        </div>
      </div>

      <div class="cfg-section" style="margin-top:16px;border-top:1px solid #eee;padding-top:12px">
        <div class="cfg-section-header" onclick="toggleCfgSection(this)">
          <span class="toggle">&#9660;</span> Display
        </div>
        <div class="cfg-section-body">
          <div class="cfg-row"><label title="Milliseconds between moves during game replay animation">Replay speed</label>
            <input type="range" id="set-speed" min="50" max="500" value="120" step="10"
              oninput="saveSetting('replaySpeed',+this.value);el('set-speed-val').textContent=this.value+'ms'">
            <span class="val" id="set-speed-val">120ms</span>
          </div>
          <div class="cfg-row"><label title="Size of stone dots on the hex board">Dot size</label>
            <input type="range" id="set-dotsize" min="1" max="5" value="2" step="0.5"
              oninput="saveSetting('dotSize',+this.value);el('set-dotsize-val').textContent=this.value;drawHex()">
            <span class="val" id="set-dotsize-val">2</span>
          </div>
          <div class="cfg-row"><label title="Radius of the empty hex grid drawn around stones">Grid radius</label>
            <input type="range" id="set-radius" min="1" max="4" value="2" step="1"
              oninput="saveSetting('emptyHexRadius',+this.value);el('set-radius-val').textContent=this.value;drawHex()">
            <span class="val" id="set-radius-val">2</span>
          </div>
          <div class="cfg-row"><label title="Show move order numbers on each stone">Move numbers</label><input type="checkbox" id="set-movenums" checked onchange="saveSetting('showMoveNums',this.checked);drawHex()"></div>
          <div class="cfg-row"><label title="Auto-refresh charts after each iteration completes">Auto-refresh</label><input type="checkbox" id="set-autorefresh" checked onchange="saveSetting('autoRefresh',this.checked)"></div>
          <div class="cfg-row"><label title="Invert all colors (dark mode)">Dark mode</label><input type="checkbox" id="set-darkmode" onchange="toggleDarkMode(this.checked)"></div>
          <div class="cfg-row"><label title="Turn everything black except one button to undo">Lights out</label><button onclick="lightsOut()" style="font:bold 10px 'SF Mono',monospace;padding:3px 10px;border:1px solid #000;background:#000;color:#fff;cursor:pointer">LIGHTS OUT</button></div>
        </div>
      </div>
    </div>
  </div>
</main>
<footer>
  <span>Iter <b id="s-iter">0</b></span>
  <span class="sep">|</span>
  <span>Games <b id="s-games">0</b></span>
  <span class="sep">|</span>
  <span>Elo <b id="s-elo">1000</b></span>
  <span class="sep">|</span>
  <span>Best <b id="s-best-elo">--</b> @ iter <b id="s-best-iter">--</b></span>
  <span class="sep">|</span>
  <span>Win P0:<b id="s-w0">0</b>% P1:<b id="s-w1">0</b>%</span>
  <span class="sep">|</span>
  <span class="res-bar">CPU <div class="res-meter"><div class="res-meter-fill" id="cpu-fill" style="width:0%"></div></div> <b id="s-cpu">0</b>%</span>
  <span class="res-bar">RAM <div class="res-meter"><div class="res-meter-fill" id="ram-fill" style="width:0%"></div></div> <b id="s-ram">0</b>%</span>
</footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<script>
// --- Train dashboard extra events ---
socket.on('iteration_start', d => {
  el('status').textContent = 'ITER ' + d.iteration + '/' + d.total;
  if (el('ls-iter')) el('ls-iter').textContent = d.iteration;
  gameStats = { w0: 0, w1: 0, totalLen: 0, count: 0 };
});
socket.on('iteration_complete', d => {
  try {
    if (d.iteration != null && el('ls-iter')) el('ls-iter').textContent = d.iteration;
    if (d.elo != null && el('s-elo')) el('s-elo').textContent = Math.round(d.elo);
    if (d.workers != null && el('ls-workers')) el('ls-workers').textContent = d.workers;
    if (el('progress-wrap')) el('progress-wrap').style.display = 'none';
    fetchCharts();
  } catch(e) { console.error('iteration_complete error:', e); }
});
socket.on('training_complete', () => {
  el('status').textContent = 'COMPLETE';
});

// --- Start / Stop training ---
function startTraining() {
  const gpi = el('cfg-games_per_iter'), ts = el('cfg-train_steps');
  const body = {};
  if (gpi && gpi.value) body.games_per_iter = +gpi.value;
  if (ts && ts.value) body.train_steps = +ts.value;
  fetch('/api/train/start', { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(r => r.json()).then(d => {
    el('status').textContent = d.status === 'started' ? 'STARTING...' : 'ALREADY RUNNING';
  });
}
function stopTraining() {
  fetch('/api/train/stop', { method: 'POST' }).then(() => {
    el('status').textContent = 'STOPPING...';
  });
}

// --- Training config ---
function cfgUpdate(key, value) {
  const body = {}; body[key] = value;
  fetch('/api/config', { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(r => r.json()).then(d => {
    const s = el('cfg-status');
    if (s) { s.textContent = 'Updated: ' + d.updated.join(', '); setTimeout(() => { s.textContent = ''; }, 3000); }
  });
}
function cfgSet(id, val) {
  const e = el(id);
  if (!e) return;
  if (e.type === 'checkbox') e.checked = !!val;
  else e.value = val;
}
function loadConfig() {
  fetch('/api/config').then(r => r.json()).then(d => {
    if (d.network) {
      ['board_size','num_channels','filters','res_blocks'].forEach(k => {
        const s = el('cfg-' + k); if (s && s.tagName === 'SPAN') s.textContent = d.network[k];
      });
    }
    if (d.search) {
      cfgSet('cfg-c_puct', d.search.c_puct); cfgSet('cfg-sims', d.search.sims);
      cfgSet('cfg-mcts_batch', d.search.mcts_batch); cfgSet('cfg-dirichlet_alpha', d.search.dirichlet_alpha);
      cfgSet('cfg-dirichlet_epsilon', d.search.dirichlet_epsilon); cfgSet('cfg-temp_threshold', d.search.temp_threshold);
    }
    if (d.training) {
      cfgSet('cfg-batch_size', d.training.batch_size); cfgSet('cfg-lr', d.training.lr);
      cfgSet('cfg-l2_reg', d.training.l2_reg); cfgSet('cfg-buffer_size', d.training.buffer_size);
    }
    if (d.pipeline) {
      cfgSet('cfg-train_steps', d.pipeline.train_steps); cfgSet('cfg-games_per_iter', d.pipeline.games_per_iter);
      cfgSet('cfg-checkpoint_every', d.pipeline.checkpoint_every); cfgSet('cfg-max_workers', d.pipeline.max_workers);
      cfgSet('cfg-games_per_future', d.pipeline.games_per_future);
    }
    if (d.elo) {
      cfgSet('cfg-eval_every', d.elo.eval_every); cfgSet('cfg-eval_games', d.elo.eval_games);
      cfgSet('cfg-eval_sims', d.elo.eval_sims); cfgSet('cfg-max_opponents', d.elo.max_opponents);
      cfgSet('cfg-baseline_games', d.elo.baseline_games); cfgSet('cfg-vault_max_models', d.elo.vault_max_models);
    }
    if (d.play_style) {
      cfgSet('cfg-play_style', d.play_style.style); cfgSet('cfg-c_blend_adjacent', d.play_style.c_blend_adjacent);
      cfgSet('cfg-c_blend_distant', d.play_style.c_blend_distant); cfgSet('cfg-distant_explore_prob', d.play_style.distant_explore_prob);
    }
    if (d.curriculum) {
      cfgSet('cfg-plateau_threshold', d.curriculum.plateau_threshold);
      cfgSet('cfg-plateau_iters', d.curriculum.plateau_iters); cfgSet('cfg-plateau_sim_boost', d.curriculum.plateau_sim_boost);
    }
    if (d.lr_schedule) {
      cfgSet('cfg-cosine_t0', d.lr_schedule.cosine_t0); cfgSet('cfg-cosine_t_mult', d.lr_schedule.cosine_t_mult);
      cfgSet('cfg-cosine_eta_min', d.lr_schedule.cosine_eta_min);
    }
    if (d.defensive) {
      cfgSet('cfg-blocking_priority_boost', d.defensive.blocking_priority_boost);
      cfgSet('cfg-survival_priority_boost', d.defensive.survival_priority_boost);
      cfgSet('cfg-use_ab_hybrid', d.defensive.use_ab_hybrid); cfgSet('cfg-ab_hybrid_depth', d.defensive.ab_hybrid_depth);
      cfgSet('cfg-threat_policy_blend', d.defensive.threat_policy_blend);
    }
    if (d.mixed_precision) {
      cfgSet('cfg-mixed_precision', d.mixed_precision.enabled); cfgSet('cfg-grad_clip_norm', d.mixed_precision.grad_clip_norm);
    }
  }).catch(() => {});
}

// --- Tab switching ---
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-content').forEach(tc => tc.style.display = 'none');
  const target = el('tab-' + name);
  if (target) target.style.display = name === 'charts' ? 'flex' : 'block';
  if (name === 'settings') loadConfig();
}

// --- Config section collapse ---
function toggleCfgSection(header) {
  const body = header.nextElementSibling;
  const tog = header.querySelector('.toggle');
  if (body) body.classList.toggle('collapsed');
  if (tog) tog.classList.toggle('collapsed');
}

setTimeout(loadConfig, 500);
</script>
<script>
// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
const DPR = window.devicePixelRatio || 1;
const socket = io();
socket.on('connect', () => {
  el('conn-dot').style.background = '#0a0'; el('conn-dot').title = 'Connected';
});
socket.on('disconnect', () => {
  el('conn-dot').style.background = '#c00'; el('conn-dot').title = 'Disconnected';
});

let stones0 = [], stones1 = [], moveOrder = [];  // moveOrder[i] = {q, r, num, player}

// KaTrain-style analysis overlay state
let analysisData = null;  // array of {value_estimate, top_moves, threat_count} per move
let overlayMode = { value: false, threat: false };

function el(id) { return document.getElementById(id); }
function setInfo(t) { el('game-info').textContent = t; }

function toggleOverlay(mode) {
  overlayMode[mode] = !overlayMode[mode];
  const btn = el('btn-' + (mode === 'value' ? 'val' : 'threat'));
  if (btn) btn.classList.toggle('active', overlayMode[mode]);
  // Show/hide value chart when value overlay is active
  const vcw = el('value-chart-wrap');
  if (vcw) vcw.style.display = (overlayMode.value && analysisData) ? '' : 'none';
  drawHex();
  if (overlayMode.value && analysisData) drawValueChart();
}

function drawValueChart() {
  if (!analysisData || !analysisData.length) return;
  const cv = el('value-chart');
  if (!cv) return;
  const { w: W, h: H } = sizeCanvas(cv);
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const n = Math.min(analysisData.length, replayMoveIdx || analysisData.length);
  if (n < 1) return;

  const pad = { l: 30, r: 8, t: 4, b: 4 };
  const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b;
  if (pW < 10 || pH < 10) return;

  // Zero line
  const zeroY = pad.t + pH / 2;
  ctx.strokeStyle = '#ccc'; ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(pad.l, zeroY); ctx.lineTo(pad.l + pW, zeroY); ctx.stroke();

  // Y axis labels
  ctx.fillStyle = '#999'; ctx.font = '8px Courier New'; ctx.textAlign = 'right';
  ctx.fillText('+1', pad.l - 3, pad.t + 8);
  ctx.fillText('-1', pad.l - 3, pad.t + pH);
  ctx.fillText('0', pad.l - 3, zeroY + 3);

  // Line chart of value estimates
  ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad.l + (i / Math.max(analysisData.length - 1, 1)) * pW;
    const v = analysisData[i].value_estimate || 0;
    const y = pad.t + pH / 2 - (v * pH / 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Current move marker
  if (replayMoveIdx > 0 && replayMoveIdx <= analysisData.length) {
    const i = replayMoveIdx - 1;
    const x = pad.l + (i / Math.max(analysisData.length - 1, 1)) * pW;
    const v = analysisData[i].value_estimate || 0;
    const y = pad.t + pH / 2 - (v * pH / 2);
    ctx.fillStyle = '#000';
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
  }
}

// Compute move quality: compare value[i] to value[i-1]
// Returns 'good', 'neutral', 'blunder', or null
function getMoveQuality(moveIdx) {
  if (!analysisData || moveIdx < 1 || moveIdx > analysisData.length) return null;
  const cur = analysisData[moveIdx - 1];
  if (moveIdx === 1) return 'neutral';
  const prev = analysisData[moveIdx - 2];
  // Value is from current player's perspective - a drop means a blunder
  // But players alternate, so we need to account for sign flip
  const delta = cur.value_estimate - prev.value_estimate;
  // Determine which player made this move
  const moveEntry = moveOrder.find(m => m.num === moveIdx);
  if (!moveEntry) return 'neutral';
  // For P0, positive value is good. For P1, negative value is good.
  // A move is a blunder if value shifted AGAINST the player who just moved.
  const playerSign = moveEntry.player === 0 ? 1 : -1;
  const effectiveDelta = delta * playerSign;
  if (effectiveDelta < -0.2) return 'blunder';
  if (effectiveDelta > 0.05) return 'good';
  return 'neutral';
}

// ---------------------------------------------------------------------------
// HiDPI canvas helper
// ---------------------------------------------------------------------------
function sizeCanvas(cv) {
  const r = cv.parentElement.getBoundingClientRect();
  const w = Math.round(r.width), h = Math.round(r.height);
  if (cv.width !== w * DPR || cv.height !== h * DPR) {
    cv.width = w * DPR; cv.height = h * DPR;
    cv.style.width = w + 'px'; cv.style.height = h + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  return { w, h };
}

// ---------------------------------------------------------------------------
// Collapsible charts
// ---------------------------------------------------------------------------
const chartState = {};
function toggleChart(name) {
  const body = el('body-' + name);
  const tog = el('tog-' + name);
  const collapsed = !body.classList.contains('collapsed');
  if (collapsed) {
    body.classList.add('collapsed');
    tog.classList.add('collapsed');
  } else {
    body.classList.remove('collapsed');
    tog.classList.remove('collapsed');
    // Redraw after expand
    setTimeout(fetchCharts, 50);
  }
  chartState[name] = collapsed;
}

// ---------------------------------------------------------------------------
// Hex canvas rendering
// ---------------------------------------------------------------------------
const HEX_SIZE = 18;
const S3 = Math.sqrt(3);

// Settings (persisted to localStorage)
const defaultSettings = { replaySpeed: 120, dotSize: 2, emptyHexRadius: 2, autoRefresh: true, showMoveNums: false };
let settings = Object.assign({}, defaultSettings);
try { const s = JSON.parse(localStorage.getItem('hexdash_settings')); if (s) Object.assign(settings, s); } catch(e) {}
function saveSetting(key, val) { settings[key] = val; localStorage.setItem('hexdash_settings', JSON.stringify(settings)); }
function axToPixel(q, r) {
  return [HEX_SIZE * (S3 * q + S3 / 2 * r), HEX_SIZE * (1.5 * r)];
}

function drawHex() {
  const cv = el('hex-canvas');
  const { w: W, h: H } = sizeCanvas(cv);
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const all = [...stones0, ...stones1];
  if (!all.length) {
    ctx.fillStyle = '#aaa';
    ctx.font = '12px Courier New';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for training game...', W / 2, H / 2);
    return;
  }

  // Compute bounding box
  let mnX = 1e9, mxX = -1e9, mnY = 1e9, mxY = -1e9;
  for (const [q, r] of all) {
    const [px, py] = axToPixel(q, r);
    if (px < mnX) mnX = px; if (px > mxX) mxX = px;
    if (py < mnY) mnY = py; if (py > mxY) mxY = py;
  }
  const mg = HEX_SIZE * 3;
  const spanX = mxX - mnX + mg * 2, spanY = mxY - mnY + mg * 2;
  const sc = Math.min(W / spanX, H / spanY, 2.5);
  const ox = W / 2 - (mnX + mxX) / 2 * sc, oy = H / 2 - (mnY + mxY) / 2 * sc;

  function toS(q, r) {
    const [px, py] = axToPixel(q, r);
    return [px * sc + ox, py * sc + oy];
  }
  function hexPath(cx, cy, sz) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = Math.PI / 3 * i - Math.PI / 6;
      const hx = cx + sz * Math.cos(a), hy = cy + sz * Math.sin(a);
      i === 0 ? ctx.moveTo(hx, hy) : ctx.lineTo(hx, hy);
    }
    ctx.closePath();
  }
  const hr = HEX_SIZE * sc * 0.88;

  // Empty hex dots around placed stones
  const stoneSet = new Set(all.map(s => s[0] + ',' + s[1]));
  const emptySet = new Set();
  const dotR = settings.emptyHexRadius;
  for (const [q, r] of all) {
    for (let dq = -dotR; dq <= dotR; dq++) {
      for (let dr = -dotR; dr <= dotR; dr++) {
        if (Math.abs(dq) + Math.abs(dr) > dotR + 1) continue;
        const key = (q + dq) + ',' + (r + dr);
        if (!stoneSet.has(key) && !emptySet.has(key)) {
          emptySet.add(key);
          const [sx, sy] = toS(q + dq, r + dr);
          ctx.fillStyle = '#ccc';
          ctx.beginPath();
          ctx.arc(sx, sy, Math.max(1.5, settings.dotSize * sc), 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  // Build quality lookup for overlay
  const qualityMap = {};
  if (overlayMode.value && analysisData) {
    for (const mv of moveOrder) {
      qualityMap[mv.q + ',' + mv.r] = getMoveQuality(mv.num);
    }
  }

  // Build threat set for overlay (cells with 4+ threats at current move)
  const threatCells = new Set();
  if (overlayMode.threat && analysisData && replayMoveIdx > 0 && replayMoveIdx <= analysisData.length) {
    const ad = analysisData[replayMoveIdx - 1];
    if (ad && ad.top_moves) {
      for (const tm of ad.top_moves) {
        threatCells.add(tm[0] + ',' + tm[1]);
      }
    }
    // Mark empty cells near threats
    if (ad && ad.threat_count >= 4) {
      // Highlight the stone itself
      const mv = moveOrder.find(m => m.num === replayMoveIdx);
      if (mv) threatCells.add(mv.q + ',' + mv.r);
    }
  }

  // P0: solid black hexagons
  for (const [q, r] of stones0) {
    const [sx, sy] = toS(q, r);
    const key = q + ',' + r;
    hexPath(sx, sy, hr);
    // Quality overlay: tint the fill color
    const qual = qualityMap[key];
    if (qual === 'blunder') {
      ctx.fillStyle = '#c00'; ctx.fill();
    } else if (qual === 'good') {
      ctx.fillStyle = '#070'; ctx.fill();
    } else {
      ctx.fillStyle = '#000'; ctx.fill();
    }
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();
    // Threat overlay ring
    if (overlayMode.threat && threatCells.has(key)) {
      hexPath(sx, sy, hr + 3 * sc);
      ctx.strokeStyle = '#f80'; ctx.lineWidth = 2.5; ctx.stroke();
    }
  }

  // P1: white hexagons with hatching
  for (const [q, r] of stones1) {
    const [sx, sy] = toS(q, r);
    const key = q + ',' + r;
    hexPath(sx, sy, hr);
    // Quality overlay: tint the fill color
    const qual = qualityMap[key];
    if (qual === 'blunder') {
      ctx.fillStyle = '#fcc'; ctx.fill();
    } else if (qual === 'good') {
      ctx.fillStyle = '#cfc'; ctx.fill();
    } else {
      ctx.fillStyle = '#fff'; ctx.fill();
    }
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.2; ctx.stroke();
    // Hatching (skip if quality-colored for clarity)
    if (!qual || qual === 'neutral') {
      ctx.save();
      hexPath(sx, sy, hr); ctx.clip();
      ctx.strokeStyle = '#000'; ctx.lineWidth = 0.6;
      const step = Math.max(3, 4 * sc);
      for (let d = -hr * 2; d <= hr * 2; d += step) {
        ctx.beginPath();
        ctx.moveTo(sx + d - hr, sy - hr);
        ctx.lineTo(sx + d + hr, sy + hr);
        ctx.stroke();
      }
      ctx.restore();
    }
    // Threat overlay ring
    if (overlayMode.threat && threatCells.has(key)) {
      hexPath(sx, sy, hr + 3 * sc);
      ctx.strokeStyle = '#f80'; ctx.lineWidth = 2.5; ctx.stroke();
    }
  }

  // Threat overlay: show top MCTS candidate moves on empty cells
  if (overlayMode.threat && analysisData && replayMoveIdx > 0 && replayMoveIdx <= analysisData.length) {
    const ad = analysisData[replayMoveIdx - 1];
    if (ad && ad.top_moves) {
      for (const tm of ad.top_moves) {
        const tmKey = tm[0] + ',' + tm[1];
        if (!stoneSet.has(tmKey)) {
          const [sx, sy] = toS(tm[0], tm[1]);
          const sz = Math.max(3, hr * 0.4 * Math.sqrt(tm[2]));
          ctx.fillStyle = 'rgba(255,136,0,0.5)';
          ctx.beginPath(); ctx.arc(sx, sy, sz, 0, Math.PI * 2); ctx.fill();
          // Show probability
          ctx.fillStyle = '#f80'; ctx.font = Math.max(6, hr * 0.35) + 'px Courier New';
          ctx.textAlign = 'center'; ctx.textBaseline = 'top';
          ctx.fillText((tm[2] * 100).toFixed(0) + '%', sx, sy + sz + 1);
        }
      }
    }
  }

  // Move numbers on stones
  if (settings.showMoveNums && moveOrder.length) {
    const fontSize = Math.max(7, Math.min(12, hr * 0.7));
    ctx.font = 'bold ' + fontSize + 'px Courier New';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const mv of moveOrder) {
      const [sx, sy] = toS(mv.q, mv.r);
      ctx.fillStyle = mv.player === 0 ? '#fff' : '#000';
      ctx.fillText(mv.num, sx, sy);
    }
  }

  // Update analysis label
  if (analysisData && replayMoveIdx > 0 && replayMoveIdx <= analysisData.length) {
    const ad = analysisData[replayMoveIdx - 1];
    const valStr = ad.value_estimate != null ? 'V:' + ad.value_estimate.toFixed(2) : '';
    const thrStr = ad.threat_count != null ? ' T:' + ad.threat_count : '';
    el('analysis-label').textContent = valStr + thrStr;
  } else {
    el('analysis-label').textContent = '';
  }
}

// ---------------------------------------------------------------------------
// Game replay
// ---------------------------------------------------------------------------
let replayTimer = null, replayBusy = false, pendingGame = null;
let gameStats = { w0: 0, w1: 0, totalLen: 0, count: 0 };

// Shared replay state for keyboard stepping
let replayPaused = false, currentGameData = null, replayMoveIdx = 0;

// Rebuild board state from moves up to index n
function rebuildBoard(moves, n) {
  stones0 = []; stones1 = []; moveOrder = [];
  let p = 0, stt = 0;
  for (let i = 0; i < n; i++) {
    const m = moves[i];
    if (p === 0) stones0.push(m); else stones1.push(m);
    moveOrder.push({ q: m[0], r: m[1], num: i + 1, player: p });
    stt++;
    const need = (i === 0) ? 1 : 2;
    if (stt >= need) { p = 1 - p; stt = 0; }
  }
}

function replayAdvance() {
  if (!currentGameData) return;
  const d = currentGameData, moves = d.moves;
  if (replayMoveIdx >= moves.length) {
    // Game finished
    if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
    const winner = d.result > 0 ? 'P0 (black)' : 'P1 (white)';
    setInfo('Game #' + d.game_idx + ': ' + winner + ' wins in ' + moves.length + ' moves');
    setTimeout(() => {
      replayBusy = false;
      if (!replayPaused && pendingGame) { const g = pendingGame; pendingGame = null; replayGame(g); }
    }, 800);
    return;
  }
  replayMoveIdx++;
  rebuildBoard(moves, replayMoveIdx);
  drawHex();
  if (overlayMode.value && analysisData) drawValueChart();
  setInfo('Game #' + d.game_idx + ' \u2014 Move ' + replayMoveIdx + '/' + moves.length +
    (replayPaused ? ' [PAUSED]' : ''));
}

function replayGame(d) {
  replayBusy = true;
  replayPaused = false;
  currentGameData = d;
  replayMoveIdx = 0;
  stones0 = []; stones1 = []; moveOrder = [];
  // Set analysis data (may be null if not available)
  analysisData = d._analysis || d.analysis || null;
  const vcw = el('value-chart-wrap');
  if (vcw) vcw.style.display = (overlayMode.value && analysisData) ? '' : 'none';
  drawHex();
  el('game-num').textContent = d.game_idx;
  setInfo('Game #' + d.game_idx + ' playing... (' + d.moves.length + ' moves)');
  if (replayTimer) clearInterval(replayTimer);
  replayTimer = setInterval(replayAdvance, settings.replaySpeed);
}

// ---------------------------------------------------------------------------
// Keyboard controls
// ---------------------------------------------------------------------------
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') {
    e.preventDefault();
    if (replayPaused) {
      // Resume auto-advance
      replayPaused = false;
      if (currentGameData && replayMoveIdx < currentGameData.moves.length) {
        if (replayTimer) clearInterval(replayTimer);
        replayTimer = setInterval(replayAdvance, settings.replaySpeed);
      }
      setInfo(el('game-info').textContent.replace(' [PAUSED]', ''));
    } else {
      // Pause
      replayPaused = true;
      if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
      setInfo(el('game-info').textContent + ' [PAUSED]');
    }
  }
  if (e.code === 'ArrowRight' && currentGameData) {
    e.preventDefault();
    // Pause auto-advance and step forward
    replayPaused = true;
    if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
    if (replayMoveIdx < currentGameData.moves.length) {
      replayMoveIdx++;
      rebuildBoard(currentGameData.moves, replayMoveIdx);
      drawHex();
      if (overlayMode.value && analysisData) drawValueChart();
      setInfo('Game #' + currentGameData.game_idx + ' \u2014 Move ' + replayMoveIdx + '/' + currentGameData.moves.length + ' [PAUSED]');
    }
  }
  if (e.code === 'ArrowLeft' && currentGameData) {
    e.preventDefault();
    // Pause auto-advance and step backward
    replayPaused = true;
    if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
    if (replayMoveIdx > 0) {
      replayMoveIdx--;
      rebuildBoard(currentGameData.moves, replayMoveIdx);
      drawHex();
      if (overlayMode.value && analysisData) drawValueChart();
      setInfo('Game #' + currentGameData.game_idx + ' \u2014 Move ' + replayMoveIdx + '/' + currentGameData.moves.length + ' [PAUSED]');
    }
  }
  if (e.code === 'KeyR' && currentGameData) {
    replayGame(currentGameData);
  }
  if (e.code === 'KeyV') {
    toggleOverlay('value');
  }
  if (e.code === 'KeyT') {
    toggleOverlay('threat');
  }
});

// ---------------------------------------------------------------------------
// Game history
// ---------------------------------------------------------------------------
const gameHistoryList = [];  // store last 20 games

function addToHistory(d) {
  gameHistoryList.push(d);
  if (gameHistoryList.length > 10) gameHistoryList.shift();
  const histEl = el('game-history');
  if (!histEl) return;
  histEl.innerHTML = '';
  gameHistoryList.forEach((g, i) => {
    const span = document.createElement('span');
    span.className = 'gh-item' + (g === currentGameData ? ' active' : '');
    const w = g.result > 0 ? 'B' : 'W';
    span.textContent = '#' + g.game_idx + ' ' + w + ' ' + g.num_moves + 'mv';
    span.onclick = () => { replayPaused = false; replayGame(g); };
    histEl.appendChild(span);
  });
  histEl.scrollTop = histEl.scrollHeight;
}

// ---------------------------------------------------------------------------
// Socket events
// ---------------------------------------------------------------------------
socket.on('game_complete', d => {
  try {
    el('status').textContent = 'GAME ' + d.game_idx;
    // Progress bar
    if (d.total_games) {
      const pw = el('progress-wrap');
      if (pw) pw.style.display = '';
      const pf = el('progress-fill');
      if (pf) pf.style.width = Math.round(d.game_idx / d.total_games * 100) + '%';
      const plab = el('progress-label');
      if (plab) plab.textContent = 'Self-play ' + d.game_idx + '/' + d.total_games;
    }
    // Stats
    gameStats.count++;
    gameStats.totalLen += (d.num_moves || 0);
    if (d.result > 0) gameStats.w0++; else if (d.result < 0) gameStats.w1++;
    const lsG = el('ls-games'); if (lsG) lsG.textContent = gameStats.count;
    const lsL = el('ls-len'); if (lsL && gameStats.count) lsL.textContent = Math.round(gameStats.totalLen / gameStats.count);
    const lsW0 = el('ls-w0'); if (lsW0 && gameStats.count) lsW0.textContent = Math.round(gameStats.w0 / gameStats.count * 100) + '%';
    const lsW1 = el('ls-w1'); if (lsW1 && gameStats.count) lsW1.textContent = Math.round(gameStats.w1 / gameStats.count * 100) + '%';
    // History + replay (NEVER interrupt a playing game)
    if (d.moves && d.moves.length) {
      // Store analysis data if present
      d._analysis = d.analysis || null;
      addToHistory(d);
      if (replayBusy) {
        pendingGame = d;  // just queue the latest, current game plays to end
      } else {
        replayGame(d);
      }
    }
  } catch (e) { console.error('game_complete error:', e); }
});

socket.on('stats_update', d => {
  try {
    if (d.iteration != null) {
      el('s-iter').textContent = d.iteration;
      el('ls-iter').textContent = d.iteration;
    }
    if (d.total_games != null) el('s-games').textContent = d.total_games;
    if (d.current_elo != null) el('s-elo').textContent = Math.round(d.current_elo);
    if (d.best_elo != null) el('s-best-elo').textContent = Math.round(d.best_elo);
    if (d.best_iteration != null) el('s-best-iter').textContent = d.best_iteration;
    if (d.workers != null) el('ls-workers').textContent = d.workers;
    if (d.win_p0 != null) el('s-w0').textContent = Math.round(d.win_p0);
    if (d.win_p1 != null) el('s-w1').textContent = Math.round(d.win_p1);
    if (d.latest_loss != null) {
      const lv = typeof d.latest_loss === 'object' ? d.latest_loss.total : d.latest_loss;
      if (lv != null) el('ls-loss').textContent = parseFloat(lv).toFixed(4);
    }
    if (d.avg_game_length) el('ls-len').textContent = d.avg_game_length;
    fetchCharts();
  } catch (e) { console.error('stats_update error:', e); }
});

socket.on('train_progress', d => {
  try {
    const pw = el('progress-wrap');
    const pf = el('progress-fill');
    const plab = el('progress-label');
    if (pw) pw.style.display = '';
    if (pf) pf.style.width = d.pct + '%';
    const phase = d.phase || 'Training';
    const lossStr = d.loss != null ? ' (loss ' + d.loss + ')' : '';
    if (plab) plab.textContent = phase + ' ' + d.step + '/' + d.total + lossStr;
    if (d.loss != null) el('ls-loss').textContent = d.loss;
    el('status').textContent = phase.toUpperCase() + ' ' + d.step + '/' + d.total;
  } catch (e) { console.error('train_progress error:', e); }
});

// ---------------------------------------------------------------------------
// Line chart (HiDPI, auto-scaling)
// ---------------------------------------------------------------------------
function drawLineChart(cv, datasets, opts) {
  if (!cv) return;
  // Skip collapsed charts
  const body = cv.closest('.chart-body');
  if (body && body.classList.contains('collapsed')) return;

  const { w: W, h: H } = sizeCanvas(cv);
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const pad = { t: 14, r: 16, b: 24, l: 52 };
  const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b;
  if (pW < 10 || pH < 10) return;

  const pts = datasets.flatMap(d => d.data);
  if (pts.length < 2) {
    ctx.fillStyle = '#ccc'; ctx.font = '11px Courier New'; ctx.textAlign = 'center';
    ctx.fillText('No data yet', W / 2, H / 2);
    return;
  }

  let xMn = Math.min(...pts.map(p => p.x)), xMx = Math.max(...pts.map(p => p.x));
  let yMn = opts.yMin != null ? opts.yMin : Math.min(...pts.map(p => p.y));
  let yMx = opts.yMax != null ? opts.yMax : Math.max(...pts.map(p => p.y));
  if (xMn === xMx) xMx = xMn + 1;
  if (opts.yMin == null) { const yP = (yMx - yMn) * 0.1 || 1; yMn -= yP; yMx += yP; }

  function toP(x, y) {
    return [pad.l + (x - xMn) / (xMx - xMn) * pW, pad.t + pH - (y - yMn) / (yMx - yMn) * pH];
  }

  // Axes
  ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + pH); ctx.lineTo(pad.l + pW, pad.t + pH);
  ctx.stroke();

  // Y ticks
  ctx.fillStyle = '#000'; ctx.font = '10px Courier New'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const yV = yMn + (yMx - yMn) * i / 4;
    const [, sy] = toP(xMn, yV);
    ctx.fillText(yV.toFixed(1), pad.l - 4, sy + 3);
    ctx.strokeStyle = '#eee'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, sy); ctx.lineTo(pad.l + pW, sy); ctx.stroke();
  }

  // X ticks
  ctx.textAlign = 'center';
  const xStep = Math.max(1, Math.ceil((xMx - xMn) / 6));
  for (let x = Math.ceil(xMn); x <= xMx; x += xStep) {
    const [sx] = toP(x, yMn);
    ctx.fillStyle = '#000'; ctx.fillText(x, sx, pad.t + pH + 14);
  }

  // Lines
  const defaultDash = [[], [6, 3], [2, 2], [8, 2, 2, 2]];
  datasets.forEach((ds, di) => {
    if (ds.data.length < 2) return;
    ctx.setLineDash(ds.dash || defaultDash[di % defaultDash.length] || []);
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.3;
    ctx.beginPath();
    ds.data.forEach((p, i) => {
      const [sx, sy] = toP(p.x, p.y);
      i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Legend
  ctx.font = '10px Courier New'; ctx.textAlign = 'left';
  datasets.forEach((ds, di) => {
    const lx = pad.l + 8 + di * 90, ly = pad.t + 10;
    ctx.setLineDash(ds.dash || defaultDash[di % defaultDash.length] || []);
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#000'; ctx.fillText(ds.label, lx + 22, ly + 3);
  });
}

// ---------------------------------------------------------------------------
// Fetch chart data from API
// ---------------------------------------------------------------------------
function fetchCharts() {
  fetch('/api/elo').then(r => r.json()).then(data => {
    const arr = Array.isArray(data) ? data : (data.elo_history || []);
    if (!arr.length) return;
    drawLineChart(el('elo-chart'),
      [{ label: 'ELO', data: arr.map(d => ({ x: d.iteration, y: d.elo })), dash: [] }], {});
  }).catch(() => {});

  fetch('/api/losses').then(r => r.json()).then(data => {
    if (!data.length) return;
    const ds = [{ label: 'total', data: data.filter(d => d.total != null).map(d => ({ x: d.iteration, y: d.total })), dash: [] }];
    if (data.some(d => d.value != null))
      ds.push({ label: 'value', data: data.filter(d => d.value != null).map(d => ({ x: d.iteration, y: d.value })), dash: [6, 3] });
    if (data.some(d => d.policy != null))
      ds.push({ label: 'policy', data: data.filter(d => d.policy != null).map(d => ({ x: d.iteration, y: d.policy })), dash: [2, 2] });
    drawLineChart(el('loss-chart'), ds, {});
  }).catch(() => {});

  fetch('/api/winrates').then(r => r.json()).then(data => {
    if (!data.length) return;
    drawLineChart(el('winrate-chart'), [
      { label: 'P0%', data: data.map(d => ({ x: d.iteration, y: d.p0 })), dash: [] },
      { label: 'P1%', data: data.map(d => ({ x: d.iteration, y: d.p1 })), dash: [6, 3] },
      { label: 'Draw%', data: data.map(d => ({ x: d.iteration, y: d.draw })), dash: [2, 2] },
    ], { yMin: 0, yMax: 100 });
  }).catch(() => {});

  fetch('/api/gamelength').then(r => r.json()).then(data => {
    if (!data.length) return;
    drawLineChart(el('gamelength-chart'),
      [{ label: 'Avg Moves', data: data.map(d => ({ x: d.iteration, y: d.avg_game_length })), dash: [] }], {});
  }).catch(() => {});

  fetch('/api/speed').then(r => r.json()).then(data => {
    if (!data.length) return;
    drawLineChart(el('speed-chart'),
      [{ label: 'Games/s', data: data.map(d => ({ x: d.iteration, y: d.games_per_sec })), dash: [] }], {});
  }).catch(() => {});

  fetch('/api/resources').then(r => r.json()).then(data => {
    const h = data.history || [];
    if (h.length < 2) return;
    drawLineChart(el('res-chart'), [
      { label: 'CPU%', data: h.map((d, i) => ({ x: i, y: d.cpu_pct })), dash: [] },
      { label: 'RAM%', data: h.map((d, i) => ({ x: i, y: d.ram_pct })), dash: [6, 3] },
    ], { yMin: 0, yMax: 100 });
  }).catch(() => {});
}

// ---------------------------------------------------------------------------
// Resource polling
// ---------------------------------------------------------------------------
setInterval(() => {
  fetch('/api/stats').then(r => r.json()).then(d => {
    if (d.cpu_pct != null) {
      el('s-cpu').textContent = Math.round(d.cpu_pct);
      el('cpu-fill').style.width = Math.round(d.cpu_pct) + '%';
    }
    if (d.ram_pct != null) {
      el('s-ram').textContent = Math.round(d.ram_pct);
      el('ram-fill').style.width = Math.round(d.ram_pct) + '%';
    }
  }).catch(() => {});
}, 5000);

// ---------------------------------------------------------------------------
// Resize handler
// ---------------------------------------------------------------------------
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { drawHex(); fetchCharts(); }, 150);
});

// ---------------------------------------------------------------------------
// Init - restore settings from localStorage
// ---------------------------------------------------------------------------
(function initSettings() {
  el('set-speed').value = settings.replaySpeed;
  el('set-speed-val').textContent = settings.replaySpeed + 'ms';
  el('set-dotsize').value = settings.dotSize;
  el('set-dotsize-val').textContent = settings.dotSize;
  el('set-radius').value = settings.emptyHexRadius;
  el('set-radius-val').textContent = settings.emptyHexRadius;
  el('set-movenums').checked = settings.showMoveNums;
  el('set-autorefresh').checked = settings.autoRefresh;
})();
setTimeout(() => { drawHex(); fetchCharts(); }, 100);
fetch('/api/stats').then(r => r.json()).then(s => {
  el('s-games').textContent = s.total_games;
  el('s-elo').textContent = Math.round(s.current_elo);
}).catch(() => {});

// --- Train dashboard extra events ---
socket.on('iteration_start', d => {
  el('status').textContent = 'ITER ' + d.iteration + '/' + d.total;
  if (el('ls-iter')) el('ls-iter').textContent = d.iteration;
  gameStats = { w0: 0, w1: 0, totalLen: 0, count: 0 };
});
socket.on('iteration_complete', d => {
  try {
    if (d.iteration != null && el('ls-iter')) el('ls-iter').textContent = d.iteration;
    if (d.elo != null && el('s-elo')) el('s-elo').textContent = Math.round(d.elo);
    if (d.workers != null && el('ls-workers')) el('ls-workers').textContent = d.workers;
    if (el('progress-wrap')) el('progress-wrap').style.display = 'none';
    fetchCharts();
  } catch(e) { console.error('iteration_complete error:', e); }
});
socket.on('training_complete', () => {
  el('status').textContent = 'COMPLETE';
});

// Update lights-out status with live info
socket.on('iteration_complete', d => {
  const lo = el('lights-out-status');
  if (lo && d.iteration != null) {
    lo.textContent = 'ITER ' + d.iteration + (d.elo ? ' | ELO ' + Math.round(d.elo) : '');
  }
});

// --- Dark mode ---
function toggleDarkMode(on) {
  document.body.classList.toggle('dark', on);
  saveSetting('darkMode', on);
  // Re-render charts and hex since filter inverts colors
  try { drawHex(); fetchCharts(); } catch(e) {}
}

// --- Lights out: black screen with minimal status ---
function lightsOut() {
  // Temporarily remove dark mode filter so overlay is pure black
  document.body.classList.remove('dark');
  const ov = el('lights-out-overlay');
  if (ov) ov.classList.add('active');
}
function lightsOff() {
  const ov = el('lights-out-overlay');
  if (ov) ov.classList.remove('active');
  // Restore dark mode if it was on
  if (settings.darkMode) document.body.classList.add('dark');
}

// Restore dark mode on load
if (settings.darkMode) {
  document.body.classList.add('dark');
  const cb = el('set-darkmode');
  if (cb) cb.checked = true;
}

</script>
<div id="lights-out-overlay">
  <span id="lights-out-status">TRAINING IN PROGRESS</span>
  <button id="lights-out-btn" onclick="lightsOff()">LIGHTS ON</button>
</div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    port = 5001
    print('HEX BOT Dashboard')
    print(f'Device: {get_device()}')
    print(f'CPU cores: {multiprocessing.cpu_count()}')
    print(f'Workers: {resource_monitor.num_threads}')
    print(f'Curriculum: 20->50->100->200 sims, games scale down as sims increase')
    print(f'LR: 0.01 -> decay 0.5 every 20 iters')
    print(f'Open http://localhost:{port}')
    # Suppress Flask/Werkzeug HTTP request spam
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True, log_output=False)
