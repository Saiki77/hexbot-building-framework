"""
HEX BOT - Training Dashboard

Black-and-white minimalist dashboard with live game visualization,
ELO progression, loss curves, and auto-scaling parallel training.

Usage: python dashboard.py
Then open http://localhost:5001
"""

from __future__ import annotations
import sys
import math
import multiprocessing
import random
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from flask import Flask, jsonify, request, Response
from flask_socketio import SocketIO

from main import HexGame
from bot import (
    HexNet, MCTS, ReplayBuffer, self_play_game, train_step,
    get_device, BOARD_SIZE, NUM_SIMULATIONS, BATCH_SIZE, LEARNING_RATE, L2_REG,
    TrainingSample, OnnxPredictor, export_onnx,
    POSITION_CATALOG, generate_puzzles, augment_sample,
    load_human_games, load_online_games, find_forced_move,
    encode_state,
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
    def __init__(self, metrics: MetricsStore, observer: DashboardObserver,
                 sio: SocketIO, resource_monitor: ResourceMonitor):
        self.metrics = metrics
        self.observer = observer
        self.sio = sio
        self.resource = resource_monitor
        self.device = get_device()
        self.net: Optional[HexNet] = None
        self.model_vault = ModelVault(max_models=200)
        self.arena = GenerationalArena(self.device)
        self._thread: Optional[threading.Thread] = None

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

    @staticmethod
    def _find_latest_checkpoint() -> Optional[str]:
        """Find the latest hex_checkpoint_N.pt file."""
        import glob
        ckpts = glob.glob('hex_checkpoint_*.pt')
        if not ckpts:
            return None
        # Sort by iteration number
        def get_iter(path):
            try:
                return int(path.replace('hex_checkpoint_', '').replace('.pt', ''))
            except ValueError:
                return -1
        ckpts.sort(key=get_iter)
        return ckpts[-1]

    def _run(self, num_iterations, games_per_iter, train_steps):
        # Use create_network factory - 'large' for maximum strength
        try:
            from bot import create_network
            self.net = create_network('standard').to(self.device)
            self._net_config = 'standard'
        except ImportError:
            self.net = HexNet(num_filters=256, num_res_blocks=12).to(self.device)
            self._net_config = 'standard'
        param_count = sum(p.numel() for p in self.net.parameters())

        print(f'\n{"="*60}')
        print(f'  TRAINING PIPELINE INITIALIZED')
        print(f'{"="*60}')
        print(f'  Network:     {self._net_config} ({param_count:,} params)')
        print(f'  Device:      {self.device}')
        print(f'  Iterations:  ∞ (runs until stopped)')
        print(f'  Games/iter:  {games_per_iter} (base, scaled by curriculum)')
        print(f'  Train steps: {train_steps}/iter')

        # Detect C engine availability
        use_v2 = False
        try:
            from bot import CGameState
            CGameState()
            use_v2 = True
            print(f'  Engine:      C engine (v2) OK:')
        except Exception as e:
            print(f'  Engine:      Python (v1) - C engine unavailable: {e}')

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.001, weight_decay=L2_REG
        )
        # Cosine annealing: cycles LR between 0.001 and 0.0001
        # T_0=50 means first cycle is 50 iterations, then doubles each cycle
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-4
        )
        replay_buffer = ReplayBuffer()
        # Cap workers for power-limited training (90W charger)
        num_workers = min(self.resource.num_threads, 5)
        auto_tuner = AutoTuner()
        self._auto_tuner = auto_tuner

        # --- Resume from checkpoint if available ---
        start_iteration = 0
        resume_path = self._find_latest_checkpoint()
        if resume_path:
            try:
                ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
                # Migrate channels (5->7) and filters (128->256) if needed
                from bot import migrate_checkpoint_5to7, migrate_checkpoint_filters
                old_shape = ckpt['model_state_dict'].get('conv_init.weight', None)
                migrated_sd = migrate_checkpoint_5to7(ckpt['model_state_dict'])
                migrated_sd = migrate_checkpoint_filters(migrated_sd)
                new_shape = migrated_sd.get('conv_init.weight', None)
                arch_changed = (old_shape is not None and new_shape is not None
                                and old_shape.shape != new_shape.shape)
                ckpt['model_state_dict'] = migrated_sd
                self.net.load_state_dict(migrated_sd, strict=False)
                # Skip optimizer restore if architecture changed (Adam momentum buffers wrong shape)
                if arch_changed:
                    print(f'    WARN: Fresh optimizer (conv_init migrated {old_shape.shape}->{new_shape.shape})')
                else:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_iteration = ckpt.get('iteration', 0) + 1

                # CRITICAL: Force LR back to a healthy value
                # The old training decayed LR to ~7e-6 (frozen). Reset it.
                for pg in optimizer.param_groups:
                    if pg['lr'] < 1e-4:
                        pg['lr'] = 0.001
                        print(f'    WARN: LR was frozen at {pg["lr"]:.2e}, reset to 0.001')

                # Restore scheduler state if available
                if 'scheduler_state_dict' in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    except Exception:
                        pass  # scheduler format changed, start fresh

                # Restore replay buffer from separate file
                import pickle
                buf_path = os.path.join(os.path.dirname(resume_path), 'replay_buffer.pkl')
                if os.path.exists(buf_path):
                    try:
                        with open(buf_path, 'rb') as f:
                            buf_data = pickle.load(f)
                        replay_buffer.buffer.extend(buf_data.get('buffer', []))
                        replay_buffer.priorities.extend(buf_data.get('priorities', []))
                        print(f'    OK: Restored replay buffer: {len(replay_buffer.buffer)} samples')
                    except Exception as e:
                        print(f'    WARN: Could not restore buffer: {e}')

                # Restore metrics
                if 'metrics' in ckpt:
                    m = ckpt['metrics']
                    self.metrics.iterations = m.get('iterations', [])
                    self.metrics.elo_history = m.get('elo_history', [])
                    self.metrics.current_elo = m.get('current_elo', 1000)
                    self.metrics.total_games = m.get('total_games', 0)
                    self.metrics.current_iteration = start_iteration
                # Restore AutoTuner (but NOT the LR - that's managed by scheduler now)
                if 'auto_tuner' in ckpt:
                    at = ckpt['auto_tuner']
                    auto_tuner.params = at.get('params', auto_tuner.params)
                    auto_tuner.params['lr'] = optimizer.param_groups[0]['lr']  # sync with optimizer
                    # Force pure self-play (override old mix from checkpoint)
                    auto_tuner.params['mix_normal'] = 1.0
                    auto_tuner.params['mix_catalog'] = 0.0
                    auto_tuner.params['mix_endgame'] = 0.0
                    auto_tuner.params['mix_formation'] = 0.0
                    auto_tuner.params['mix_sequence'] = 0.0
                    auto_tuner.loss_history = at.get('loss_history', [])
                    auto_tuner.elo_history = at.get('elo_history', [])
                current_lr = optimizer.param_groups[0]['lr']
                print(f'  OK: Resumed from {resume_path} (iter {start_iteration})')
                print(f'    ELO: {self.metrics.current_elo:.0f}, '
                      f'Games: {self.metrics.total_games}, '
                      f'LR: {current_lr:.5f}')
            except Exception as e:
                print(f'  WARN: Failed to resume: {e} - starting fresh')
                start_iteration = 0
        else:
            print(f'  No checkpoint found - starting fresh')

        print(f'  Workers:     {num_workers}')
        print(f'  LR:          {auto_tuner.params["lr"]} -> auto-tuned')
        print(f'  AutoTuner:   OK: (self-improving)')
        print(f'{"="*60}\n')

        for iteration in range(start_iteration, num_iterations):
            if self.observer.should_stop():
                print(f'\n  STOP: Training stopped by user at iteration {iteration}')
                break
            self.observer.on_iteration_start(iteration, num_iterations)

            prev_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            # --- Curriculum (AutoTuner can override sims) ---
            current_sims = max(auto_tuner.params['sims'],
                               get_curriculum_sims(iteration))
            current_games = get_curriculum_games(iteration, games_per_iter)
            current_lr = optimizer.param_groups[0]['lr']

            hours = (time.perf_counter() - _curriculum_start_time) / 3600 if _curriculum_start_time else 0
            print(f'  ┌─ Iter {iteration+1}/{num_iterations} '
                  f'│ sims={current_sims} │ games={current_games} │ lr={current_lr:.4f} │ {hours:.1f}h')

            # --- Load human games on first iteration ---
            if iteration == 0:
                human_path = os.path.join(os.path.dirname(__file__), 'human_games.jsonl')
                if os.path.exists(human_path):
                    t_human = time.perf_counter()
                    human_samples = load_human_games(human_path, max_games=500, min_elo=1000)
                    for s in human_samples:
                        s.priority = 0.8  # lower than self-play to avoid dominating
                        replay_buffer.push(s)
                    t_human = time.perf_counter() - t_human
                    print(f'  │  Human games: {len(human_samples)} samples loaded ({t_human:.1f}s)')
                else:
                    print(f'  │  Human games: not found (run scrape_games.py first)')

            # --- Export ONNX for fast CPU workers ---
            t0 = time.perf_counter()
            t_export_start = time.perf_counter()
            self.net.eval()
            onnx_path = f'/tmp/hex_model_{iteration}.onnx'
            export_onnx(self.net, onnx_path)
            self.net.to(self.device)  # move back to GPU after export
            t_export = time.perf_counter() - t_export_start
            print(f'  │  ONNX export: {t_export:.1f}s')

            # --- Adaptive game mix: 5 categories (AutoTuner-driven) ---
            import random as _rng
            ap = auto_tuner.params
            mix = (ap['mix_normal'], ap['mix_catalog'], ap['mix_endgame'],
                   ap['mix_formation'], ap['mix_sequence'])
            n_normal = int(current_games * mix[0])
            n_catalog = int(current_games * mix[1])
            n_endgame = int(current_games * mix[2])
            n_formation = int(current_games * mix[3])
            n_sequence = current_games - n_normal - n_catalog - n_endgame - n_formation
            print(f'  │  Game mix: {n_normal} normal + {n_catalog} catalog + '
                  f'{n_endgame} L1-endgame + {n_formation} L2-formation + {n_sequence} L3-sequence')

            # Generate position lists for workers
            catalog_keys = list(POSITION_CATALOG.keys())
            catalog_positions = [
                (POSITION_CATALOG[_rng.choice(catalog_keys)], None)
                for _ in range(n_catalog)
            ]

            # Guided positions by level
            from bot import GUIDED_POSITIONS, get_guided_positions_by_level
            l1 = get_guided_positions_by_level(1)
            l2 = get_guided_positions_by_level(2)
            l3 = get_guided_positions_by_level(3)

            endgame_positions = [_rng.choice(l1) if l1 else (None, None) for _ in range(n_endgame)]
            formation_positions = [_rng.choice(l2) if l2 else (None, None) for _ in range(n_formation)]
            sequence_positions = [_rng.choice(l3) if l3 else (None, None) for _ in range(n_sequence)]

            # Also generate some random puzzles as fallback
            puzzle_positions = generate_puzzles(n_endgame // 2)
            extra_puzzles = [(p, None) for p in puzzle_positions]

            # All positions: (position_dict_or_None, hint_moves_or_None)
            all_positions = (
                catalog_positions +
                endgame_positions +
                formation_positions +
                sequence_positions +
                extra_puzzles[:n_endgame // 2] +
                [(None, None)] * n_normal
            )
            # Trim to exact game count
            all_positions = all_positions[:current_games]
            while len(all_positions) < current_games:
                all_positions.append((None, None))
            _rng.shuffle(all_positions)

            # Distribute across workers with positions
            chunks = []
            chunk_positions = []
            remaining = current_games
            pos_offset = 0
            for i in range(num_workers):
                c = remaining // (num_workers - i)
                chunks.append(c)
                chunk_positions.append(all_positions[pos_offset:pos_offset + c])
                pos_offset += c
                remaining -= c

            active_chunks = [c for c in chunks if c > 0]
            print(f'  │  Dispatching {len(active_chunks)} workers: '
                  f'{active_chunks}')

            total_samples = 0
            total_moves = 0
            wins = [0, 0, 0]
            game_idx = 0
            collected_samples = []  # for augmentation
            worker_errors = 0

            t_selfplay_start = time.perf_counter()

            if use_v2 and self.device.type == 'cuda':
                # CUDA threaded: shared GPU network, no process spawn overhead
                from concurrent.futures import ThreadPoolExecutor
                from bot import self_play_game_v2 as _spv2, BatchedMCTS as _BMCTS
                import io as _io, contextlib as _cl

                print(f'  |  Self-play: CUDA threaded ({num_workers} threads, '
                      f'{current_sims} sims)...')
                self.net.eval()
                _mcts = _BMCTS(self.net, num_simulations=current_sims, batch_size=64)
                _lock = threading.Lock()

                def _play_game(gi):
                    pos = flat_pos[gi] if gi < len(flat_pos) else None
                    with _cl.redirect_stdout(_io.StringIO()):
                        samples, moves, *rest = _spv2(self.net, _mcts, start_position=pos)
                    ana = rest[0] if rest else None
                    rv = samples[0].result if samples else 0.0
                    return samples, moves, rv, ana

                flat_pos = []
                for c, pos in zip(chunks, chunk_positions):
                    for gi in range(c):
                        flat_pos.append(pos[gi] if pos and gi < len(pos) else None)

                with ThreadPoolExecutor(max_workers=num_workers) as pool:
                    futs = {pool.submit(_play_game, i): i for i in range(current_games)}
                    for future in as_completed(futs):
                        if self.observer.should_stop():
                            break
                        try:
                            samples, move_hist, result_val, ana_data = future.result()
                        except Exception as e:
                            worker_errors += 1
                            print(f'  |  Thread error: {e}')
                            continue
                        for s in samples:
                            replay_buffer.push(s)
                            collected_samples.append(s)
                        total_samples += len(samples)
                        total_moves += len(move_hist)
                        if result_val > 0: wins[0] += 1
                        elif result_val < 0: wins[1] += 1
                        else: wins[2] += 1
                        game_idx += 1
                        self.observer.on_game_complete(
                            game_idx, current_games,
                            [list(m) for m in move_hist],
                            result_val, len(samples))

            elif use_v2:
                # MPS/CPU: ProcessPool with C engine + PyTorch
                net_state = {k: v.cpu() for k, v in self.net.state_dict().items()}

                # Asymmetric self-play: 25% of workers use an older checkpoint as opponent
                n_asymmetric = 0
                if len(self.model_vault) > 1:
                    n_asymmetric = max(1, num_workers // 4)

                print(f'  │  Self-play: V2 (C engine + MCTS {current_sims} sims'
                      f'{f", {n_asymmetric} asymmetric" if n_asymmetric else ""})...')

                with ProcessPoolExecutor(max_workers=num_workers) as pool:
                    futures = []
                    # 2 games per future: first results after ~1 min, good streaming
                    GAMES_PER_FUTURE = 2
                    flat_positions = []
                    for c, pos in zip(chunks, chunk_positions):
                        for gi in range(c):
                            p = pos[gi] if pos and gi < len(pos) else None
                            flat_positions.append(p)
                    for i in range(0, len(flat_positions), GAMES_PER_FUTURE):
                        batch_pos = flat_positions[i:i+GAMES_PER_FUTURE]
                        # Wrap in list if positions are provided
                        pos_arg = [p for p in batch_pos] if any(batch_pos) else None
                        futures.append(
                            pool.submit(_self_play_worker_v2, net_state,
                                        self._net_config, current_sims,
                                        len(batch_pos), pos_arg,
                                        use_alphabeta=False)
                        )
                    print(f'  │  Dispatching {len(futures)} futures ({GAMES_PER_FUTURE} games each, {num_workers} workers)')
                    for future in as_completed(futures):
                        if self.observer.should_stop():
                            break
                        try:
                            batch_results = future.result()
                        except Exception as e:
                            worker_errors += 1
                            print(f'  │  WARN: V2 Worker error: {e}')
                            import traceback
                            traceback.print_exc()
                            continue

                        for serialized, move_hist, result_val, n_samples in batch_results:
                            for sd in serialized:
                                sample = TrainingSample(
                                    encoded_state=torch.from_numpy(sd['state']),
                                    policy_target=sd['policy'],
                                    player=sd['player'],
                                    result=sd['result'],
                                    threat_label=sd.get('threat'),
                                    priority=sd.get('priority', 1.0),
                                )
                                replay_buffer.push(sample)
                                collected_samples.append(sample)
                            total_samples += n_samples
                            total_moves += len(move_hist)
                            if result_val > 0:
                                wins[0] += 1
                            elif result_val < 0:
                                wins[1] += 1
                            else:
                                wins[2] += 1
                            game_idx += 1
                            from datetime import datetime
                            winner = 'P0' if result_val > 0 else ('P1' if result_val < 0 else 'draw')
                            print(f'  │  [{datetime.now().strftime("%H:%M:%S")}] Game {game_idx}/{current_games}: {winner} in {len(move_hist)} moves ({n_samples} samples)')
                            self.observer.on_game_complete(
                                game_idx, current_games, move_hist,
                                result_val, n_samples,
                            )
            else:
                # V1 fallback: ONNX workers
                print(f'  │  Self-play: V1 (ONNX Runtime)...')
                with ProcessPoolExecutor(max_workers=num_workers) as pool:
                    futures = [
                        pool.submit(_self_play_worker, onnx_path, current_sims, c, pos)
                        for c, pos in zip(chunks, chunk_positions) if c > 0
                    ]
                    for future in as_completed(futures):
                        if self.observer.should_stop():
                            break
                        try:
                            batch_results = future.result()
                        except Exception as e:
                            worker_errors += 1
                            print(f'  │  WARN: Worker error: {e}')
                            import traceback
                            traceback.print_exc()
                            continue

                        for serialized, move_hist, result_val, n_samples in batch_results:
                            for sd in serialized:
                                sample = TrainingSample(
                                    encoded_state=torch.from_numpy(sd['state']),
                                    policy_target=sd['policy'],
                                    player=sd['player'],
                                    result=sd['result'],
                                    threat_label=sd.get('threat'),
                                    priority=sd.get('priority', 1.0),
                                )
                                replay_buffer.push(sample)
                                collected_samples.append(sample)
                            total_samples += n_samples
                            total_moves += len(move_hist)
                            if result_val > 0:
                                wins[0] += 1
                            elif result_val < 0:
                                wins[1] += 1
                            else:
                                wins[2] += 1
                            game_idx += 1
                            from datetime import datetime
                            winner = 'P0' if result_val > 0 else ('P1' if result_val < 0 else 'draw')
                            print(f'  │  [{datetime.now().strftime("%H:%M:%S")}] Game {game_idx}/{current_games}: {winner} in {len(move_hist)} moves ({n_samples} samples)')
                            self.observer.on_game_complete(
                                game_idx, current_games, move_hist,
                                result_val, n_samples,
                            )

            t_selfplay = time.perf_counter() - t_selfplay_start
            games_per_sec = game_idx / t_selfplay if t_selfplay > 0 else 0
            print(f'  │  Self-play done: {game_idx} games, {total_samples} samples, '
                  f'{t_selfplay:.1f}s ({games_per_sec:.1f} games/s)')
            print(f'  │  Wins: P0={wins[0]} P1={wins[1]} draw={wins[2]}')
            if worker_errors > 0:
                print(f'  │  WARN: {worker_errors} worker(s) failed')

            # --- Load new online games (human feedback) ---
            online_path = os.path.join(os.path.dirname(__file__), 'online_games.jsonl')
            if not hasattr(self, '_online_lines_read'):
                self._online_lines_read = 0
            try:
                online_samples, new_pos = load_online_games(
                    online_path, start_line=self._online_lines_read)
                if online_samples:
                    for s in online_samples:
                        replay_buffer.push(s)
                        collected_samples.append(s)  # include in augmentation too
                    self._online_lines_read = new_pos
                    print(f'  │  Online games: +{len(online_samples)} samples (priority 2.0)')
            except Exception as e:
                pass  # online games are optional

            # --- Symmetry augmentation: 3x data multiplier (hex-on-grid valid) ---
            t_aug_start = time.perf_counter()
            aug_count = 0
            for sample in collected_samples:
                for aug in augment_sample(sample):
                    replay_buffer.push(aug)
                    aug_count += 1
            total_samples += aug_count
            t_aug = time.perf_counter() - t_aug_start
            print(f'  │  Augmentation: +{aug_count} samples ({t_aug:.1f}s) '
                  f'-> buffer={len(replay_buffer)}/{replay_buffer.buffer.maxlen}')

            sp_time = time.perf_counter() - t0

            # --- GPU training (aggressive) ---
            losses = {'total': 0, 'value': 0, 'policy': 0}
            train_time = 0
            if len(replay_buffer) >= BATCH_SIZE:
                print(f'  │  Training: {train_steps} steps on {self.device} '
                      f'(batch={BATCH_SIZE}, buffer={len(replay_buffer)})...')
                t1 = time.perf_counter()
                self.net.train()
                for step in range(train_steps):
                    losses = train_step(
                        self.net, optimizer, replay_buffer, self.device
                    )
                    # Emit progress every 50 steps (less overhead)
                    if step % 50 == 0 or step == train_steps - 1:
                        self.sio.emit('train_progress', {
                            'step': step + 1,
                            'total': train_steps,
                            'loss': round(losses['total'], 4),
                            'pct': round((step + 1) / train_steps * 100),
                        })
                train_time = time.perf_counter() - t1
                steps_per_sec = train_steps / train_time if train_time > 0 else 0
                scheduler.step()
                # Sync LR from scheduler to AutoTuner (for display)
                current_lr = optimizer.param_groups[0]['lr']
                auto_tuner.params['lr'] = current_lr
                print(f'  │  Training done: {train_time:.1f}s ({steps_per_sec:.0f} steps/s) '
                      f'loss={losses["total"]:.4f} (v={losses["value"]:.4f} p={losses["policy"]:.4f}) '
                      f'lr={current_lr:.6f}')
            else:
                print(f'  │  Training skipped: buffer too small ({len(replay_buffer)}<{BATCH_SIZE})')

            # ELO eval (every 2 iterations) - generational tournament
            elo_str = ''
            if len(replay_buffer) >= BATCH_SIZE and (iteration + 1) % 2 == 0:
                # Store current generation in vault
                self.model_vault.add(iteration + 1, self.net.state_dict())
                n_opp = min(self.arena.max_opponents, len(self.model_vault) - 1)
                n_games = n_opp * self.arena.games_per_opponent
                print(f'  │  ELO evaluation ({n_games} games vs {n_opp} generations, '
                      f'vault={len(self.model_vault)})...')
                t_elo = time.perf_counter()
                new_elo = self.arena.evaluate(
                    self.net, self.model_vault, self.metrics.current_elo
                )
                t_elo = time.perf_counter() - t_elo
                delta = new_elo - self.metrics.current_elo
                sign = '+' if delta >= 0 else ''
                self.metrics.update_elo(iteration + 1, new_elo)
                update_curriculum_plateau(new_elo)
                elo_str = f' │ ELO {new_elo:.0f} ({sign}{delta:.0f})'
                stall_info = f' stall={_curriculum_stall_iters}' if _curriculum_stall_iters > 0 else ''
                print(f'  │  ELO: {new_elo:.0f} ({sign}{delta:.0f}) in {t_elo:.1f}s{stall_info}')

            res_snap = self.resource.snapshot()
            avg_len = total_moves / max(game_idx, 1)
            total_time = time.perf_counter() - t0
            metrics = {
                'iteration': iteration + 1,
                'games': game_idx,
                'samples': total_samples,
                'wins': wins,
                'self_play_time': round(sp_time, 1),
                'train_time': round(train_time, 1),
                'loss': losses,
                'buffer_size': len(replay_buffer),
                'avg_game_length': round(avg_len, 1),
                'elo': round(self.metrics.current_elo, 1),
                'workers': num_workers,
                'cpu_pct': res_snap['cpu_pct'],
                'ram_pct': res_snap['ram_pct'],
                'sims': current_sims,
                'lr': round(optimizer.param_groups[0]['lr'], 5),
            }
            self.observer.on_iteration_complete(metrics)

            # --- AutoTuner: observe and adapt ---
            gps = game_idx / t_selfplay if t_selfplay > 0 else 1.0
            at_metrics = {
                'total_loss': metrics.get('loss', {}).get('total', 0),
                'value_loss': metrics.get('loss', {}).get('value', 0),
                'policy_loss': metrics.get('loss', {}).get('policy', 0),
                'elo': self.metrics.current_elo,
                'games_per_sec': gps,
                'buffer_fill': len(replay_buffer) / replay_buffer.buffer.maxlen,
            }
            new_params = auto_tuner.update(at_metrics, iteration)
            # Apply LR change
            for pg in optimizer.param_groups:
                pg['lr'] = new_params['lr']

            print(f'  └─ Iter {iteration+1} done: {total_time:.1f}s total '
                  f'│ {game_idx} games │ {total_samples} samples '
                  f'│ CPU {res_snap["cpu_pct"]:.0f}% │ RAM {res_snap["ram_pct"]:.0f}%'
                  f'{elo_str}')
            print()

            if (iteration + 1) % 5 == 0:
                ckpt_path = f'hex_checkpoint_{iteration+1}.pt'
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': {
                        'iterations': self.metrics.iterations,
                        'elo_history': self.metrics.elo_history,
                        'current_elo': self.metrics.current_elo,
                        'total_games': self.metrics.total_games,
                    },
                    'auto_tuner': {
                        'params': auto_tuner.params,
                        'loss_history': auto_tuner.loss_history,
                        'elo_history': auto_tuner.elo_history,
                    },
                }, ckpt_path)
                # Save replay buffer separately (too large for .pt)
                import pickle
                try:
                    buf_path = 'replay_buffer.pkl'
                    with open(buf_path, 'wb') as f:
                        pickle.dump({
                            'buffer': list(replay_buffer.buffer),
                            'priorities': list(replay_buffer.priorities),
                        }, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f'  WARN: Buffer save failed: {e}')
                print(f'   Checkpoint saved: {ckpt_path} + buffer ({len(replay_buffer.buffer)} samples)')

        print(f'\n{"="*60}')
        print(f'  TRAINING COMPLETE - {num_iterations} iterations')
        print(f'  Final ELO: {self.metrics.current_elo:.0f}')
        print(f'  Total games: {self.metrics.total_games}')
        print(f'{"="*60}\n')
        self.observer.on_training_complete()


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
        'vault_size': len(training_mgr.model_vault) if training_mgr else 0,
        'matchups': training_mgr.arena.get_matchup_summary() if training_mgr else {},
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
    if training_mgr and hasattr(training_mgr, '_auto_tuner'):
        return jsonify(training_mgr._auto_tuner.get_status())
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
    """Get current training configuration."""
    from bot import (BATCH_SIZE, LEARNING_RATE, L2_REG, NUM_SIMULATIONS,
                     NUM_FILTERS, NUM_RES_BLOCKS, PLAY_STYLE,
                     C_BLEND_ADJACENT, C_BLEND_DISTANT, DIRICHLET_ALPHA,
                     TEMP_THRESHOLD, REPLAY_BUFFER_SIZE)
    return jsonify({
        'network': {
            'filters': NUM_FILTERS, 'res_blocks': NUM_RES_BLOCKS,
        },
        'training': {
            'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE,
            'l2_reg': L2_REG, 'buffer_size': REPLAY_BUFFER_SIZE,
        },
        'search': {
            'sims': NUM_SIMULATIONS, 'dirichlet_alpha': DIRICHLET_ALPHA,
            'temp_threshold': TEMP_THRESHOLD,
        },
        'play_style': {
            'style': PLAY_STYLE,
            'c_blend_adjacent': C_BLEND_ADJACENT,
            'c_blend_distant': C_BLEND_DISTANT,
        },
    })


@app.route('/api/config', methods=['POST'])
def api_config_update():
    """Update training config (applies to next iteration)."""
    import bot
    data = request.json or {}
    updated = []
    # Training params
    if 'batch_size' in data:
        bot.BATCH_SIZE = int(data['batch_size']); updated.append('batch_size')
    if 'lr' in data:
        bot.LEARNING_RATE = float(data['lr']); updated.append('lr')
    if 'sims' in data:
        bot.NUM_SIMULATIONS = int(data['sims']); updated.append('sims')
    if 'dirichlet_alpha' in data:
        bot.DIRICHLET_ALPHA = float(data['dirichlet_alpha']); updated.append('dirichlet_alpha')
    if 'temp_threshold' in data:
        bot.TEMP_THRESHOLD = int(data['temp_threshold']); updated.append('temp_threshold')
    if 'play_style' in data:
        bot.PLAY_STYLE = data['play_style']; updated.append('play_style')
    if 'c_blend_adjacent' in data:
        bot.C_BLEND_ADJACENT = float(data['c_blend_adjacent']); updated.append('c_blend_adjacent')
    if 'c_blend_distant' in data:
        bot.C_BLEND_DISTANT = float(data['c_blend_distant']); updated.append('c_blend_distant')
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
.right{width:50%;display:flex;flex-direction:column;overflow-y:auto}
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
.settings-btn{cursor:pointer;font-size:18px;margin-left:12px;user-select:none}
.settings-panel{position:absolute;top:42px;right:20px;background:#fff;border:1px solid #000;
  padding:16px;z-index:100;font:11px 'SF Mono','Courier New',monospace;min-width:260px;box-shadow:2px 2px 0 #0001}
.settings-row{display:flex;align-items:center;gap:8px;margin-bottom:10px}
.settings-row label{width:110px;font-weight:700;flex-shrink:0}
.settings-row input[type=range]{flex:1;accent-color:#000}
.settings-row span.val{width:50px;text-align:right;font-size:10px}
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
</style>
</head>
<body>
<header>
  <span class="title">Hex Bot Training</span>
  <button id="btn-start" onclick="startTraining()" style="background:#000;color:#fff;border:none;padding:4px 10px;font:13px 'Courier New';cursor:pointer">&#9654;</button><button id="btn-stop" onclick="stopTraining()" style="background:#fff;color:#000;border:1px solid #000;padding:4px 10px;font:13px 'Courier New';cursor:pointer">&#9632;</button><span class="status" id="status">IDLE</span>
  <span class="conn-dot" id="conn-dot" style="background:#ccc" title="Disconnected"></span>
  <span class="settings-btn" onclick="toggleSettings()">&#9881;</span>
</header>
<div id="settings-panel" class="settings-panel" style="display:none">
  <div style="font-weight:700;margin-bottom:12px;letter-spacing:2px;font-size:10px;text-transform:uppercase">Settings</div>
  <div class="settings-row">
    <label>Replay speed</label>
    <input type="range" id="set-speed" min="50" max="500" value="120" step="10"
      oninput="saveSetting('replaySpeed',+this.value);el('set-speed-val').textContent=this.value+'ms'">
    <span class="val" id="set-speed-val">120ms</span>
  </div>
  <div class="settings-row">
    <label>Dot size</label>
    <input type="range" id="set-dotsize" min="1" max="5" value="2" step="0.5"
      oninput="saveSetting('dotSize',+this.value);el('set-dotsize-val').textContent=this.value;drawHex()">
    <span class="val" id="set-dotsize-val">2</span>
  </div>
  <div class="settings-row">
    <label>Grid radius</label>
    <input type="range" id="set-radius" min="1" max="4" value="2" step="1"
      oninput="saveSetting('emptyHexRadius',+this.value);el('set-radius-val').textContent=this.value;drawHex()">
    <span class="val" id="set-radius-val">2</span>
  </div>
  <div class="settings-row">
    <label>Move numbers</label>
    <input type="checkbox" id="set-movenums" checked
      onchange="saveSetting('showMoveNums',this.checked);drawHex()">
  </div>
  <div class="settings-row">
    <label>Auto-refresh</label>
    <input type="checkbox" id="set-autorefresh" checked
      onchange="saveSetting('autoRefresh',this.checked)">
  </div>
</div>
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

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js">
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
  fetch('/api/train/start', { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({})
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
function updateConfig(key, value) {
  const body = {}; body[key] = value;
  fetch('/api/config', { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(r => r.json()).then(d => {
    const s = el('cfg-status');
    if (s) s.textContent = 'Updated: ' + d.updated.join(', ');
    setTimeout(() => { if (s) s.textContent = ''; }, 3000);
  });
}
function loadConfig() {
  fetch('/api/config').then(r => r.json()).then(d => {
    if (d.search) {
      if (el('cfg-sims')) el('cfg-sims').value = d.search.sims;
      if (el('cfg-dirichlet')) el('cfg-dirichlet').value = d.search.dirichlet_alpha;
      if (el('cfg-temp')) el('cfg-temp').value = d.search.temp_threshold;
    }
    if (d.training) {
      if (el('cfg-batch')) el('cfg-batch').value = d.training.batch_size;
      if (el('cfg-lr')) el('cfg-lr').value = d.training.lr;
    }
    if (d.play_style) {
      if (el('cfg-playstyle')) el('cfg-playstyle').value = d.play_style.style;
    }
  }).catch(() => {});
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
function toggleSettings() {
  const p = el('settings-panel');
  p.style.display = p.style.display === 'none' ? '' : 'none';
}

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

// --- Start / Stop training ---
function startTraining() {
  fetch('/api/train/start', { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({})
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
function updateConfig(key, value) {
  const body = {}; body[key] = value;
  fetch('/api/config', { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(r => r.json()).then(d => {
    const s = el('cfg-status');
    if (s) s.textContent = 'Updated: ' + d.updated.join(', ');
    setTimeout(() => { if (s) s.textContent = ''; }, 3000);
  });
}
function loadConfig() {
  fetch('/api/config').then(r => r.json()).then(d => {
    if (d.search) {
      if (el('cfg-sims')) el('cfg-sims').value = d.search.sims;
      if (el('cfg-dirichlet')) el('cfg-dirichlet').value = d.search.dirichlet_alpha;
      if (el('cfg-temp')) el('cfg-temp').value = d.search.temp_threshold;
    }
    if (d.training) {
      if (el('cfg-batch')) el('cfg-batch').value = d.training.batch_size;
      if (el('cfg-lr')) el('cfg-lr').value = d.training.lr;
    }
    if (d.play_style) {
      if (el('cfg-playstyle')) el('cfg-playstyle').value = d.play_style.style;
    }
  }).catch(() => {});
}
setTimeout(loadConfig, 500);
</script>
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
