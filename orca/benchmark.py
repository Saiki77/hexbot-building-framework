"""
Comprehensive benchmark for comparing performance across platforms.

Measures every stage of the training pipeline to identify bottlenecks
on different hardware (Mac MPS, CUDA, CPU).

Usage:
    python -m orca.benchmark                    # full benchmark
    python -m orca.benchmark --quick            # fast subset (~30s)
    python -m orca.benchmark --section nn       # specific section
    python -m orca.benchmark --output bench.json # save results

Sections: engine, nn, search, selfplay, training, augmentation, all
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


def _fmt(n):
    if n >= 1_000_000: return f"{n/1_000_000:,.1f}M"
    if n >= 1_000: return f"{n/1_000:,.1f}K"
    return f"{n:,.1f}"


def _get_system_info() -> Dict:
    """Collect detailed system info for reproducibility."""
    import platform
    import psutil
    import torch

    info = {
        'platform': sys.platform,
        'python': platform.python_version(),
        'torch': torch.__version__,
    }

    # CPU
    try:
        if sys.platform == 'darwin':
            import subprocess
            chip = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                stderr=subprocess.DEVNULL).decode().strip()
            info['cpu'] = chip
        elif sys.platform == 'linux':
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        info['cpu'] = line.split(':')[1].strip()
                        break
        else:
            info['cpu'] = platform.processor() or 'Unknown'
    except Exception:
        info['cpu'] = platform.processor() or 'Unknown'

    info['cpu_cores'] = psutil.cpu_count(logical=False) or 0
    info['cpu_threads'] = psutil.cpu_count(logical=True) or 0
    info['ram_gb'] = round(psutil.virtual_memory().total / 1e9, 1)

    # GPU
    from bot import get_device
    device = get_device()
    info['device'] = str(device)
    if device.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        info['vram_gb'] = round(vram / 1e9, 1)
    elif device.type == 'mps':
        info['gpu'] = 'Apple MPS'
        # Try to get Apple chip name
        try:
            import subprocess
            chip = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                stderr=subprocess.DEVNULL).decode().strip()
            info['gpu'] = f"Apple MPS ({chip.split('Apple ')[-1] if 'Apple' in chip else chip})"
        except Exception:
            pass
    else:
        info['gpu'] = 'CPU only'

    # Hexbot version
    try:
        from importlib.metadata import version
        info['hexbot_version'] = version('hexbot')
    except Exception:
        info['hexbot_version'] = 'dev'

    return info


def _load_checkpoint_net(device):
    """Load the actual Orca checkpoint for realistic benchmarks."""
    from orca.network import create_network
    from bot import get_device
    try:
        from orca import Orca
        bot = Orca.load(sims=1)
        net = bot._net.to(device)
        net.eval()
        return net, True
    except Exception:
        # Fallback to random weights
        net = create_network('standard').to(device)
        net.eval()
        return net, False


# ---------------------------------------------------------------------------
# 1. C Engine benchmarks
# ---------------------------------------------------------------------------

def bench_engine() -> Dict:
    """C engine: random games, scored moves, search, clone, undo, threats."""
    from hexgame import HexGame
    import random

    results = {}

    # Random games
    t0 = time.perf_counter()
    n_games = 500
    total_stones = 0
    for i in range(n_games):
        g = HexGame(max_stones=100)
        rng = random.Random(i)
        while not g.is_over:
            moves = g.legal_moves()
            g.place(*rng.choice(moves))
        total_stones += g.total_stones
    elapsed = time.perf_counter() - t0
    results['random_games_per_sec'] = n_games / elapsed
    results['random_stones_per_sec'] = total_stones / elapsed

    # Scored moves
    t0 = time.perf_counter()
    n_pos = 5000
    scored = 0
    for i in range(n_pos):
        g = HexGame()
        rng = random.Random(i)
        for _ in range(rng.randint(3, 10)):
            if g.is_over: break
            g.place(*rng.choice(g.legal_moves()))
        if not g.is_over:
            g.scored_moves(20)
            scored += 1
    elapsed = time.perf_counter() - t0
    results['scored_moves_per_sec'] = scored / elapsed

    # Alpha-beta search
    g = HexGame()
    g.place(0, 0); g.place(2, 0); g.place(2, -1); g.place(1, 0); g.place(0, 1)
    for depth in [4, 6, 8]:
        t0 = time.perf_counter()
        r = g.search(depth=depth)
        elapsed = time.perf_counter() - t0
        results[f'ab_depth{depth}_nodes_per_sec'] = r['nodes'] / elapsed if elapsed > 0 else 0
        results[f'ab_depth{depth}_time'] = elapsed

    # Clone
    t0 = time.perf_counter()
    n_clones = 50000
    for _ in range(n_clones):
        g.clone()
    elapsed = time.perf_counter() - t0
    results['clones_per_sec'] = n_clones / elapsed

    # Place + undo
    g2 = HexGame()
    g2.place(0, 0); g2.place(1, 0)
    moves = g2.legal_moves()[:10]
    t0 = time.perf_counter()
    n_ops = 50000
    for i in range(n_ops):
        g2.place(*moves[i % len(moves)])
        g2.undo()
    elapsed = time.perf_counter() - t0
    results['place_undo_per_sec'] = n_ops / elapsed

    # Threat detection
    g3 = HexGame()
    for q, r in [(0,0), (5,0), (5,-1), (1,0), (2,0), (6,0), (6,-1), (3,0), (4,0)]:
        g3.place(q, r)
    t0 = time.perf_counter()
    n_checks = 20000
    for _ in range(n_checks):
        g3.has_winning_move(0)
        g3.has_winning_move(1)
        g3.count_winning_moves(0)
    elapsed = time.perf_counter() - t0
    results['threat_checks_per_sec'] = (n_checks * 3) / elapsed

    return results


# ---------------------------------------------------------------------------
# 2. Neural network benchmarks
# ---------------------------------------------------------------------------

def bench_nn(configs=None) -> Dict:
    """NN inference: forward pass at various batch sizes and architectures."""
    import torch
    from orca.network import create_network
    from bot import get_device

    device = get_device()
    results = {'device': str(device)}

    if configs is None:
        configs = ['fast', 'standard', 'hex-masked', 'hex-native']

    for cfg in configs:
        net = create_network(cfg).to(device)
        net.eval()
        params = sum(p.numel() for p in net.parameters())
        results[f'{cfg}_params'] = params

        for bs in [1, 8, 32, 64, 128]:
            dummy = torch.randn(bs, 7, 19, 19).to(device)
            # Warmup
            for _ in range(3):
                with torch.no_grad(): net(dummy)
            if device.type == 'mps': torch.mps.synchronize()
            elif device.type == 'cuda': torch.cuda.synchronize()

            n_iters = max(10, 100 // bs)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                with torch.no_grad(): net(dummy)
            if device.type == 'mps': torch.mps.synchronize()
            elif device.type == 'cuda': torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            total = n_iters * bs
            results[f'{cfg}_bs{bs}_pos_per_sec'] = total / elapsed
            results[f'{cfg}_bs{bs}_latency_ms'] = elapsed / n_iters * 1000

        del net
        if device.type == 'cuda': torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# 3. NN Latency (single position, detailed)
# ---------------------------------------------------------------------------

def bench_latency() -> Dict:
    """Single-position NN latency stats."""
    import torch
    from orca.network import create_network
    from bot import get_device

    device = get_device()
    net = create_network('standard').to(device)
    net.eval()
    dummy = torch.randn(1, 7, 19, 19).to(device)

    for _ in range(10):
        with torch.no_grad(): net(dummy)

    lats = []
    for _ in range(100):
        if device.type == 'mps': torch.mps.synchronize()
        elif device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(): net(dummy)
        if device.type == 'mps': torch.mps.synchronize()
        elif device.type == 'cuda': torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)

    return {
        'mean_ms': float(np.mean(lats)),
        'p50_ms': float(np.percentile(lats, 50)),
        'p95_ms': float(np.percentile(lats, 95)),
        'p99_ms': float(np.percentile(lats, 99)),
        'min_ms': float(np.min(lats)),
        'max_ms': float(np.max(lats)),
    }


# ---------------------------------------------------------------------------
# 4. MCTS Search benchmarks
# ---------------------------------------------------------------------------

def bench_search() -> Dict:
    """MCTS search: sims/sec at different configurations (uses C MCTS when available)."""
    import torch
    from orca.encoding import CGameState
    from bot import get_device

    device = get_device()
    net, loaded = _load_checkpoint_net(device)
    results = {}

    # Use C MCTS if available
    try:
        from orca.c_mcts import CMCTSSearch
        mcts_cls = lambda sims, bs: CMCTSSearch(net, num_simulations=sims, batch_size=bs)
        results['engine'] = 'c_mcts'
    except Exception:
        from orca.search import BatchedMCTS
        mcts_cls = lambda sims, bs: BatchedMCTS(net, num_simulations=sims, batch_size=bs)
        results['engine'] = 'python'

    for sims in [50, 100, 200]:
        for bs in [8, 32, 64]:
            mcts = mcts_cls(sims, bs)
            game = CGameState(max_total_stones=200)
            game.place_stone(0, 0)
            game.place_stone(1, 0)
            game.place_stone(1, -1)

            t0 = time.perf_counter()
            n_searches = 3
            for _ in range(n_searches):
                mcts.search(game, temperature=0.1, add_noise=False)
            elapsed = time.perf_counter() - t0

            results[f'mcts_sims{sims}_bs{bs}_sec_per_search'] = elapsed / n_searches
            results[f'mcts_sims{sims}_bs{bs}_searches_per_sec'] = n_searches / elapsed

    return results


# ---------------------------------------------------------------------------
# 4b. Threaded Self-play (CUDA benefits most, MPS has GIL issues)
# ---------------------------------------------------------------------------

def _gpu_selfplay_worker(result_path, n, use_c, device_str):
    """Subprocess worker for threaded GPU self-play benchmark.
    Top-level function so it's picklable on macOS (spawn method).
    """
    try:
        import torch as _torch, json as _json
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import io, contextlib, time as _time

        with contextlib.redirect_stdout(io.StringIO()):
            from orca.data import self_play_game_v2
        from orca.network import create_network

        _dev = _torch.device(device_str)
        try:
            from orca import Orca
            bot = Orca.load(sims=1)
            net = bot._net.to(_dev)
            net.eval()
            loaded = True
        except Exception:
            net = create_network('standard').to(_dev)
            net.eval()
            loaded = False

        if use_c:
            from orca.c_mcts import CMCTSSearch
            mcts = CMCTSSearch(net, num_simulations=50, batch_size=64)
        else:
            from orca.search import BatchedMCTS
            mcts = BatchedMCTS(net, num_simulations=50, batch_size=64)

        times, lengths = [], []
        def _play(i):
            t0 = _time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                samples, moves, *_ = self_play_game_v2(net, mcts)
            return _time.perf_counter() - t0, len(moves)

        wall_start = _time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(4, n)) as pool:
            futs = [pool.submit(_play, i) for i in range(n)]
            for f in as_completed(futs):
                try:
                    t, l = f.result(timeout=120)
                    times.append(t)
                    lengths.append(l)
                except Exception:
                    pass
        wall_time = _time.perf_counter() - wall_start
        ng = len(times)
        total_moves = sum(lengths)
        result = {
            'games': ng, 'wall_time': wall_time,
            'avg_time_per_game': sum(times) / max(ng, 1),
            'games_per_hour': ng / wall_time * 3600 if wall_time > 0 else 0,
            'throughput_factor': (ng / wall_time) / (1 / (sum(times) / max(ng, 1))) if times else 0,
            'avg_game_length': sum(lengths) / max(ng, 1),
            'positions_per_sec': total_moves / wall_time if wall_time > 0 else 0,
            'checkpoint_loaded': loaded,
        }
        with open(result_path, 'w') as f:
            _json.dump(result, f)
    except Exception as e:
        import json as _json
        with open(result_path, 'w') as f:
            _json.dump({'skipped': True, 'reason': str(e)}, f)


def bench_gpu_selfplay(n_games=3) -> Dict:
    """Threaded self-play with shared GPU network. CUDA only.
    MPS segfaults with concurrent forward_pv (not thread-safe).
    Runs in a subprocess so crashes don't kill the benchmark.
    """
    import torch
    from bot import get_device

    device = get_device()
    if device.type != 'cuda':
        return {'skipped': True, 'reason': f'CUDA only (MPS not thread-safe for NN inference)'}

    use_c_mcts = False
    try:
        from orca.c_mcts import CMCTSSearch
        use_c_mcts = True
    except Exception:
        pass

    import multiprocessing as mp
    import tempfile

    result_file = tempfile.mktemp(suffix='.json')
    p = mp.Process(target=_gpu_selfplay_worker,
                   args=(result_file, n_games, use_c_mcts, str(device)))
    p.start()
    p.join(timeout=180)

    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        return {'skipped': True, 'reason': 'Timed out (180s)'}

    if p.exitcode != 0:
        return {'skipped': True, 'reason': f'Subprocess crashed (exit code {p.exitcode})'}

    try:
        import json as _json
        with open(result_file) as f:
            return _json.load(f)
    except Exception:
        return {'skipped': True, 'reason': 'No results from subprocess'}
    finally:
        try:
            os.unlink(result_file)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 4c. CPU Self-play (for Mac comparison: is MPS actually faster than CPU?)
# ---------------------------------------------------------------------------

def bench_cpu_selfplay(n_games=3) -> Dict:
    """Self-play forced to CPU for comparison against MPS/CUDA."""
    import torch
    from orca.network import create_network
    from orca.search import BatchedMCTS
    from bot import get_device

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        from orca.data import self_play_game_v2

    # Force CPU
    net, loaded = _load_checkpoint_net(torch.device('cpu'))
    net = net.to('cpu')
    mcts = BatchedMCTS(net, num_simulations=50, batch_size=32)

    times, lengths = [], []
    for _ in range(n_games):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            samples, moves, *_ = self_play_game_v2(net, mcts)
        times.append(time.perf_counter() - t0)
        lengths.append(len(moves))

    total = sum(times)
    total_moves = sum(lengths)
    return {
        'games': n_games,
        'total_time': total,
        'avg_time_per_game': total / n_games,
        'games_per_hour': n_games / total * 3600,
        'avg_game_length': sum(lengths) / n_games,
        'positions_per_sec': total_moves / total if total > 0 else 0,
        'checkpoint_loaded': loaded,
        'device': 'cpu',
    }


# ---------------------------------------------------------------------------
# 5. Self-play benchmarks
# ---------------------------------------------------------------------------

def bench_selfplay(n_games=3) -> Dict:
    """Full self-play pipeline: game generation speed with actual checkpoint."""
    import torch
    from bot import get_device

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        from orca.data import self_play_game_v2

    device = get_device()
    net, loaded = _load_checkpoint_net(device)

    try:
        from orca.c_mcts import CMCTSSearch
        mcts = CMCTSSearch(net, num_simulations=50, batch_size=50)
    except Exception:
        from orca.search import BatchedMCTS
        mcts = BatchedMCTS(net, num_simulations=50, batch_size=32)

    times, lengths, sample_counts = [], [], []
    for _ in range(n_games):
        t0 = time.perf_counter()
        samples, moves, *_ = self_play_game_v2(net, mcts)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        lengths.append(len(moves))
        sample_counts.append(len(samples))

    total = sum(times)
    total_moves = sum(lengths)
    return {
        'games': n_games,
        'total_time': total,
        'avg_time_per_game': total / n_games,
        'games_per_hour': n_games / total * 3600,
        'avg_game_length': sum(lengths) / n_games,
        'avg_samples_per_game': sum(sample_counts) / n_games,
        'positions_per_sec': total_moves / total if total > 0 else 0,
        'checkpoint_loaded': loaded,
    }


# ---------------------------------------------------------------------------
# 5b. Self-play scaling (multiple sim counts)
# ---------------------------------------------------------------------------

def bench_selfplay_scaling(n_games=3) -> Dict:
    """Self-play at multiple sim levels to show scaling behavior."""
    import torch
    from orca.encoding import CGameState
    from bot import get_device
    import io, contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        from orca.data import self_play_game_v2

    device = get_device()
    net, loaded = _load_checkpoint_net(device)
    results = {'checkpoint_loaded': loaded, 'levels': []}

    # Use C MCTS if available
    try:
        from orca.c_mcts import CMCTSSearch
        mcts_fn = lambda s: CMCTSSearch(net, num_simulations=s, batch_size=min(64, s))
    except Exception:
        from orca.search import BatchedMCTS
        mcts_fn = lambda s: BatchedMCTS(net, num_simulations=s, batch_size=64)

    for sims in [50, 100, 200, 400]:
        mcts = mcts_fn(sims)
        times, lengths = [], []
        for _ in range(n_games):
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                samples, moves, *_ = self_play_game_v2(net, mcts)
            times.append(time.perf_counter() - t0)
            lengths.append(len(moves))

        total_t = sum(times)
        total_m = sum(lengths)
        results['levels'].append({
            'sims': sims,
            'games': n_games,
            'total_time': total_t,
            'avg_sec_per_game': total_t / n_games,
            'games_per_hour': n_games / total_t * 3600,
            'avg_moves': total_m / n_games,
            'positions_per_sec': total_m / total_t if total_t > 0 else 0,
            'ms_per_move': total_t / max(total_m, 1) * 1000,
        })

    return results


# ---------------------------------------------------------------------------
# 5c. C MCTS vs Python MCTS head-to-head
# ---------------------------------------------------------------------------

def bench_mcts_compare() -> Dict:
    """Compare C MCTS and Python MCTS on identical positions."""
    import torch
    from orca.search import BatchedMCTS
    from orca.encoding import CGameState
    from bot import get_device

    device = get_device()
    net, loaded = _load_checkpoint_net(device)
    results = {'checkpoint_loaded': loaded, 'comparisons': []}

    # Try C MCTS
    try:
        from orca.c_mcts import CMCTSSearch
        has_c = True
    except Exception:
        has_c = False
        results['c_mcts_available'] = False
        return results

    results['c_mcts_available'] = True

    for sims in [50, 200]:
        py_mcts = BatchedMCTS(net, num_simulations=sims, batch_size=64)
        c_mcts = CMCTSSearch(net, num_simulations=sims, batch_size=64)
        N = 5

        # Set up a mid-game position
        def make_game():
            g = CGameState()
            g.place_stone(0, 0); g.place_stone(1, 0); g.place_stone(1, -1)
            g.place_stone(-1, 1); g.place_stone(0, 2); g.place_stone(2, -1)
            return g

        py_times = []
        for _ in range(N):
            game = make_game()
            t0 = time.perf_counter()
            py_mcts.search(game, temperature=0.1, add_noise=False)
            py_times.append(time.perf_counter() - t0)

        c_times = []
        for _ in range(N):
            game = make_game()
            t0 = time.perf_counter()
            c_mcts.search(game, temperature=0.1, add_noise=False)
            c_times.append(time.perf_counter() - t0)

        py_avg = sum(py_times) / N
        c_avg = sum(c_times) / N
        results['comparisons'].append({
            'sims': sims,
            'python_ms': py_avg * 1000,
            'c_mcts_ms': c_avg * 1000,
            'speedup': py_avg / c_avg if c_avg > 0 else 0,
        })

    return results


# ---------------------------------------------------------------------------
# 6. Training step benchmarks
# ---------------------------------------------------------------------------

def bench_training(n_steps=20) -> Dict:
    """Training step speed at different batch sizes."""
    import torch
    from orca.network import create_network
    from orca.samples import TrainingSample
    from orca.replay import ReplayBuffer
    from orca.data import train_step
    from bot import get_device

    device = get_device()
    net = create_network('standard').to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    buf = ReplayBuffer(capacity=10000)

    # Fill buffer with dummy samples
    for i in range(2000):
        buf.push(TrainingSample(
            encoded_state=torch.randn(7, 19, 19),
            policy_target=np.random.dirichlet(np.ones(361)),
            player=i % 2, result=1.0 if i % 2 == 0 else -1.0,
            threat_label=np.zeros(4, dtype=np.float32),
            priority=1.0,
        ))

    results = {}
    for bs in [256, 512, 1024]:
        # Warmup
        train_step(net, optimizer, buf, device, batch_size=bs)

        t0 = time.perf_counter()
        for _ in range(n_steps):
            train_step(net, optimizer, buf, device, batch_size=bs)
        elapsed = time.perf_counter() - t0

        results[f'train_bs{bs}_steps_per_sec'] = n_steps / elapsed
        results[f'train_bs{bs}_samples_per_sec'] = (n_steps * bs) / elapsed
        results[f'train_bs{bs}_ms_per_step'] = elapsed / n_steps * 1000

    return results


# ---------------------------------------------------------------------------
# 7. Augmentation benchmarks
# ---------------------------------------------------------------------------

def bench_augmentation() -> Dict:
    """Augmentation speed: grid-safe + axial rotations."""
    import torch
    from orca.samples import TrainingSample
    from orca.augment import augment_sample

    sample = TrainingSample(
        encoded_state=torch.randn(7, 19, 19),
        policy_target=np.random.dirichlet(np.ones(361)),
        player=0, result=1.0,
        threat_label=np.zeros(4, dtype=np.float32),
    )

    # Warmup
    augment_sample(sample)

    t0 = time.perf_counter()
    n = 500
    total_augs = 0
    for _ in range(n):
        augs = augment_sample(sample)
        total_augs += len(augs)
    elapsed = time.perf_counter() - t0

    return {
        'augments_per_sec': n / elapsed,
        'avg_transforms': total_augs / n,
        'samples_per_sec': total_augs / elapsed,
        'ms_per_sample': elapsed / n * 1000,
    }


# ---------------------------------------------------------------------------
# 8. Replay buffer benchmarks
# ---------------------------------------------------------------------------

def bench_replay() -> Dict:
    """Replay buffer push + sample speed."""
    import torch
    from orca.samples import TrainingSample
    from orca.replay import ReplayBuffer

    buf = ReplayBuffer(capacity=100000)

    # Push
    t0 = time.perf_counter()
    n = 10000
    for i in range(n):
        buf.push(TrainingSample(
            encoded_state=torch.randn(7, 19, 19),
            policy_target=np.random.dirichlet(np.ones(361)),
            player=i % 2, result=1.0,
            threat_label=np.zeros(4, dtype=np.float32),
        ))
    push_time = time.perf_counter() - t0

    # Sample
    t0 = time.perf_counter()
    n_batches = 200
    for _ in range(n_batches):
        buf.sample(512)
    sample_time = time.perf_counter() - t0

    return {
        'push_us_per_sample': push_time / n * 1e6,
        'push_samples_per_sec': n / push_time,
        'sample_us_per_batch': sample_time / n_batches * 1e6,
        'sample_batches_per_sec': n_batches / sample_time,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: Dict, device_info: Dict):
    W = 76
    print()
    print("+" + "=" * (W - 2) + "+")
    print("|" + "  HEXBOT PLATFORM BENCHMARK".center(W - 2) + "|")
    print("+" + "=" * (W - 2) + "+")

    dev = device_info.get('device', '?')
    gpu = device_info.get('gpu', '')
    cpu = device_info.get('cpu', '')
    ram = device_info.get('ram_gb', 0)
    vram = device_info.get('vram_gb', '')
    torch_v = device_info.get('torch', '')
    print(f"|  Device: {dev}  |  PyTorch: {torch_v}  |  Platform: {sys.platform}".ljust(W - 1) + "|")
    if gpu:
        gpu_line = f"|  GPU: {gpu}"
        if vram:
            gpu_line += f"  ({vram}GB VRAM)"
        print(gpu_line.ljust(W - 1) + "|")
    if cpu:
        print(f"|  CPU: {cpu}  ({device_info.get('cpu_cores', '?')}C/{device_info.get('cpu_threads', '?')}T)  RAM: {ram}GB".ljust(W - 1) + "|")
    print("+" + "-" * (W - 2) + "+")

    # Engine
    if 'engine' in results:
        e = results['engine']
        print(f"|  C ENGINE".ljust(W - 1) + "|")
        print(f"|    Random games:     {_fmt(e['random_games_per_sec']):>10}/s".ljust(W - 1) + "|")
        print(f"|    Scored moves:     {_fmt(e['scored_moves_per_sec']):>10}/s".ljust(W - 1) + "|")
        print(f"|    AB depth-8:       {_fmt(e.get('ab_depth8_nodes_per_sec', 0)):>10} nodes/s".ljust(W - 1) + "|")
        print(f"|    Clone:            {_fmt(e['clones_per_sec']):>10}/s".ljust(W - 1) + "|")
        print(f"|    Place+undo:       {_fmt(e['place_undo_per_sec']):>10}/s".ljust(W - 1) + "|")
        print(f"|    Threat checks:    {_fmt(e['threat_checks_per_sec']):>10}/s".ljust(W - 1) + "|")
        print("|" + "-" * (W - 2) + "|")

    # NN
    if 'nn' in results:
        n = results['nn']
        print(f"|  NEURAL NETWORK ({n.get('device', '?')})".ljust(W - 1) + "|")
        for cfg in ['fast', 'standard', 'hex-masked', 'hex-native']:
            key = f'{cfg}_bs64_pos_per_sec'
            if key in n:
                params = n.get(f'{cfg}_params', 0)
                pps = n[key]
                lat = n.get(f'{cfg}_bs64_latency_ms', 0)
                bs1 = n.get(f'{cfg}_bs1_pos_per_sec', 1)
                bs128 = n.get(f'{cfg}_bs128_pos_per_sec', 0)
                scale = f"{bs128/bs1:.0f}x" if bs1 > 0 and bs128 > 0 else ""
                print(f"|    {cfg:16s} {_fmt(params):>6} params  {_fmt(pps):>8} pos/s  {lat:>5.1f}ms  {scale:>5}".ljust(W - 1) + "|")
        print("|" + "-" * (W - 2) + "|")

    # Latency
    if 'latency' in results:
        l = results['latency']
        print(f"|  NN LATENCY (single position)".ljust(W - 1) + "|")
        print(f"|    mean={l['mean_ms']:.2f}ms  p50={l['p50_ms']:.2f}ms  p95={l['p95_ms']:.2f}ms  p99={l['p99_ms']:.2f}ms".ljust(W - 1) + "|")
        print("|" + "-" * (W - 2) + "|")

    # Search
    if 'search' in results:
        s = results['search']
        print(f"|  MCTS SEARCH".ljust(W - 1) + "|")
        for sims in [50, 100, 200]:
            key = f'mcts_sims{sims}_bs64_sec_per_search'
            if key in s:
                sps = s.get(f'mcts_sims{sims}_bs64_searches_per_sec', 0)
                t = s[key]
                print(f"|    {sims} sims (bs=64):  {t:.2f}s/search  {sps:.1f} searches/s".ljust(W - 1) + "|")
        print("|" + "-" * (W - 2) + "|")

    # Self-play
    if 'selfplay' in results:
        sp = results['selfplay']
        ckpt = "checkpoint" if sp.get('checkpoint_loaded') else "random weights"
        pps = sp.get('positions_per_sec', 0)
        n_g = sp.get('games', 0)
        print(f"|  SELF-PLAY (50 sims, process-based, {ckpt}, {n_g} games)".ljust(W - 1) + "|")
        print(f"|    {pps:.1f} pos/s  {sp['games_per_hour']:.0f} games/hr  {sp['avg_time_per_game']:.1f}s/game  {sp['avg_game_length']:.0f} avg moves".ljust(W - 1) + "|")
        print("|" + "-" * (W - 2) + "|")

    if 'gpu_selfplay' in results:
        gsp = results['gpu_selfplay']
        if gsp.get('skipped'):
            print(f"|  THREADED SELF-PLAY: skipped ({gsp.get('reason', 'no GPU')})".ljust(W - 1) + "|")
        else:
            ckpt = "checkpoint" if gsp.get('checkpoint_loaded') else "random weights"
            factor = gsp.get('throughput_factor', 0)
            pps = gsp.get('positions_per_sec', 0)
            n_g = gsp.get('games', 0)
            print(f"|  THREADED SELF-PLAY (50 sims, C MCTS, shared GPU, {ckpt}, {n_g} games)".ljust(W - 1) + "|")
            print(f"|    {pps:.1f} pos/s  {gsp['games_per_hour']:.0f} games/hr  {gsp['avg_time_per_game']:.1f}s/game  {gsp['avg_game_length']:.0f} avg moves".ljust(W - 1) + "|")
            print(f"|    wall={gsp['wall_time']:.1f}s for {gsp['games']} games  parallelism={factor:.1f}x".ljust(W - 1) + "|")
        print("|" + "-" * (W - 2) + "|")

    if 'cpu_selfplay' in results:
        csp = results['cpu_selfplay']
        ckpt = "checkpoint" if csp.get('checkpoint_loaded') else "random weights"
        pps = csp.get('positions_per_sec', 0)
        n_g = csp.get('games', 0)
        print(f"|  CPU SELF-PLAY (50 sims, forced CPU, {ckpt}, {n_g} games)".ljust(W - 1) + "|")
        print(f"|    {pps:.1f} pos/s  {csp['games_per_hour']:.0f} games/hr  {csp['avg_time_per_game']:.1f}s/game  {csp['avg_game_length']:.0f} avg moves".ljust(W - 1) + "|")
        # Show comparison if we have the main selfplay result
        if 'selfplay' in results:
            sp = results['selfplay']
            ratio = csp['games_per_hour'] / max(sp['games_per_hour'], 1)
            faster = "GPU" if ratio < 1 else "CPU"
            print(f"|    vs GPU self-play: {faster} is {abs(1/ratio - 1)*100 if ratio < 1 else abs(ratio - 1)*100:.0f}% faster".ljust(W - 1) + "|")
        print("|" + "-" * (W - 2) + "|")

    # Training
    if 'training' in results:
        tr = results['training']
        print(f"|  TRAINING STEP".ljust(W - 1) + "|")
        for bs in [256, 512, 1024]:
            key = f'train_bs{bs}_steps_per_sec'
            if key in tr:
                sps = tr[key]
                ms = tr.get(f'train_bs{bs}_ms_per_step', 0)
                samples_s = tr.get(f'train_bs{bs}_samples_per_sec', 0)
                print(f"|    batch={bs:>4}:  {sps:.1f} steps/s  {ms:.0f}ms/step  {_fmt(samples_s)} samples/s".ljust(W - 1) + "|")
        print("|" + "-" * (W - 2) + "|")

    # Augmentation
    if 'augmentation' in results:
        a = results['augmentation']
        print(f"|  AUGMENTATION".ljust(W - 1) + "|")
        print(f"|    {a['augments_per_sec']:.0f} augments/s  {a['avg_transforms']:.0f} transforms/sample  {a['ms_per_sample']:.1f}ms/sample".ljust(W - 1) + "|")
        print("|" + "-" * (W - 2) + "|")

    # Replay
    if 'replay' in results:
        r = results['replay']
        print(f"|  REPLAY BUFFER".ljust(W - 1) + "|")
        print(f"|    push: {r['push_us_per_sample']:.1f}us/sample  sample(512): {r['sample_us_per_batch']:.0f}us/batch".ljust(W - 1) + "|")
        print("|" + "-" * (W - 2) + "|")

    # Summary comparison table
    sp_modes = []
    if 'selfplay' in results and not results['selfplay'].get('skipped'):
        sp = results['selfplay']
        sp_modes.append(('Process (GPU)', sp['games_per_hour'], sp['avg_time_per_game'], sp['avg_game_length'], sp.get('positions_per_sec', 0)))
    if 'gpu_selfplay' in results and not results['gpu_selfplay'].get('skipped'):
        gsp = results['gpu_selfplay']
        sp_modes.append(('Threaded (GPU)', gsp['games_per_hour'], gsp['avg_time_per_game'], gsp['avg_game_length'], gsp.get('positions_per_sec', 0)))
    if 'cpu_selfplay' in results and not results['cpu_selfplay'].get('skipped'):
        csp = results['cpu_selfplay']
        sp_modes.append(('Process (CPU)', csp['games_per_hour'], csp['avg_time_per_game'], csp['avg_game_length'], csp.get('positions_per_sec', 0)))

    if len(sp_modes) >= 2:
        print("|" + "-" * (W - 2) + "|")
        print(f"|  SELF-PLAY COMPARISON (pos/sec = stable metric, games/hr varies with game length)".ljust(W - 1) + "|")
        print(f"|  {'Mode':<22} {'Pos/sec':>10} {'Games/hr':>10} {'Sec/game':>10} {'Avg moves':>10}".ljust(W - 1) + "|")
        best_pps = max(m[4] for m in sp_modes)
        for name, gph, spg, avg_len, pps in sp_modes:
            marker = " <-- best" if pps == best_pps else ""
            print(f"|    {name:<20} {pps:>10.1f} {gph:>10.0f} {spg:>10.1f} {avg_len:>10.0f}{marker}".ljust(W - 1) + "|")

    # Scaling test
    if 'scaling' in results:
        sc = results['scaling']
        if sc.get('levels'):
            print("|" + "-" * (W - 2) + "|")
            print(f"|  SELF-PLAY SCALING (process-based)".ljust(W - 1) + "|")
            print(f"|  {'Sims':>6} {'Games/hr':>10} {'Pos/sec':>10} {'ms/move':>10} {'Avg moves':>10}".ljust(W - 1) + "|")
            for lv in sc['levels']:
                print(f"|  {lv['sims']:>6} {lv['games_per_hour']:>10.0f} {lv['positions_per_sec']:>10.1f} {lv['ms_per_move']:>10.0f} {lv['avg_moves']:>10.0f}".ljust(W - 1) + "|")

    # MCTS comparison
    if 'mcts_compare' in results:
        mc = results['mcts_compare']
        if mc.get('comparisons'):
            print("|" + "-" * (W - 2) + "|")
            print(f"|  C MCTS vs PYTHON MCTS".ljust(W - 1) + "|")
            print(f"|  {'Sims':>6} {'Python':>12} {'C MCTS':>12} {'Speedup':>10}".ljust(W - 1) + "|")
            for c in mc['comparisons']:
                print(f"|  {c['sims']:>6} {c['python_ms']:>10.1f}ms {c['c_mcts_ms']:>10.1f}ms {c['speedup']:>9.2f}x".ljust(W - 1) + "|")

    print("+" + "=" * (W - 2) + "+")

    # Shareable markdown summary
    _print_markdown_summary(results, device_info)
    print()


def _print_markdown_summary(results: Dict, device_info: Dict):
    """Print a copy-pasteable markdown block for sharing."""
    dev = device_info.get('device', '?')
    gpu = device_info.get('gpu', dev)
    cpu = device_info.get('cpu', '?')
    ram = device_info.get('ram_gb', 0)

    # Collect key metrics
    metrics = []
    nn = results.get('nn', {})
    nn_key = 'standard_bs64_pos_per_sec'
    if nn_key in nn:
        metrics.append(('NN inference (bs64)', f"{_fmt(nn[nn_key])} pos/s"))

    search = results.get('search', {})
    sk = 'mcts_sims50_bs64_searches_per_sec'
    if sk in search:
        metrics.append(('Search (50 sims)', f"{search[sk]:.1f} searches/s"))

    sp = results.get('selfplay', {})
    if sp and not sp.get('skipped'):
        metrics.append(('Self-play (50 sims)', f"{sp['games_per_hour']:,.0f} games/hr"))
        metrics.append(('Self-play pos/s', f"{sp.get('positions_per_sec', 0):.1f} pos/s"))

    gsp = results.get('gpu_selfplay', {})
    if gsp and not gsp.get('skipped'):
        metrics.append(('Threaded GPU', f"{gsp['games_per_hour']:,.0f} games/hr"))

    # Scaling
    sc = results.get('scaling', {})
    if sc.get('levels'):
        for lv in sc['levels']:
            if lv['sims'] in (200, 400):
                metrics.append((f"Self-play ({lv['sims']} sims)", f"{lv['games_per_hour']:,.0f} games/hr"))

    engine = results.get('engine', {})
    if 'place_undo_per_sec' in engine:
        metrics.append(('C engine ops', f"{_fmt(engine['place_undo_per_sec'])} place+undo/s"))

    tr = results.get('training', {})
    tk = 'train_bs1024_steps_per_sec'
    if tk in tr:
        metrics.append(('Training (bs1024)', f"{tr[tk]:.1f} steps/s"))

    mc = results.get('mcts_compare', {})
    if mc.get('comparisons'):
        for c in mc['comparisons']:
            if c['sims'] == 50:
                metrics.append(('C MCTS speedup (50s)', f"{c['speedup']:.2f}x"))

    # Score: games/hr at 50 sims (the universal comparison number)
    score = int(sp.get('games_per_hour', 0)) if sp else 0

    # Network info
    net_params = nn.get('standard_params', 0)
    net_name = 'standard'

    print()
    print("```")
    print(f"## Hexbot Benchmark Results")
    print(f"- **Platform**: {cpu}, {ram}GB RAM, {gpu}")
    print(f"- **Network**: {net_name} ({net_params:,} params)")
    print(f"- **Score**: {score:,} games/hr at 50 sims")
    print()
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    for name, val in metrics:
        print(f"| {name} | {val} |")
    print("```")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hexbot platform benchmark - compare performance across hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Just run: python -m orca.benchmark

Modes:
  (default)   Standard benchmark (~1-2 min) - all the useful sections
  --full      Everything including slow CPU self-play (~3-5 min)
  --quick     Minimal test (~30s)
  --section X Run specific section(s), comma-separated

Sections: engine, nn, latency, search, selfplay, gpu_selfplay, cpu_selfplay,
           scaling, mcts_compare, training, augmentation, replay
        """)
    parser.add_argument("--section", type=str, default=None,
                        help="Specific section(s) to run (comma-separated)")
    parser.add_argument("--full", action="store_true",
                        help="Run everything including slow CPU self-play")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (~30s, fewer iterations)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    import torch
    from bot import get_device

    device = get_device()
    device_info = _get_system_info()

    if args.section:
        sections = args.section.lower().split(',')
    elif args.full:
        sections = [
            'engine', 'nn', 'latency', 'search', 'selfplay', 'gpu_selfplay',
            'cpu_selfplay', 'scaling', 'mcts_compare', 'training', 'augmentation', 'replay',
        ]
    elif args.quick:
        sections = ['engine', 'nn', 'selfplay', 'mcts_compare']
    else:
        # Default: everything useful, skip slow CPU self-play
        sections = [
            'engine', 'nn', 'latency', 'search', 'selfplay', 'gpu_selfplay',
            'scaling', 'mcts_compare', 'training', 'augmentation', 'replay',
        ]

    is_quick = args.quick

    results = {}

    total_start = time.perf_counter()
    for section in sections:
        section = section.strip()
        print(f"  [{section}]...", end=' ', flush=True)
        t0 = time.perf_counter()

        try:
            if section == 'engine':
                results['engine'] = bench_engine()
            elif section == 'nn':
                configs = ['fast', 'standard'] if is_quick else None
                results['nn'] = bench_nn(configs)
            elif section == 'latency':
                results['latency'] = bench_latency()
            elif section == 'search':
                results['search'] = bench_search()
            elif section == 'selfplay':
                n = 5 if is_quick else 20
                results['selfplay'] = bench_selfplay(n)
            elif section == 'gpu_selfplay':
                n = 5 if is_quick else 20
                results['gpu_selfplay'] = bench_gpu_selfplay(n)
            elif section == 'cpu_selfplay':
                n = 3 if is_quick else 5
                results['cpu_selfplay'] = bench_cpu_selfplay(n)
            elif section == 'scaling':
                n = 3 if is_quick else 5
                results['scaling'] = bench_selfplay_scaling(n)
            elif section == 'mcts_compare':
                results['mcts_compare'] = bench_mcts_compare()
            elif section == 'training':
                n = 5 if is_quick else 20
                results['training'] = bench_training(n)
            elif section == 'augmentation':
                results['augmentation'] = bench_augmentation()
            elif section == 'replay':
                results['replay'] = bench_replay()

            print(f"{time.perf_counter() - t0:.1f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            results[section] = {'skipped': True, 'reason': str(e)}

    total_time = time.perf_counter() - total_start
    print(f"\n  Total: {total_time:.1f}s")

    print_report(results, device_info)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'device': device_info, 'results': results}, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
