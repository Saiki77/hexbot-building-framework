"""
Distributed training for Orca.

Current status:

- `SelfPlayPool`: fully implemented. Wraps ProcessPoolExecutor with
  game-aware batching for parallel self-play game generation on a
  single machine.

- `MultiGPUTrainer`: **STUB**. Accepts a `num_gpus` argument and
  currently falls back to single-GPU `OrcaTrainer`. Real PyTorch
  DistributedDataParallel integration is not yet implemented. See
  TODO comments in the class body.

- `RayTrainer`: **STUB**. Initializes Ray (if installed) but does not
  spawn remote actors. Currently equivalent to running `OrcaTrainer`
  with more local workers. Multi-machine scaling is not yet
  implemented. See TODO comments in the class body.

For single-machine parallel self-play, use `SelfPlayPool` directly or
`python -m orca.train --workers N`. The stub classes are kept for API
compatibility and to mark the intended extension points; they emit a
runtime warning on construction so users are not surprised.
"""

import multiprocessing
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


class SelfPlayPool:
    """Pool of self-play workers for parallel game generation.

    Wraps ProcessPoolExecutor with game-aware batching and result streaming.
    Each worker loads the network once and plays multiple games.

    Usage:
        pool = SelfPlayPool(num_workers=5, games_per_worker=4)
        results = pool.generate(net_state, num_sims=200, total_games=40)
        for samples, moves, result in results:
            replay_buffer.push(samples)
    """

    def __init__(self, num_workers: int = 5, games_per_worker: int = 2):
        self.num_workers = num_workers
        self.games_per_worker = games_per_worker

    def generate(self, net_state_dict: dict, net_config: str,
                 num_sims: int, total_games: int,
                 positions: Optional[list] = None) -> List:
        """Generate self-play games in parallel.

        Returns list of (serialized_samples, move_history, result, n_samples).
        """
        from orca.train import _self_play_worker_v2

        # Split games into futures
        gpw = self.games_per_worker
        all_results = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = []
            games_left = total_games
            while games_left > 0:
                n = min(gpw, games_left)
                futures.append(
                    pool.submit(_self_play_worker_v2, net_state_dict,
                                net_config, num_sims, n, None,
                                use_alphabeta=False)
                )
                games_left -= n

            for future in as_completed(futures):
                try:
                    batch = future.result()
                    all_results.extend(batch)
                except Exception as e:
                    print(f"  Worker error: {e}")

        return all_results


class MultiGPUTrainer:
    """STUB: Multi-GPU training is not yet implemented.

    Current behaviour: falls back to single-GPU `OrcaTrainer` regardless
    of `num_gpus`. The DistributedDataParallel integration is left as a
    deliberate stub; see TODO inside `run()`.

    Use `OrcaTrainer` directly until this is finished. The class exists
    to mark the intended extension point and keep imports stable.
    """

    def __init__(self, num_gpus: int = None, **trainer_kwargs):
        import torch
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        self.num_gpus = num_gpus
        self.trainer_kwargs = trainer_kwargs

        if not torch.cuda.is_available():
            raise RuntimeError("MultiGPUTrainer requires CUDA")

        warnings.warn(
            "MultiGPUTrainer is a stub: real DDP is not implemented. "
            "It will fall back to single-GPU OrcaTrainer. Use OrcaTrainer "
            "directly for clarity, or wait for DDP to land.",
            UserWarning, stacklevel=2,
        )

    def run(self, iterations: int = 100):
        """STUB: run on a single GPU until DDP is implemented.

        TODO: implement true multi-GPU via torch.nn.parallel.DistributedDataParallel.
        Launch pattern would be:
            torchrun --nproc_per_node=N -m orca.train

        For now this just delegates to OrcaTrainer on cuda:0.
        """
        from orca.train import OrcaTrainer
        trainer = OrcaTrainer(
            iterations=iterations,
            device='cuda:0',
            **self.trainer_kwargs,
        )
        trainer.run()


class RayTrainer:
    """STUB: Multi-machine Ray-based training is not yet implemented.

    Current behaviour: if Ray is installed, initializes a Ray runtime
    for visibility, then runs `OrcaTrainer` locally with `num_workers`
    process workers. There are no Ray remote actors and no cross-machine
    coordination yet.

    The class exists to mark the intended extension point. Use
    `OrcaTrainer --workers N` or `SelfPlayPool` directly until this is
    finished.
    """

    def __init__(self, num_workers: int = 8, **trainer_kwargs):
        self.num_workers = num_workers
        self.trainer_kwargs = trainer_kwargs
        warnings.warn(
            "RayTrainer is a stub: remote actors are not implemented. "
            "It will run locally with multiprocessing workers. Use "
            "OrcaTrainer --workers N for clarity.",
            UserWarning, stacklevel=2,
        )

    def run(self, iterations: int = 100):
        """STUB: run locally with multiprocessing until Ray actors are implemented.

        TODO: convert self-play workers into ray.remote actors so the
        pool can span machines. Driver keeps the trainer + GPU; actors
        run inference + game logic with periodic network state pulls.
        """
        try:
            import ray
            if not ray.is_initialized():
                ray.init()
            print(f"Ray cluster: {ray.cluster_resources()}")
        except ImportError:
            print("Ray not installed. Install with: pip install ray "
                  "(falling back to local multiprocessing).")

        from orca.train import OrcaTrainer
        OrcaTrainer(
            iterations=iterations,
            num_workers=self.num_workers,
            **self.trainer_kwargs,
        ).run()
