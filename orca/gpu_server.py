"""
GPU inference server for CUDA-accelerated self-play.

On CUDA systems, the GPU is underutilized because each worker process
runs NN inference on CPU via ONNX. This server keeps the network on GPU
and batches inference requests from multiple workers.

Architecture:
    Workers (CPU) -> Queue -> GPU Server -> Queue -> Workers

    Workers play games using the C engine (CPU), send board positions
    to the GPU server for NN evaluation, and receive policy+value back.

This gives 5-10x speedup on CUDA by:
1. Network stays on GPU (no CPU<->GPU transfer per worker)
2. Batched inference (multiple positions in one forward pass)
3. Workers don't need to load PyTorch (smaller memory footprint)

Usage:
    from orca.gpu_server import GPUInferenceServer

    server = GPUInferenceServer(net, device='cuda', batch_size=64)
    server.start()

    # Workers call:
    policy, value = server.evaluate(encoded_state)

    server.stop()
"""

import os
import sys
import time
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


class GPUInferenceServer:
    """Centralized GPU inference for self-play workers.

    Collects evaluation requests from multiple threads/processes,
    batches them, runs a single GPU forward pass, and returns results.

    Workers call evaluate() which blocks until the batch is processed.
    The server thread collects requests and fires batches either when
    batch_size is reached or after max_wait_ms.
    """

    def __init__(self, net, device='cuda', batch_size: int = 64,
                 max_wait_ms: float = 5.0):
        """
        Args:
            net: PyTorch network (moved to device)
            device: torch device string
            batch_size: max positions per batch
            max_wait_ms: max wait before firing an incomplete batch
        """
        self.net = net.to(device)
        self.net.eval()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_wait = max_wait_ms / 1000.0

        self._request_queue = deque()
        self._lock = threading.Lock()
        self._new_request = threading.Event()
        self._running = False
        self._thread = None

        # Stats
        self.total_evaluations = 0
        self.total_batches = 0

    def start(self):
        """Start the inference server thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the inference server."""
        self._running = False
        self._new_request.set()  # wake up the loop
        if self._thread:
            self._thread.join(timeout=2.0)

    def evaluate(self, encoded_state: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Submit a position for evaluation. Blocks until result is ready.

        Args:
            encoded_state: (C, H, W) tensor

        Returns:
            (policy_logits, value) - numpy array and float
        """
        result_event = threading.Event()
        result_holder = [None, None]  # [policy, value]

        with self._lock:
            self._request_queue.append((encoded_state, result_event, result_holder))
        self._new_request.set()

        result_event.wait()  # block until batch is processed
        return result_holder[0], result_holder[1]

    def evaluate_batch(self, states: List[torch.Tensor]) -> List[Tuple[np.ndarray, float]]:
        """Submit multiple positions at once. More efficient than individual calls.

        Args:
            states: list of (C, H, W) tensors

        Returns:
            list of (policy_logits, value) tuples
        """
        events = []
        holders = []
        with self._lock:
            for s in states:
                evt = threading.Event()
                holder = [None, None]
                self._request_queue.append((s, evt, holder))
                events.append(evt)
                holders.append(holder)
        self._new_request.set()

        # Wait for all
        for evt in events:
            evt.wait()
        return [(h[0], h[1]) for h in holders]

    def _loop(self):
        """Main server loop - collects and processes batches."""
        while self._running:
            # Wait for requests
            self._new_request.wait(timeout=self.max_wait)
            self._new_request.clear()

            # Collect batch
            batch = []
            with self._lock:
                while self._request_queue and len(batch) < self.batch_size:
                    batch.append(self._request_queue.popleft())

            if not batch:
                continue

            # Stack tensors and run inference
            states = torch.stack([req[0] for req in batch]).to(self.device)

            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        policy_logits, values = self.net.forward_pv(states)
                else:
                    policy_logits, values = self.net.forward_pv(states)

            # Distribute results
            policy_np = policy_logits.cpu().numpy()
            values_np = values.cpu().numpy()

            for i, (_, evt, holder) in enumerate(batch):
                holder[0] = policy_np[i]
                holder[1] = float(values_np[i])
                evt.set()

            self.total_evaluations += len(batch)
            self.total_batches += 1

    @property
    def avg_batch_size(self) -> float:
        if self.total_batches == 0:
            return 0
        return self.total_evaluations / self.total_batches

    def __repr__(self):
        return (f"GPUInferenceServer(device={self.device}, batch={self.batch_size}, "
                f"evals={self.total_evaluations}, batches={self.total_batches}, "
                f"avg_batch={self.avg_batch_size:.1f})")
