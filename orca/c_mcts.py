"""
C MCTS — GIL-free MCTS search using engine.c tree operations.

Architecture:
  1. Python calls mcts_select_batch() → C returns leaf encodings (GIL released)
  2. Python does net.forward_pv() → GPU batch inference (GIL held briefly)
  3. Python calls mcts_expand_backup() → C expands + backpropagates (GIL released)
  4. Repeat until num_simulations done
  5. Python calls mcts_get_policy() → C returns visit distribution

Each thread gets its own MCTSTree in C — zero shared state, zero locks.
This eliminates the GIL bottleneck that made threaded GPU self-play slower
than process-based.
"""

from __future__ import annotations
import ctypes
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    from orca.config import (
        C_PUCT, NUM_SIMULATIONS, DIRICHLET_ALPHA, DIRICHLET_EPSILON,
        BOARD_SIZE, NUM_CHANNELS,
    )
except ImportError:
    C_PUCT = 1.5
    NUM_SIMULATIONS = 400
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25
    BOARD_SIZE = 19
    NUM_CHANNELS = 7

_lib = None


def _get_lib():
    """Load and configure the C engine."""
    global _lib
    if _lib is not None:
        return _lib
    try:
        from hexgame import _load_engine
        _lib = _load_engine()
    except Exception:
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'engine.so')
        _lib = ctypes.CDLL(path)

    _FA = ctypes.POINTER(ctypes.c_float)
    _I16A = ctypes.POINTER(ctypes.c_int16)
    _I32A = ctypes.POINTER(ctypes.c_int32)
    _VP = ctypes.c_void_p

    # Tree lifecycle
    _lib.mcts_tree_new.argtypes = [_VP, ctypes.c_float, ctypes.c_int]
    _lib.mcts_tree_new.restype = _VP
    _lib.mcts_tree_destroy.argtypes = [_VP]

    # Core operations
    _lib.mcts_select_batch.argtypes = [_VP, ctypes.c_int]
    _lib.mcts_select_batch.restype = ctypes.c_int
    _lib.mcts_expand_backup.argtypes = [_VP, _FA, _FA, ctypes.c_int]
    _lib.mcts_expand_backup.restype = ctypes.c_int

    # Dirichlet + policy
    _lib.mcts_apply_dirichlet.argtypes = [_VP, _FA, ctypes.c_int, ctypes.c_float]
    _lib.mcts_get_policy.argtypes = [_VP, ctypes.c_float, _I16A, _I16A, _FA]
    _lib.mcts_get_policy.restype = ctypes.c_int

    # Accessors
    _lib.mcts_root_child_count.argtypes = [_VP]
    _lib.mcts_root_child_count.restype = ctypes.c_int
    _lib.mcts_get_leaf_encodings.argtypes = [_VP]
    _lib.mcts_get_leaf_encodings.restype = _FA
    _lib.mcts_get_leaf_count.argtypes = [_VP]
    _lib.mcts_get_leaf_count.restype = ctypes.c_int
    _lib.mcts_get_node_count.argtypes = [_VP]
    _lib.mcts_get_node_count.restype = ctypes.c_int

    return _lib


class CMCTSSearch:
    """C-accelerated MCTS. Drop-in replacement for BatchedMCTS.

    The tree select/expand/backup loop runs entirely in C with the GIL released.
    Only the NN forward pass runs in Python.
    """

    def __init__(
        self,
        net,
        num_simulations: int = NUM_SIMULATIONS,
        batch_size: int = 64,
        c_puct: float = C_PUCT,
    ):
        self.net = net
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.lib = _get_lib()
        self._device = next(net.parameters()).device

        # For analysis compatibility
        self._last_root_value = 0.0
        self._last_top_moves_data = []

    def search(
        self,
        game,  # CGameState
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> Dict[Tuple[int, int], float]:
        """Run C MCTS. Returns {(q, r): probability}."""
        lib = self.lib

        # AB hybrid pre-check: detect forced wins in shallow search
        try:
            from orca.config import USE_AB_HYBRID, AB_HYBRID_DEPTH
            if USE_AB_HYBRID and AB_HYBRID_DEPTH > 0 and hasattr(game, '_lib'):
                ab_val = ctypes.c_float(0)
                ab_ste = ctypes.c_int(0)
                game._lib.c_ab_solve(game._ptr, AB_HYBRID_DEPTH,
                                     ctypes.byref(ab_val), ctypes.byref(ab_ste))
                if abs(ab_val.value) >= 1.0:
                    q_arr = (ctypes.c_int * 10)()
                    r_arr = (ctypes.c_int * 10)()
                    s_arr = (ctypes.c_int * 10)()
                    n = game._lib.board_get_scored_moves(game._ptr, q_arr, r_arr, s_arr, 1)
                    if n > 0:
                        return {(q_arr[0], r_arr[0]): 1.0}
        except Exception:
            pass

        # Create C tree
        tree = lib.mcts_tree_new(
            game._ptr,
            ctypes.c_float(self.c_puct),
            ctypes.c_int(self.batch_size),
        )
        if not tree:
            return {}

        enc_size = self.batch_size * NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE
        first_batch = True

        try:
            sims_done = 0
            while sims_done < self.num_simulations:
                batch = min(self.batch_size, self.num_simulations - sims_done)
                n_leaves = lib.mcts_select_batch(tree, batch)

                # Apply Dirichlet noise after first expansion (root is now expanded)
                if first_batch and add_noise:
                    n_ch = lib.mcts_root_child_count(tree)
                    if n_ch > 0:
                        noise = np.random.dirichlet(
                            [DIRICHLET_ALPHA] * n_ch
                        ).astype(np.float32)
                        lib.mcts_apply_dirichlet(
                            tree,
                            noise.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            n_ch,
                            ctypes.c_float(DIRICHLET_EPSILON),
                        )
                    first_batch = False

                if n_leaves > 0:
                    # Zero-copy read of C leaf encodings as numpy
                    enc_ptr = lib.mcts_get_leaf_encodings(tree)
                    enc_np = np.ctypeslib.as_array(
                        enc_ptr,
                        shape=(n_leaves, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE),
                    )

                    # Batch NN inference
                    tensor = torch.from_numpy(enc_np).to(self._device)
                    with torch.no_grad():
                        self.net.eval()
                        pol, val = self.net.forward_pv(tensor)

                    pol_np = pol.cpu().contiguous().numpy().astype(np.float32)
                    val_np = val.cpu().contiguous().numpy().flatten().astype(np.float32)

                    # Pass back to C for expand + backup
                    lib.mcts_expand_backup(
                        tree,
                        pol_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        val_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        n_leaves,
                    )

                sims_done += batch

            # Extract final policy
            out_q = (ctypes.c_int16 * 1200)()
            out_r = (ctypes.c_int16 * 1200)()
            out_p = (ctypes.c_float * 1200)()
            n = lib.mcts_get_policy(
                tree,
                ctypes.c_float(temperature),
                out_q, out_r, out_p,
            )

            policy = {}
            for i in range(n):
                policy[(out_q[i], out_r[i])] = float(out_p[i])

            # Store for analysis
            self._last_root_value = val_np[0] if n_leaves > 0 else 0.0
            self._last_top_moves_data = sorted(
                policy.items(), key=lambda x: x[1], reverse=True
            )[:5]

            return policy

        finally:
            lib.mcts_tree_destroy(tree)

    def last_root_value(self) -> float:
        return self._last_root_value

    def last_top_moves(self, k: int = 5):
        """Return top-k moves as [(q, r, prob), ...]."""
        return [(q, r, p) for (q, r), p in self._last_top_moves_data[:k]]
