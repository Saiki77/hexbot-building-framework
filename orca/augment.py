"""Hex-valid data augmentation (symmetry transforms)."""
from typing import List
import numpy as np
import torch

from orca.samples import TrainingSample

try:
    from orca.config import BOARD_SIZE
except ImportError:
    BOARD_SIZE = 19


def _axial_rotate(state, policy, rotate_fn):
    """Rotate state and policy via axial coordinate transform."""
    N = BOARD_SIZE
    C = N // 2
    s_new = np.zeros_like(state)
    p_new = np.zeros_like(policy)
    for i in range(N):
        for j in range(N):
            q, r = i - C, j - C
            q2, r2 = rotate_fn(q, r)
            i2, j2 = q2 + C, r2 + C
            if 0 <= i2 < N and 0 <= j2 < N:
                s_new[:, i2, j2] = state[:, i, j]
                p_new[i2, j2] = policy[i, j]
    return s_new, p_new


def augment_sample(sample: TrainingSample) -> List[TrainingSample]:
    """Apply hex symmetry augmentations.

    Returns up to 7 new samples:
    - 3 grid-safe transforms (0.8x priority)
    - 4 axial rotations: 60, 120, 240, 300 degrees (0.7x priority)
    """
    state = sample.encoded_state.numpy()
    policy = sample.policy_target.reshape(BOARD_SIZE, BOARD_SIZE)
    aug = []

    grid_transforms = [
        (lambda s: s[:, ::-1, ::-1].copy(),
         lambda p: p[::-1, ::-1].copy()),
        (lambda s: s.transpose(0, 2, 1).copy(),
         lambda p: p.T.copy()),
        (lambda s: s[:, ::-1, ::-1].transpose(0, 2, 1).copy(),
         lambda p: p[::-1, ::-1].T.copy()),
    ]

    for s_fn, p_fn in grid_transforms:
        s_new = s_fn(state)
        p_new = p_fn(policy).flatten()
        ps = p_new.sum()
        if ps > 0:
            p_new = p_new / ps
        aug.append(TrainingSample(
            encoded_state=torch.from_numpy(np.ascontiguousarray(s_new)),
            policy_target=np.ascontiguousarray(p_new),
            player=sample.player, result=sample.result,
            threat_label=sample.threat_label,
            priority=sample.priority * 0.8,
        ))

    axial_rotations = [
        lambda q, r: (-r, q + r),
        lambda q, r: (-q - r, q),
        lambda q, r: (r, -q - r),
        lambda q, r: (q + r, -q),
    ]

    for rot_fn in axial_rotations:
        s_new, p_new_2d = _axial_rotate(state, policy, rot_fn)
        p_new = p_new_2d.flatten()
        ps = p_new.sum()
        if ps <= 0:
            continue
        p_new = p_new / ps
        aug.append(TrainingSample(
            encoded_state=torch.from_numpy(np.ascontiguousarray(s_new)),
            policy_target=np.ascontiguousarray(p_new),
            player=sample.player, result=sample.result,
            threat_label=sample.threat_label,
            priority=sample.priority * 0.7,
        ))

    return aug
