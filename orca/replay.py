"""Prioritized experience replay buffer."""
import collections
from typing import List, Tuple
import numpy as np

from orca.samples import TrainingSample

try:
    from orca.config import REPLAY_BUFFER_SIZE
except ImportError:
    REPLAY_BUFFER_SIZE = 400_000


class ReplayBuffer:
    """Prioritized replay buffer - samples proportional to priority."""

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self.buffer: collections.deque = collections.deque(maxlen=capacity)
        self.priorities: collections.deque = collections.deque(maxlen=capacity)

    def push(self, sample: TrainingSample) -> None:
        self.buffer.append(sample)
        self.priorities.append(sample.priority)

    def sample(self, batch_size: int) -> Tuple[List[TrainingSample], List[int]]:
        n = min(batch_size, len(self.buffer))
        priors = np.array(self.priorities, dtype=np.float64)
        priors /= priors.sum()
        indices = np.random.choice(len(self.buffer), size=n, replace=False, p=priors)
        buf = list(self.buffer)
        return [buf[i] for i in indices], indices.tolist()

    def update_priorities(self, indices: List[int], errors: List[float]) -> None:
        for idx, err in zip(indices, errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = abs(err) + 0.01

    def __len__(self) -> int:
        return len(self.buffer)
