"""Training sample dataclass."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch

@dataclass
class TrainingSample:
    encoded_state: torch.Tensor
    policy_target: np.ndarray
    player: int
    result: Optional[float] = None
    threat_label: Optional[np.ndarray] = None
    priority: float = 1.0

_ZERO_THREAT = np.zeros(4, dtype=np.float32)
