from dataclasses import dataclass
import numpy as np


@dataclass
class Measurement2d:
    """A 2d measurement."""
    value: np.ndarray
