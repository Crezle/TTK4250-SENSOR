import numpy as np
from dataclasses import dataclass

from senfuslib import SensorModel
from states import MeasPos, StateCV


@dataclass
class SensorPos(SensorModel):
    std_pos: float

    def h(self, x: StateCV) -> np.ndarray:
        return MeasPos.from_array(self.H(x) @ x)

    def H(self, x: StateCV) -> np.ndarray:
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

    def R(self, x: StateCV) -> np.ndarray:
        return np.eye(2) * self.std_pos ** 2
