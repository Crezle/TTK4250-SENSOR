import numpy as np
from dataclasses import dataclass, field
from typing import Sequence

from senfuslib import SensorModel, MultiVarGauss
from states import MeasPos, StateCV
from scipy.stats import poisson
import logging
from functools import cache


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


@dataclass
class SensorPosClutter(SensorModel[Sequence[MeasPos]]):
    sensor: SensorModel[MeasPos]
    prob_detect: float  # P_D in the book
    clutter_density: float  # lambda in the book

    x_min: float = field(default=None)
    x_max: float = field(default=None)
    y_min: float = field(default=None)
    y_max: float = field(default=None)

    assert_inside: bool = field(default=True)

    @property
    def area(self) -> float:
        """V_k in the book."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def h(self, x: StateCV) -> MeasPos:
        meas = self.sensor.h(x)
        inside = (self.x_min <= meas.x <= self.x_max
                  and self.y_min <= meas.y <= self.y_max)
        msg = f"Measurement {meas} outside of sensor area"
        self.area
        if inside:
            return meas
        elif self.assert_inside:
            raise ValueError(msg)
        else:
            logging.warning("Measurement outside of sensor area")
            return None

    def H(self, x: StateCV) -> np.ndarray:
        return self.sensor.H(x)

    def R(self, x: StateCV) -> np.ndarray:
        return self.sensor.R(x)

    def sample_from_state(self, x: StateCV
                          ) -> Sequence[MeasPos]:
        zs = []
        if np.random.rand() < self.prob_detect:
            zs.append(super().sample_from_state(x))
        n_clutter = poisson.rvs(self.clutter_density * self.area)
        clutter = np.random.uniform((self.x_min, self.y_min),
                                    (self.x_max, self.y_max),
                                    size=(n_clutter, 2))
        zs.extend(MeasPos(*m, isclutter=True) for m in clutter)
        np.random.shuffle(zs)
        return zs
