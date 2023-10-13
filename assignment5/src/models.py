from dataclasses import dataclass, field
from typing import Generic, Optional, Sequence, TypeVar
import numpy as np
from senfuslib import DynamicModel
from scipy.linalg import expm

from states import StateCV, MeasPos

S = TypeVar('S', bound=np.ndarray)  # State type


@dataclass
class ModelCV(DynamicModel):
    std_vel: float

    def f_c(self, x: StateCV) -> StateCV:
        return np.array([x.u, x.v, 0, 0])

    def A_c(self, x: StateCV) -> np.ndarray:
        return np.array([[0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])

    def Q_c(self, x: StateCV) -> np.ndarray:
        return np.diag([0, 0, self.std_vel ** 2, self.std_vel ** 2])


@dataclass
class ModelCT(DynamicModel):
    std_vel: float
    rate: int

    def f_c(self, x: StateCV) -> StateCV:
        return np.array([x.u, x.v,
                         -self.rate*x.v, self.rate*x.u])

    def A_c(self, x: StateCV) -> np.ndarray:
        return np.array([[0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, -self.rate],
                         [0, 0, self.rate, 0]])

    def Q_c(self, state: StateCV) -> np.ndarray:
        return np.diag([0, 0, self.std_vel ** 2, self.std_vel ** 2])


@dataclass
class ModelImm(DynamicModel[StateCV]):
    models: Sequence[DynamicModel[StateCV]]
    hold_times: np.ndarray
    jump_mat: np.ndarray

    _pi_mat_c_cache: Optional[np.ndarray] = None
    _pi_mat_d_cache: dict[float, np.ndarray] = field(default_factory=dict)

    def __post__init__(self):
        self.jump_mat /= np.sum(self.jump_mat, axis=1)[:, None]

    def f_c(self, x: S, model: int) -> S:
        return self.models[model].f_c(x)

    def get_pi_mat_c(self) -> np.ndarray:
        """Compute the continuous-time transition matrix. 
        See https://en.wikipedia.org/wiki/Continuous-time_Markov_chain """
        if (pi_mat_c := self._pi_mat_c_cache) is not None:
            return pi_mat_c
        tinv = 1 / self.hold_times
        pi_mat_c = -np.diag(tinv) + tinv[:, None]*self.jump_mat
        self._pi_mat_c_cache = pi_mat_c
        return pi_mat_c

    def get_pi_mat_d(self, dt: float) -> np.ndarray:
        """Compute the discrete-time transition matrix"""
        if (pi_mat_d := self._pi_mat_d_cache.get(dt, None)) is not None:
            return pi_mat_d
        pi_mat_d = expm(self.get_pi_mat_c()*dt)
        self._pi_mat_d_cache[dt] = pi_mat_d
        return pi_mat_d

    def step_simulation(self, x: StateCV, dt: float) -> S:
        model_idx_prev = x.prev_mode
        pi_mat = self.get_pi_mat_d(dt)
        prob = pi_mat[model_idx_prev, :]
        model_idx = np.random.choice(prob.size, 1, p=prob).item()

        x_next = self.models[model_idx].step_simulation(x, dt)
        x_next.prev_mode = model_idx
        return x_next
