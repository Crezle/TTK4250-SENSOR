from dataclasses import dataclass

import numpy as np
import scipy as sp
from numpy import ndarray
from mytypes import MultiVarGauss
from solution import dynamicmodels as dynamicmodels_solu


@dataclass
class WhitenoiseAcceleration2D:
    """
    A white noise acceleration model, also known as constan velocity.
    States are position and speed.
    """
    sigma_a: float  # noise standard deviation

    def f(self, x: ndarray, dt: float,) -> ndarray:
        """Calculate the zero noise transition from x given dt."""

        x_next = self.F(x, dt) @ x
        
        return x_next

    def F(self, x: ndarray, dt: float,) -> ndarray:
        """Calculate the discrete transition matrix given dt
        See (4.64) in the book."""

        A = np.array([[0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])
        F = sp.linalg.expm(A * dt)
        
        return F

    def Q(self, x: ndarray, dt: float,) -> ndarray:
        """Calculate the discrete transition Covariance.
        See(4.64) in the book."""

        # TODO replace this with own code
        Q = np.identity(4) * dt
        
        Q[0][0] *= (dt**2)/3
        Q[1][1] *= (dt**2)/3
        Q[0][2] = (dt**2)/2
        Q[1][3] = (dt**2)/2
        Q[2][0] = (dt**2)/2
        Q[3][1] = (dt**2)/2

        Q *= self.sigma_a**2

        return Q

    def predict_state(self,
                      state_est: MultiVarGauss,
                      dt: float,
                      ) -> MultiVarGauss:
        """Given the current state estimate, 
        calculate the predicted state estimate.
        See 2. and 3. of Algorithm 1 in the book."""
        x_upd_prev, P = state_est
        x_upd_prev = np.array(x_upd_prev)
        P = np.array(P)

        F = self.F(x_upd_prev, dt)  # TODO
        Q = self.Q(x_upd_prev, dt)  # TODO

        x_pred = F @ x_upd_prev  # TODO
        P_pred = F @ P @ F.T + Q  # TODO

        state_pred_gauss = MultiVarGauss(x_pred, P_pred)

        return state_pred_gauss
