from dataclasses import dataclass

import numpy as np
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

        # TODO replace this with own code
        x_next = dynamicmodels_solu.WhitenoiseAcceleration2D.f(self, x, dt)
        return x_next

    def F(self, x: ndarray, dt: float,) -> ndarray:
        """Calculate the discrete transition matrix given dt
        See (4.64) in the book."""

        # TODO replace this with own code
        F = dynamicmodels_solu.WhitenoiseAcceleration2D.F(self, x, dt)
        return F

    def Q(self, x: ndarray, dt: float,) -> ndarray:
        """Calculate the discrete transition Covariance.
        See(4.64) in the book."""

        # TODO replace this with own code
        Q = dynamicmodels_solu.WhitenoiseAcceleration2D.Q(self, x, dt)
        return Q

    def predict_state(self,
                      state_est: MultiVarGauss,
                      dt: float,
                      ) -> MultiVarGauss:
        """Given the current state estimate, 
        calculate the predicted state estimate.
        See 2. and 3. of Algorithm 1 in the book."""
        x_upd_prev, P = state_est

        F = None  # TODO
        Q = None  # TODO

        x_pred = None  # TODO
        P_pred = None  # TODO

        state_pred_gauss = MultiVarGauss(x_pred, P_pred)

        # TODO replace this with own code
        state_pred_gauss = dynamicmodels_solu.WhitenoiseAcceleration2D.predict_state(
            self, state_est, dt)

        return state_pred_gauss
