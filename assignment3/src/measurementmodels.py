from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from mytypes import MultiVarGauss
from solution import measurementmodels as measurementmodels_solu


@dataclass
class CartesianPosition2D:
    sigma_z: float

    def h(self, x: ndarray) -> ndarray:
        """Calculate the noise free measurement value of x."""

        # TODO replace this with own code
        x_h = measurementmodels_solu.CartesianPosition2D.h(self, x)
        return x_h

    def H(self, x: ndarray) -> ndarray:
        """Get the measurement matrix at x."""

        # TODO replace this with own code
        H = measurementmodels_solu.CartesianPosition2D.H(self, x)
        return H

    def R(self, x: ndarray) -> ndarray:
        """Get the measurement covariance matrix at x."""

        # TODO replace this with own code
        R = measurementmodels_solu.CartesianPosition2D.R(self, x)
        return R

    def predict_measurement(self,
                            state_est: MultiVarGauss
                            ) -> MultiVarGauss:
        """Get the predicted measurement distribution given a state estimate.
        See 4. and 6. of Algorithm 1 in the book.
        """
        z_hat = None  # TODO
        H = None  # TODO
        S = None  # TODO

        measure_pred_gauss = MultiVarGauss(z_hat, S)

        # TODO replace this with own code
        measure_pred_gauss = measurementmodels_solu.CartesianPosition2D.predict_measurement(
            self, state_est)

        return measure_pred_gauss
