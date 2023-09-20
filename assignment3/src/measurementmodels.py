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
        
        x_h = self.H(x) @ x
        
        return x_h

    def H(self, x: ndarray) -> ndarray:
        """Get the measurement matrix at x."""

        H = np.block([np.identity(2), np.zeros((2,2))])    

        return H

    def R(self, x: ndarray) -> ndarray:
        """Get the measurement covariance matrix at x."""

        # TODO replace this with own code
        R = np.identity(2) * self.sigma_z**2

        return R

    def predict_measurement(self,
                            state_est: MultiVarGauss
                            ) -> MultiVarGauss:
        """Get the predicted measurement distribution given a state estimate.
        See 4. and 6. of Algorithm 1 in the book.
        """
        x_hat, P = state_est
        z_hat = self.H(x_hat) @ x_hat # TODO
        H = self.H(x_hat)  # TODO
        S = H @ P @ H.T + self.R(x_hat)  # TODO

        measure_pred_gauss = MultiVarGauss(z_hat, S)

        return measure_pred_gauss
