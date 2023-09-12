from dataclasses import dataclass
import numpy as np

from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from solution import sensor_model as sensor_model_solu


@dataclass
class LinearSensorModel2d:
    """A 2d sensor model"""
    H: np.ndarray
    R: np.ndarray

    def get_pred_meas(self, state_est: MultiVarGauss2d) -> MultiVarGauss2d:
        pred_mean = self.H @ state_est.mean # TODO (z^c)
        pred_cov = self.R + self.H @ state_est.cov @ self.H.T # TODO

        pred_meas = MultiVarGauss2d(pred_mean, pred_cov)

        return pred_meas

    def meas_as_gauss(self, meas: Measurement2d) -> MultiVarGauss2d:
        """Get the measurement as a Gaussian distribution."""
        meas_gauss = MultiVarGauss2d(meas.value, self.R)
        return meas_gauss
