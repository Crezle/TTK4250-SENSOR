from solution import ekf as ekf_solu
"""
Notation:
----------
x is generally used for either the state or the mean of a gaussian. It should be clear from context which it is.
P is used about the state covariance
z is a single measurement
Z are multiple measurements so that z = Z[k] at a given time step k
v is the innovation z - h(x)
S is the innovation covariance
"""
from dataclasses import dataclass
import numpy as np

from dynamicmodels import WhitenoiseAcceleration2D
from measurementmodels import CartesianPosition2D
from mytypes import MultiVarGauss, Measurement2d


@dataclass
class ExtendedKalmanFilter:
    dyn_modl: WhitenoiseAcceleration2D
    sens_modl: CartesianPosition2D

    def step(self,
             state_old: MultiVarGauss,
             meas: Measurement2d,
             ) -> MultiVarGauss:
        """Given previous state estimate and measurement, 
        return new state estimate.

        Relationship between variable names and equations in the book:
        \hat{x}_{k|k_1} = pres_state.mean
        P_{k|k_1} = pres_state.cov
        \hat{z}_{k|k-1} = pred_meas.mean
        \hat{S}_k = pred_meas.cov
        \hat{x}_k = upd_state_est.mean
        P_k = upd_state_est.cov
        """
        state_pred = self.dyn_modl.predict_state(state_old, meas.dt)
        meas_pred = self.sens_modl.predict_measurement(state_pred)

        z = meas.value
        z_hat = meas_pred.mean
        x_hat = state_pred.mean

        H = self.sens_modl.H(state_pred)  # TODO
        P_pred = state_pred.cov
        S = meas_pred.cov
        kalman_gain = P_pred @ H.T @ np.linalg.inv(S)  # TODO
        innovation = z - z_hat  # TODO

        state_upd_mean = x_hat + kalman_gain @ innovation  # TODO
        state_upd_cov = (np.identity(len(x_hat)) - kalman_gain @ H) @ P_pred  # TODO

        state_upd = MultiVarGauss(state_upd_mean, state_upd_cov)

        return state_upd, state_pred, meas_pred
