import numpy as np

import scipy.linalg
from dataclasses import dataclass, field
from typing import Tuple

from senfuslib import MultiVarGauss
from states import (ImuMeasurement,

                    GnssMeasurement, EskfState)
from states import NominalState, ErrorState

from quaternion import RotationQuaterion
from utils.cross_matrix import get_cross_matrix
from sensors import SensorGNSS
from models import ModelIMU
from solution import eskf as eskf_solu


@dataclass
class ESKF():
    model: ModelIMU
    sensor: SensorGNSS

    def predict_from_imu(self,
                         x_est_prev: EskfState,
                         z_imu: ImuMeasurement,
                         dt: float
                         ) -> EskfState:
        """Method called every time an IMU measurement is received

        Args:
            x_nom_prev: previous eskf state
            z_imu: raw IMU measurement
            dt: time since last imu measurement
        Returns:
            x_est_pred: predicted eskf state
        """
        if dt == 0:

            # TODO remove this
            x_est_prev = eskf_solu.ESKF.predict_from_imu(
                self, x_est_prev, z_imu, dt)
            return x_est_prev

        x_est_prev_nom = x_est_prev.nom
        z_corr = None  # TODO
        x_est_pred_nom = None  # TODO
        x_est_pred_err = None  # TODO

        x_est_pred = EskfState(x_est_pred_nom, x_est_pred_err)

        # TODO remove this
        x_est_pred = eskf_solu.ESKF.predict_from_imu(
            self, x_est_prev, z_imu, dt)
        return x_est_pred

    def update_err_from_gnss(self,
                             x_est_pred: EskfState,
                             z_est_pred: MultiVarGauss[GnssMeasurement],
                             z_gnss: GnssMeasurement
                             ) -> MultiVarGauss[ErrorState]:
        """Update the error state from a gnss measurement

        Hint: see (10.75)
        Due to numerical error its recomended use the robust calculation of 
        posterior covariance, somtimes called Joseph form:
            I_WH = np.eye(*P.shape) - W @ H
            x_err_cov_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)
        Remember that:
            S = H @ P @ H.T + R
        and that:
            np.linalg.solve(S, H.T) is faster than np.linalg.inv(S)

        Args:
            x_est_pred: predicted nominal and error state (gaussian)
            z_est_pred: predicted gnss measurement (gaussian)
            z_gnss: gnss measurement

        Returns:
            x_est_upd_err: updated error state gaussian
        """
        x_nom = x_est_pred.nom
        x_err = x_est_pred.err
        z_pred, S = z_est_pred

        innovation = None  # TODO
        H = None  # TODO
        P = None  # TODO
        R = None  # TODO
        W = None  # TODO
        x_err_upd = np.zeros(15)  # TODO
        I_WH = None  # TODO
        x_err_cov_upd = np.eye(15)  # TODO

        x_err_upd = ErrorState.from_array(x_err_upd)
        x_est_upd_err = MultiVarGauss[ErrorState](x_err_upd, x_err_cov_upd)

        # TODO remove this
        x_est_upd_err = eskf_solu.ESKF.update_err_from_gnss(
            self, x_est_pred, z_est_pred, z_gnss)
        return x_est_upd_err

    def inject(self,
               x_est_nom: NominalState,
               x_est_err: MultiVarGauss[ErrorState],
               ) -> EskfState:
        """Perform the injection step

        Hint: see (10.85) and (10.72) on how to inject into nominal state.
        See (10.86) on how to find error covariance after injection

        Args:
            x_nom_prev: previous nominal state
            x_err_upd: updated error state gaussian

        Returns:
            x_est_inj: eskf state after injection
        """
        pos_inj = np.zeros(3)
        vel_inj = np.zeros(3)
        ori_inj = RotationQuaterion(1, np.zeros(3))
        accm_bias_inj = np.zeros(3)
        gyro_bias_inj = np.zeros(3)

        x_nom_inj = NominalState(pos_inj, vel_inj, ori_inj,
                                 accm_bias_inj, gyro_bias_inj)

        P_inj = np.eye(15)
        x_err_inj = MultiVarGauss[ErrorState](np.zeros(15), P_inj)
        x_est_inj = EskfState(x_nom_inj, x_err_inj)

        # TODO remove this
        x_est_inj = eskf_solu.ESKF.inject(self, x_est_nom, x_est_err)
        return x_est_inj

    def update_from_gnss(self,
                         x_est_pred: EskfState,
                         z_gnss: GnssMeasurement,
                         ) -> Tuple[NominalState,
                                    MultiVarGauss[ErrorState],
                                    MultiVarGauss]:
        """Method called every time an gnss measurement is received.


        Args:
            x_est_pred: previous estimated esfk state
            z_gnss: gnss measurement

        Returns:
            x_est_upd: updated eskf state
            z_est_upd: predicted measurement gaussian

        """
        z_est_pred = None  # TODO
        x_est_upd_err = None  # TODO
        x_est_upd = None  # TODO

        # TODO remove this
        x_est_upd, z_est_pred = eskf_solu.ESKF.update_from_gnss(
            self, x_est_pred, z_gnss)

        return x_est_upd, z_est_pred
