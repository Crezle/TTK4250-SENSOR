from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import scipy as sp # Used for block diagonal matrix

from senfuslib import MultiVarGauss
from states import NominalState, GnssMeasurement, EskfState
from utils.cross_matrix import get_cross_matrix
from solution import sensors as sensors_solu


@dataclass
class SensorGNSS:
    gnss_std_ne: float
    gnss_std_d: float
    lever_arm: 'np.ndarray[3]'
    R: 'np.ndarray[3, 3]' = field(init=False)

    def __post_init__(self):
        self.R = np.diag([self.gnss_std_ne**2,
                          self.gnss_std_ne**2,
                          self.gnss_std_d**2])

    def H(self, x_nom: NominalState) -> 'np.ndarray[3, 15]':
        """Get the measurement jacobian, H with respect to the error state.

        Hint: the gnss antenna has a relative position to the center given by
        self.lever_arm. How will the gnss measurement change if the drone is 
        rotated differently? Use get_cross_matrix and some other stuff. 

        Returns:
            H (ndarray[3, 15]): the measurement matrix
        """
        H = np.block([np.eye(3), np.zeros((3, 3)), -x_nom.ori.as_rotmat() @ get_cross_matrix(self.lever_arm), np.zeros((3, 6))])

        return H

    def pred_from_est(self, x_est: EskfState,
                      ) -> MultiVarGauss[GnssMeasurement]:
        """Predict the gnss measurement

        Args:
            x_est: eskf state

        Returns:
            z_gnss_pred_gauss: gnss prediction gaussian
        """
        x_est_nom = x_est.nom
        x_est_err = x_est.err
        epsilon = x_est_nom.ori.epsilon
        eta = x_est_nom.ori.eta
        Q = (1/2) * np.array([[-epsilon[0], -epsilon[1], -epsilon[2]], 
                              [eta, -epsilon[2], epsilon[1]],
                              [epsilon[2], eta, -epsilon[0]],
                              [-epsilon[1], epsilon[0], eta]])
        
        X_delta = sp.linalg.block_diag(np.eye(6), Q, np.eye(6))
        
        z_q = 2 * np.array([[eta, epsilon[0], -epsilon[1], -epsilon[2]],
                            [epsilon[2], epsilon[1], epsilon[0], -eta],
                            [epsilon[1], epsilon[2], eta, epsilon[0]]])
        H_x = np.block([np.zeros((3, 6)), z_q, np.zeros((3, 6))])

        # H = H_x @ X_delta, H_x is (3x16), X_delta is (16x15)
        
        H = H_x @ X_delta

        z_pred = H @ x_est_err.mean  # TODO
        S = H @ x_est_err.cov @ H.T + self.R  # TODO

        z_pred = GnssMeasurement.from_array(z_pred)
        z_gnss_pred_gauss = MultiVarGauss[GnssMeasurement](z_pred, S)

        return z_gnss_pred_gauss
