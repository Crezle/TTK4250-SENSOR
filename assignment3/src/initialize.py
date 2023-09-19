import numpy as np

from mytypes import Measurement2d, MultiVarGauss
from tuning import EKFParams
from solution import initialize as initialize_solu


def get_init_CV_state(meas0: Measurement2d, meas1: Measurement2d,
                      ekf_params: EKFParams) -> MultiVarGauss:
    """This function will estimate the initial state and covariance from
    the two first measurements"""
    dt = meas1.dt
    z0, z1 = meas0.value, meas1.value
    sigma_a = ekf_params.sigma_a
    sigma_z = ekf_params.sigma_z

    K_p1 = np.identity(2)
    K_p0 = np.zeros((2, 2))
    K_u1 = (1/dt) * np.identity(2)
    K_u0 = -K_u1
    
    K_top = np.hstack((K_p1, K_p0))
    K_bot = np.hstack((K_u1, K_u0))
    
    K = np.vstack((K_top, K_bot))
    z = np.hstack((np.array(z1), np.array(z0)))
    
    R = np.identity(2) * sigma_z**2
    br_matr = (1/dt**2) * (2*R + sigma_a**2 * (dt**3)/3 * np.identity(2))
    
    cov_top = np.hstack((R, (1/dt) * R))
    cov_bot = np.hstack(((1/dt) * R, br_matr))
    
    mean = K @ z  # TODO
    cov = np.vstack((cov_top, cov_bot)) # TODO

    init_state = MultiVarGauss(mean, cov)

    return init_state
