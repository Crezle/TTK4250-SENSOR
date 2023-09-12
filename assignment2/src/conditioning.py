import numpy as np
from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from sensor_model import LinearSensorModel2d
from solution import conditioning as conditioning_solu


def get_cond_state(state: MultiVarGauss2d,
                   sens_modl: LinearSensorModel2d,
                   meas: Measurement2d
                   ) -> MultiVarGauss2d:
    pred_meas = None  # TODO (What to do with these?)
    pred_cov = sens_modl.R + sens_modl.H @ state.cov @ sens_modl.H.T # Added by me
    kalman_gain = state.cov @ sens_modl.H @ np.linalg.inv(pred_cov)  # TODO
    innovation = None  # TODO (What to do with these?)
    cond_mean = state.mean + kalman_gain @ (meas.value - sens_modl.H @ state.mean)  # TODO
    cond_cov = (np.identity(2) - kalman_gain @ sens_modl.H) @ state.cov # TODO

    cond_state = MultiVarGauss2d(cond_mean, cond_cov)

    return cond_state
