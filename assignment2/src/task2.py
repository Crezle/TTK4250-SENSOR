import numpy as np
from scipy.stats import norm
from typing import Tuple

from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from sensor_model import LinearSensorModel2d
from conditioning import get_cond_state
from solution import task2 as task2_solu


def get_conds(state: MultiVarGauss2d,
              sens_model_c: LinearSensorModel2d, meas_c: Measurement2d,
              sens_model_r: LinearSensorModel2d, meas_r: Measurement2d
              ) -> Tuple[MultiVarGauss2d, MultiVarGauss2d]:

    cond_c = get_cond_state(state, sens_model_c, meas_c)
    cond_r = get_cond_state(state, sens_model_r, meas_r)

    return cond_c, cond_r


def get_double_conds(state: MultiVarGauss2d,
                     sens_model_c: LinearSensorModel2d, meas_c: Measurement2d,
                     sens_model_r: LinearSensorModel2d, meas_r: Measurement2d
                     ) -> Tuple[MultiVarGauss2d, MultiVarGauss2d]:

    cond_c, cond_r = get_conds(state, sens_model_c, meas_c, sens_model_r, meas_r)
    
    cond_cr = get_cond_state(cond_c, sens_model_r, meas_r)
    cond_rc = get_cond_state(cond_r, sens_model_c, meas_c)

    return cond_cr, cond_rc


def get_prob_over_line(gauss: MultiVarGauss2d) -> float:

    problem_transform = [-1, 1]
    
    # TODO replace this with own code
    prob = task2_solu.get_prob_over_line(gauss)
    return prob
