import numpy as np

from task2 import get_conds, get_double_conds, get_prob_over_line
from sensor_model import LinearSensorModel2d
from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from utils.plotting import show_results


def main():
    x_bar = np.zeros(2)
    P = 25 * np.eye(2)

    z_c = np.array([2, 14])
    H_c = np.eye(2)
    R_c = np.array([[79, 36], [36, 36]])

    z_r = np.array([-4, 6])
    H_r = np.eye(2)
    R_r = np.array([[28, 4], [4, 22]])

    state_est = MultiVarGauss2d(x_bar, P)

    camera = LinearSensorModel2d(H_c, R_c)
    radar = LinearSensorModel2d(H_r, R_r)

    meas_c = Measurement2d(z_c)
    meas_r = Measurement2d(z_r)

    meas_gauss_c = camera.meas_as_gauss(meas_c)
    meas_gauss_r = radar.meas_as_gauss(meas_r)

    cond_c, cond_r = get_conds(state_est, camera, meas_c, radar, meas_r)

    print(f'cond_c={cond_c}\n cond_r={cond_r}')

    cond_cr, cond_rc = get_double_conds(state_est,
                                        camera, meas_c,
                                        radar, meas_r)

    print(f'cond_cr={cond_cr}\n cond_rc={cond_rc}')

    prob_above_line_cr = get_prob_over_line(cond_cr)

    print(f"Prob that cond_cr is above x_2 = x_1 + 5 is {prob_above_line_cr}")

    show_results(state_est, meas_gauss_c, meas_gauss_r,
                 cond_c, cond_r, cond_cr)


if __name__ == '__main__':
    main()
