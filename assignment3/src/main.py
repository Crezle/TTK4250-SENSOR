from dataloading import load_data
from plotting import Plotter

from mytypes import MultiVarGauss, Measurement2d, TimeSequence
from measurementmodels import CartesianPosition2D
from dynamicmodels import WhitenoiseAcceleration2D
from ekf import ExtendedKalmanFilter
from tuning import EKFParams, SimParams
from initialize import get_init_CV_state
from analysis import ConsistencyAnalysis


def run_ekf(ekf_params: EKFParams,
            measurements: TimeSequence[Measurement2d]):
    """This function will estimate the initial state and covariance from
    the measurements and iterate the kalman filter through the data.
    """
    # create the model and estimator object
    dyn_modl = WhitenoiseAcceleration2D(ekf_params.sigma_a)
    meas_modl = CartesianPosition2D(ekf_params.sigma_z)
    ekf_filter = ExtendedKalmanFilter(dyn_modl, meas_modl)

    state_ests = TimeSequence[MultiVarGauss]()
    state_preds = TimeSequence[MultiVarGauss]()
    meas_preds = TimeSequence[MultiVarGauss]()

    if state_est := ekf_params.init_state is None:
        t0, z0 = measurements.pop(0)
        t1, z1 = measurements.pop(0)
        state_est = get_init_CV_state(z0, z1, EKFParams)

    state_ests.insert(t1, state_est)
    for t, meas in measurements.items():
        state_est, state_pred, meas_pred = ekf_filter.step(state_est, meas)
        state_ests.insert(t, state_est)
        state_preds.insert(t, state_pred)
        meas_preds.insert(t, meas_pred)

    return state_ests, state_preds, meas_preds


def main():
    ground_truth, measurements = load_data(SimParams)

    state_ests, state_preds, meas_preds = run_ekf(EKFParams, measurements)

    consistency_analysis = ConsistencyAnalysis(state_ests, meas_preds,
                                               ground_truth, measurements)
    plotter = Plotter(ground_truth, measurements,
                      state_ests, state_preds, meas_preds,
                      consistency_analysis)
    plotter.show()


if __name__ == '__main__':
    main()
