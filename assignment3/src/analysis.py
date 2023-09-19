from typing import Union
import numpy as np
from scipy.stats import chi2
from mytypes import TimeSequence, MultiVarGauss, Measurement2d
from dataclasses import dataclass
from solution import analysis as analysis_solu


def get_nis(meas_pred: MultiVarGauss, meas: Measurement2d) -> float:
    """Calculate the normalized innovation squared (NIS), this can be seen as 
    the normalized measurement prediction error squared. 
    See (4.66 in the book). 
    Tip: use the mahalanobis_distance method of meas_pred, (3.2) in the book
    """

    # TODO replace this with own code
    nis = analysis_solu.get_nis(meas_pred, meas)

    return nis


def get_nees(state_est: MultiVarGauss, x_gt: np.ndarray):
    """Calculate the normalized estimation error squared (NEES)
    See (4.65 in the book). 
    Tip: use the mahalanobis_distance method of x_gauss, (3.2) in the book
    """

    # TODO replace this with own code
    NEES = analysis_solu.get_nees(state_est, x_gt)
    return NEES


@dataclass
class ConsistencyData:
    tseq: TimeSequence[float]
    lower: float
    median: float
    upper: float
    above_median: float
    in_interval: float
    alpha: float
    df: int
    thresholds = property(lambda self: (self.lower, self.median, self.upper))


class ConsistencyAnalysis:
    def __init__(self,
                 states: TimeSequence[MultiVarGauss],
                 meas_preds: TimeSequence[MultiVarGauss],
                 gt: TimeSequence[np.ndarray],
                 measurements: TimeSequence[Measurement2d],
                 ):

        self.nis_pos = self.get_cons_data(get_nis, meas_preds, measurements)
        self.nes_all = self.get_cons_data(get_nees, states, gt, range(4))
        self.nes_pos = self.get_cons_data(get_nees, states, gt, range(2))
        self.nes_vel = self.get_cons_data(get_nees, states, gt, range(2, 4))

    @staticmethod
    def get_cons_data(func: Union[get_nis, get_nees],
                      ests: TimeSequence[MultiVarGauss],
                      trues: TimeSequence[Union[np.ndarray, Measurement2d]],
                      marginal_indices=None,
                      alpha=0.95):
        data = TimeSequence()
        for t, est in ests.items():
            true = trues.get_t(t)
            if marginal_indices:
                est = est.get_marginalized(marginal_indices)
                if isinstance(true, np.ndarray):
                    true = true[marginal_indices]
                elif isinstance(true, Measurement2d):
                    true = true.value[marginal_indices]
                else:
                    raise TypeError("true should be ndarray or Measurement2d")
            data.insert(t, func(est, true))

        df = len(marginal_indices) if marginal_indices else est.ndim

        lower, upper = chi2.interval(alpha, df)
        median = chi2.mean(df)

        values = data.values_as_array()
        n = len(values)
        above_median = np.count_nonzero(values > median)/n
        in_interval = np.count_nonzero((values > lower) & (values < upper))/n

        return ConsistencyData(data, lower, median, upper,
                               above_median, in_interval, alpha, df)
