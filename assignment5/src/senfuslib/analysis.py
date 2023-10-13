from typing import Sequence, TypeVar, Union
import numpy as np
from scipy.stats import chi2
from dataclasses import dataclass
from senfuslib import TimeSequence, MultiVarGauss, NamedArray

S = TypeVar('S', bound=np.ndarray)  # State type
M = TypeVar('M', bound=np.ndarray)  # Measurement type


@dataclass
class ConsistencyData:
    tseq: TimeSequence[MultiVarGauss[S]]

    lower: float
    median: float
    upper: float
    above_median: float
    in_interval: float
    alpha: float
    dof: int

    @property
    def thresholds(self):
        return self.lower, self.median, self.upper


@dataclass
class ConsistencyAnalysis:
    x_gts: TimeSequence[S]
    zs: TimeSequence[M]
    x_ests: TimeSequence[MultiVarGauss[S]]
    z_preds: TimeSequence[MultiVarGauss[S]]

    def get_nis(self, indices=None, alpha=0.95) -> ConsistencyData:
        return self._get_nisornees(self.z_preds, self.zs, indices, alpha)

    def get_nees(self, indices=None, alpha=0.95) -> ConsistencyData:
        return self._get_nisornees(self.x_ests, self.x_gts, indices, alpha)

    def get_err(self, idx):
        if isinstance(idx, str):
            idx = getattr(self.x_gts.values[0].indices, idx)[0]
        err_tseq = TimeSequence[float]()
        std_tseq = TimeSequence[float]()

        for t, x_est in self.x_ests.items():
            err_tseq.insert(t, x_est.mean[idx] - self.x_gts[t][idx])
            std_tseq.insert(t, x_est.cov[idx, idx])
        return err_tseq, std_tseq

    def _get_nisornees(self,
                       gauss_tseq: TimeSequence[MultiVarGauss[NamedArray]],
                       vec_tseq: TimeSequence[NamedArray],
                       indices: Sequence[Union[int, str]],
                       alpha: float,
                       ) -> TimeSequence[float]:

        g0 = gauss_tseq.values[0]
        indices_new = []
        for i, idx in enumerate(indices):
            if isinstance(idx, str):
                new_idx = getattr(g0.mean.indices, idx)
                assert len(new_idx) == 1
                indices_new.append(new_idx[0])
            else:
                indices_new.append(idx)
        val_tseq = TimeSequence[float]()
        for t, g in gauss_tseq.items():
            g_marginal = g.get_marginalized(indices_new)
            vec_marginal = vec_tseq[t][indices_new]
            val_tseq.insert(t, g_marginal.mahalanobis_distance(vec_marginal))

        vals = val_tseq.values

        dof = len(indices_new)
        lower, upper = chi2.interval(alpha, dof)
        median = chi2.mean(dof)
        n = len(val_tseq)
        above_median = np.count_nonzero(vals > median)/n
        in_interval = np.count_nonzero((vals > lower) & (vals < upper))/n

        return ConsistencyData(val_tseq, lower, median, upper,
                               above_median, in_interval, alpha, dof)
