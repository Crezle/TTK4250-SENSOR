from dataclasses import dataclass
import numpy as np
from solution import gaussian as gaussian_solu


@dataclass
class MultiVarGauss2d:
    """A 2d multivariate Gaussian distribution."""
    mean: np.ndarray
    cov: np.ndarray

    def __iter__(self):
        """Enable iteration over the mean and covariance.
        i.e.
            est = MultiVarGauss2d(mean=[1, 2], cov=[[1, 0], [0, 1]])
            mean, cov = est
        """
        return iter((self.mean, self.cov))

    def get_transformed(self, lin_transform: np.ndarray) -> 'MultiVarGauss2d':
        transformed_mean = lin_transform @ self.mean  # TODO
        transformed_cov = lin_transform @ self.cov @ lin_transform.T  # TODO
        transformed = MultiVarGauss2d(transformed_mean, transformed_cov)

        return transformed

    def __str__(self) -> str:
        """Used for pretty printing"""
        def sci(x): return np.format_float_scientific(x, 3, pad_left=2)
        out = '\n'
        for i in range(self.mean.shape[0]):
            mline = sci(self.mean[i])
            cline = ' |'.join(sci(self.cov[i, j])
                              for j in range(self.cov.shape[1]))
            out += f"|{mline} |      |{cline}|\n"
        return out
