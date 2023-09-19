from dataclasses import dataclass, field
import numpy as np
from config import DEBUG


@dataclass(frozen=True)
class MultiVarGauss:
    """A class for using Gaussians"""
    mean: np.ndarray
    cov: np.ndarray

    def __post_init__(self):
        """Makes the class immutable"""
        if self.mean is not None and self.cov is not None:
            self.mean.setflags(write=False)
            self.cov.setflags(write=False)
            if DEBUG:
                self._debug()

    @property
    def ndim(self) -> int:
        return self.mean.shape[0]

    def mahalanobis_distance(self, x: np.ndarray) -> float:
        """Calculate the mahalanobis distance between self and x.

        This is also known as the quadratic form of the Gaussian.
        See (3.2) in the book.
        """
        err = x.reshape(-1, 1) - self.mean.reshape(-1, 1)
        mahalanobis_distance = float(err.T @ np.linalg.solve(self.cov, err))
        return mahalanobis_distance

    def get_marginalized(self, intices):
        i_idx, j_idx = np.meshgrid(intices, intices,
                                   sparse=True, indexing='ij')
        mean = self.mean[i_idx.ravel()]
        cov = self.cov[i_idx, j_idx]
        return MultiVarGauss(mean, cov)

    def _debug(self):
        assert self.mean.ndim == 1
        assert self.cov.ndim == 2
        assert self.mean.shape[0] == self.cov.shape[0] == self.cov.shape[1]
        assert np.all(np.isfinite(self.mean))
        assert np.all(np.isfinite(self.cov))
        assert np.allclose(self.cov, self.cov.T)
        assert np.all(np.linalg.eigvals(self.cov) >= 0)

    def __iter__(self):
        """Enable iteration over the mean and covariance.
        i.e.
            est = MultiVarGauss2d(mean=[1, 2], cov=[[1, 0], [0, 1]])
            mean, cov = est
        """
        return iter((self.mean, self.cov))

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
