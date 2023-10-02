from dataclasses import dataclass
import numpy as np
from typing import Sequence, TypeVar, Generic

from senfuslib import MultiVarGauss
from config import DEBUG
from solution import gaussian_mixture as gaussian_mixture_solu

S = TypeVar('S', bound=np.ndarray)  # State type


@dataclass
class GaussianMixture(Generic[S]):
    weights: np.ndarray
    gaussians: Sequence[MultiVarGauss[S]]

    def __post_init__(self):
        if DEBUG:
            self._debug()

    def mean(self):
        """Find the mean of the gaussian mixture.
        Hint: Use (6.24) from the book."""

        for index, gaussian in enumerate(self.gaussians):
            if index == 0:
                mean = self.weights[index] * gaussian.mean # This is to secure that "mean" posesses same type as the content in "self.gaussians"
            else:
                mean += self.weights[index] * gaussian.mean
            
        return mean

    def cov(self):
        """Find the covariance of the gaussian mixture.
        Hint: Use (6.25) from the book."""

        for index, gaussian in enumerate(self.gaussians):
            if index == 0:
                spread_of_inno = self.weights[index] * gaussian.mean * gaussian.mean.T
            else:
                spread_of_inno += self.weights[index] * gaussian.mean * gaussian.mean.T
            
        spread_of_inno -= self.mean() * self.mean().T

        for index, gaussian in enumerate(self.gaussians):
            if index == 0:
                cov = self.weights[index] * gaussian.cov # This is to secure that "mean" posesses same type as the content in "self.gaussians"
            else:
                cov += self.weights[index] * gaussian.cov

        cov += spread_of_inno
        
        return cov

    def reduce(self) -> MultiVarGauss[S]:
        """Recude the gaussian mixture to a single gaussian."""

        mean = self.mean().view(self.gaussians[0].mean.__class__)
        cov = self.cov()

        gauss = MultiVarGauss(mean, cov)
        return gauss

    def reduce_partial(self, indices: Sequence[int]):
        weights_to_reduce = np.array([self.weights[i] for i in indices])
        gauss_to_reduce = [self.gaussians[i] for i in indices]
        reduced = GaussianMixture(weights_to_reduce/np.sum(weights_to_reduce),
                                  gauss_to_reduce).reduce()

        keep_indices = list(set(range(len(self))) - set(indices))
        weights_to_keep = [self.weights[i] for i in keep_indices]
        gauss_to_keep = [self.gaussians[i] for i in keep_indices]
        out = GaussianMixture(
            np.array([sum(weights_to_reduce), *weights_to_keep]),
            [reduced, *gauss_to_keep])
        return out

    def pdf(self, x):
        return np.sum(self.weights * np.array([g.pdf(x)
                                              for g in self.gaussians]))

    def __len__(self):
        return len(self.gaussians)

    def _debug(self):
        assert self.weights.ndim == 1
        assert self.weights.shape[0] == len(self.gaussians)
        assert np.isclose(np.sum(self.weights), 1)

    def __getitem__(self, idx):
        return GaussianMixture(self.weights[idx], self.gaussians[idx])
