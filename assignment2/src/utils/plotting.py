import numpy as np
from typing import Sequence
from gaussian import MultiVarGauss2d
from matplotlib import pyplot as plt


t = np.linspace(0, np.pi*2, 100)
circle_points = np.array([np.cos(t), np.sin(t)])


def ellipse_points(gauss=MultiVarGauss2d,
                   ) -> tuple[Sequence[float], Sequence[float]]:
    """Get the points belonging to the 1 sigma ellipsis"""
    mean, cov = gauss
    points = np.linalg.cholesky(cov) @ circle_points + mean[:, None]
    return points[0, :], points[1, :]


def plot_gauss(ax: plt.Axes, gauss: MultiVarGauss2d,
               name: str, color: str):
    """Plot a gaussian on an axis"""
    x, y = ellipse_points(gauss)
    ax.plot(x, y, '-', color=color, label=name)
    ax.plot(gauss.mean[0], gauss.mean[1], 'x', color=color)


def show_results(state_est: MultiVarGauss2d,
                 meas_gauss_c: MultiVarGauss2d, meas_gauss_r: MultiVarGauss2d,
                 cond_c: MultiVarGauss2d, cond_r: MultiVarGauss2d,
                 cond_cr: MultiVarGauss2d):
    """Plot the results"""
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

    for name, gauss, color in [('state_est', state_est, 'C0'),
                               ('meas_c', meas_gauss_c, 'C1'),
                               ('cond_c', cond_c, 'C2')]:
        plot_gauss(axs[0, 0], gauss, name, color)

    for name, gauss, color in [('state_est', state_est, 'C0'),
                               ('meas_r', meas_gauss_r, 'C3'),
                               ('cond_r', cond_r, 'C4')]:
        plot_gauss(axs[0, 1], gauss, name, color)

    for name, gauss, color in [('cond_c', cond_c, 'C2'),
                               ('cond_r', cond_r, 'C4'),
                               ('cond_cr', cond_cr, 'C5')]:
        plot_gauss(axs[1, 0], gauss, name, color)

    plot_gauss(axs[1, 1], cond_cr, 'cond_cr', 'C5')
    axs[1, 1].axline((0, 5), slope=1, label='$x_2 = x_1 + 5$')
    for ax in axs.flatten():
        ax.legend()
        ax.set_aspect('equal')
    fig.tight_layout()
    plt.show()
