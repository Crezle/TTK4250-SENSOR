import numpy as np
from matplotlib import pyplot as plt


from senfuslib import MultiVarGauss
from gaussian_mixture import GaussianMixture


def get_gauss_mixture(weights, means, stds) -> GaussianMixture:
    return GaussianMixture(np.array(weights),
                           [MultiVarGauss(np.array([m]), np.array([[std**2]]))
                            for (m, std) in zip(means, stds)])


def show_gauss_mixture(mix: GaussianMixture, title: str):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].set_title(title)
    times = np.linspace(-5, 10, 100)
    for i, gaus in enumerate(mix.gaussians):
        axs[0].plot(times, [gaus.pdf(t) for t in times],
                    label=f"Gaussian {i}")
    axs[0].legend()

    for i in range(3):
        tojoin = list(set(range(3)) - set([i]))
        joined = mix.reduce_partial(tojoin)
        axs[1].plot(times, [joined.pdf(t) for t in times],
                    label=f'pdf from joining {tojoin[0]} and {tojoin[1]}')

    axs[1].plot(times, [mix.pdf(t) for t in times], label='real pdf')
    axs[1].legend()

    fig.tight_layout()


def task1_show_gaussians():
    a = get_gauss_mixture([1/3, 1/3, 1/3], [0, 2, 4.5], [1, 1, 1])
    b = get_gauss_mixture([1/6, 4/6, 1/6], [0, 2, 4.5], [1, 1, 1])
    c = get_gauss_mixture([1/3, 1/3, 1/3], [0, 2, 4.5], [1, 1.5, 1.5])
    d = get_gauss_mixture([1/3, 1/3, 1/3], [0, 0, 2.5], [1, 1.5, 1.5])

    for i, mix in enumerate([a, b, c, d]):
        show_gauss_mixture(mix, title=f'Task 1, gaussian Mixture {i}')
