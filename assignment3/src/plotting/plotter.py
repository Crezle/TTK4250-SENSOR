from dataclasses import dataclass
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mytypes import TimeSequence, Measurement2d, MultiVarGauss
from analysis import ConsistencyAnalysis
mpl.rcParams['axes.grid'] = True


@dataclass
class Plotter:
    ground_truth: TimeSequence[np.ndarray]
    measurements: TimeSequence[Measurement2d]
    states: TimeSequence[MultiVarGauss]
    state_preds: TimeSequence[MultiVarGauss]
    meas_preds: TimeSequence[MultiVarGauss]
    consistency_analysis: ConsistencyAnalysis

    def __post_init__(self):
        self.gt_arr = self.ground_truth.values_as_array()
        self.pos_est_arr = self.states.field_as_array('mean')
        self.meas_arr = self.measurements.field_as_array('value')

        self.c_meas = 'tab:gray'
        self.c_gt = 'tab:orange'
        self.c_est = 'tab:blue'

    def show(self):
        fig, axs = plt.subplots(4, 1, figsize=(12, 10))
        self.plot_consistency_analysis(axs)
        fig.tight_layout()

        fig, ax = plt.subplots(figsize=(12, 10))
        self.plot_trajectory_2d(ax)
        fig.tight_layout()

        plt.show()

    def plot_trajectory_2d(self, ax):
        ax.scatter(*self.meas_arr.T, 2, self.c_meas, alpha=0.9, label=r"$z$")
        ax.plot(*self.gt_arr.T[:2], c=self.c_gt, label=r"$x_{gt}$")
        ax.plot(*self.pos_est_arr.T[:2], c=self.c_est, label=r"$x_{est}$",
                alpha=0.9)
        ax.set_title("Position")
        ax.set_aspect('equal')
        ax.legend()

    def plot_consistency_analysis(self, axs):
        cons_vals = [
            self.consistency_analysis.nes_all,
            self.consistency_analysis.nes_pos,
            self.consistency_analysis.nes_vel,
            self.consistency_analysis.nis_pos
        ]
        cons_names = ["NEES", "NEES_pos", "NEES_vel", "NIS"]
        for ax, data, name in zip(axs, cons_vals, cons_names):
            times = data.tseq.times
            sym = rf"$\chi^2_{data.df}$"
            ax.plot(times, data.tseq.values, c=self.c_est)
            ax.set_yscale('log')
            ax.set_title(name)
            labels = [f"{sym} {data.alpha:.0%} interval ({data.in_interval:.1%} inside)",
                      f"{sym} median ({data.above_median:.1%} above)",
                      None]
            colors = ['tab:orange', 'tab:green', 'tab:orange']
            for val, label, color in zip(data.thresholds, labels, colors):
                ax.axhline(val, ls='--', label=label, color=color)
                ax.legend()
