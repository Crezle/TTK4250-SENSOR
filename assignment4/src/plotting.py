from dataclasses import dataclass, field, fields
from matplotlib import pyplot as plt
import numpy as np

from senfuslib import (TimeSequence, Simulator, ConsistencyAnalysis,
                       plot_field, scatter_field, fill_between_field,
                       ax_config, fig_config, show_consistency)
from states import StateCV, MeasPos
from models import ModelImm
from gaussian_mixture import GaussianMixture


@dataclass
class PlotterIMM:
    ca: ConsistencyAnalysis
    gt: TimeSequence[StateCV]
    meas: TimeSequence[MeasPos]
    ests: TimeSequence[GaussianMixture[StateCV]]
    est_means: TimeSequence[StateCV] = field(init=False)
    sim: Simulator

    def __post_init__(self):
        for field in (f.name for f in fields(self)):
            if isinstance(getattr(self, field, None), TimeSequence):
                setattr(self, field, getattr(self, field)[::0.1])

        self.est_means = TimeSequence(((t, v.reduce().mean)
                                       for (t, v) in self.ests.items()))
        self.mode_colors = ['C0', 'C1', 'tab:pink']

        imm_model: ModelImm = self.sim.dynamic_model
        w1, w2 = [model.rate for model in imm_model.models[1:]]
        self.mode_names = ['CV', f'$CT_{{\omega = {w1}}}$',
                           f'$CT_{{\omega = {w2}}}$']

    def show(self):
        gt_kwr = dict(linestyle='-', alpha=1)
        est_kwr = dict(linestyle='-', alpha=1)
        gt_label = 'gt$_@$'
        est_label = 'est$_@$'
        figsize = (12, 8)

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=figsize)
        scatter_field(ax, self.meas, x='x', y='y', s=1,
                      marker='.', c='k', label='z')
        plot_field(ax, self.gt, fac=self.fac_gt_2d, **gt_kwr)
        ax_config(ax, 'Pos x', 'Pos y', aspect='equal',
                  title='Ground truth position and mode')
        fig_config(fig, 'GT position 2D')

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        scatter_field(ax, self.meas, x='x', y='y', s=1,
                      marker='.', c='k', label='z')
        plot_field(ax, self.ests, fac=self.fac_est_2d, **est_kwr)
        ax_config(ax, 'Pos_x', 'Pos_y', aspect='equal',
                  title='Estimated position and most likely mode')
        fig_config(fig, 'Estimated position 2D')

        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
        ax = axs[0]
        plot_field(ax, self.gt, y=['x', 'y'], label=gt_label, **gt_kwr)
        plot_field(ax, self.est_means, y=['x', 'y'], label=est_label, **gt_kwr)
        ax_config(ax, '', 'Pos', title='Position')
        ax = axs[1]
        plot_field(ax, self.gt, y=['u', 'v'], label=gt_label, **gt_kwr)
        plot_field(ax, self.est_means, y=['u', 'v'], label=est_label, **gt_kwr)
        ax_config(ax, '', 'Vel', title='Velocity')
        ax = axs[2]
        fill_between_field(ax, self.gt, fac=self.fac_gt_modes, **gt_kwr)
        plot_field(ax, self.ests, y=[0, 1, 2],
                   fac=self.fac_est_modes, **est_kwr)
        ax_config(ax, 'Time', 'Model', title='Model', ylim=(0, 1.2))
        fig_config(fig, 'States and modes')

        fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)
        axs_nis = axs[:1]
        axs_nees = axs[1:]
        show_consistency(self.ca, axs_nis, axs_nees,  [['x', 'y']],
                         [['x', 'y', 'u', 'v'], ['x', 'y'], ['u', 'v']])
        fig_config(fig, 'Time plots')

    def fac_gt_2d(self, data: StateCV, **_) -> float:
        kwargs = dict(
            color=self.mode_colors[data.prev_mode],
            label=f'gt s={self.mode_names[data.prev_mode]}')
        data_out = (data.x, data.y)
        return data_out, kwargs

    def fac_est_2d(self, data: GaussianMixture[StateCV], **_):
        argmax = np.argmax(data.weights)
        kwargs = dict(
            color=self.mode_colors[argmax],
            label=f'est argmax_s={self.mode_names[argmax]}')
        mean = data.reduce().mean
        data_out = (mean.x, mean.y)
        return data_out, kwargs

    def fac_gt_modes(self, t: float, data: StateCV, **_) -> float:
        kwargs = dict(
            color=self.mode_colors[data.prev_mode],
            label=f'gt s={self.mode_names[data.prev_mode]}')
        return (t, 1, 1.1), kwargs

    def fac_est_modes(self, t: float, data: GaussianMixture[StateCV], y: int,
                      **_) -> float:
        weight = data.weights[y]
        kwargs = dict(
            color=self.mode_colors[y],
            label=f'est p(s={self.mode_names[y]})')
        return (t, weight), kwargs
