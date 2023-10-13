from dataclasses import dataclass, field, fields
from typing import Sequence
from matplotlib import pyplot as plt
import numpy as np

from senfuslib import (TimeSequence, Simulator, ConsistencyAnalysis,
                       GaussianMixture, MultiVarGauss, NamedArray,
                       plot_field, scatter_field, fill_between_field,
                       ax_config, fig_config, show_consistency)
from states import StateCV, MeasPos
from models import ModelImm


@dataclass
class PlotterPDAF:
    ca: ConsistencyAnalysis
    gt: TimeSequence[StateCV]
    meas: TimeSequence[MeasPos]
    ests: TimeSequence[GaussianMixture[StateCV]]
    sim: Simulator
    gated_indices: TimeSequence[set[int]]

    def __post_init__(self):
        for f in (f.name for f in fields(self)):
            if isinstance(getattr(self, f, None), TimeSequence):
                setattr(self, f, getattr(self, f)[::0.1])

        self.mode_colors = ['C0', 'C1', 'tab:pink']
        self.est_means = self.ests.map(lambda x: x.mean)

        imm_model: ModelImm = self.sim.dynamic_model
        w1, w2 = [model.rate for model in imm_model.models[1:]]
        self.mode_names = ['CV', f'$CT_{{\omega = {w1}}}$',
                           f'$CT_{{\omega = {w2}}}$']

    def show(self):
        gt_label = 'gt$_@$'
        est_label = 'est$_@$'
        gt_kwr = dict(linestyle=(0, (5, 1)), alpha=0.7, label=gt_label)
        est_kwr = dict(linestyle='-', alpha=0.7, label=est_label)

        clutter_kwr = dict(s=2, marker='.', color='gray', alpha=0.5)
        figsize = (12, 8)

        """GT"""
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=figsize)

        scatter_field(ax, self.meas, fac=self.fac_gt_clutter_2d,
                      **clutter_kwr, label='clutter')
        scatter_field(ax, self.meas, fac=self.fac_gt_meas_2d,
                      s=8, marker='.', c='g', label='z', alpha=0.5)
        plot_field(ax, self.gt, fac=self.fac_gt_2d, **gt_kwr)

        ax_config(ax, 'Pos x', 'Pos y', aspect='equal',
                  title='Ground truth position and mode')
        fig_config(fig, 'GT position 2D')

        """Estimated 2D"""
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        scatter_field(ax, self.meas, fac=self.fac_est_meas_2d_tp,
                      s=8, marker='.', c='g', label='used meas')
        scatter_field(ax, self.meas, fac=self.fac_est_meas_2d_tn,
                      **clutter_kwr,
                      label='ignored clutter')
        scatter_field(ax, self.meas, fac=self.fac_est_meas_2d_fn,
                      s=8, marker='.', c='red', label='ignored meas')
        scatter_field(ax, self.meas, fac=self.fac_est_meas_2d_fp,
                      s=12, marker='.', c='purple', label='used clutter')

        plot_field(ax, self.est_means, x='x', y='y', **est_kwr)
        ax_config(ax, 'Pos_x', 'Pos_y', aspect='equal',
                  title='Estimated position')
        fig_config(fig, 'Estimated position 2D')

        """States and modes"""
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
        ax = axs[0]
        plot_field(ax, self.est_means, y=['x', 'y'], **est_kwr)
        plot_field(ax, self.gt, y=['x', 'y'], **gt_kwr)
        ax_config(ax, '', 'Pos', title='Position')
        ax = axs[1]
        plot_field(ax, self.est_means, y=['u', 'v'], **est_kwr)
        plot_field(ax, self.gt, y=['u', 'v'], **gt_kwr)
        ax_config(ax, '', 'Vel', title='Velocity')

        fig_config(fig, 'States and modes')

        # """Consistency"""
        fig_cons, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
        axs_nis = []
        axs_nees = axs[:]
        fig_err, axs_err = plt.subplots(4, 1, figsize=figsize, sharex=True)
        show_consistency(self.ca,
                         axs_nis, [],
                         axs_nees, ['xyuv', 'xy', 'uv'],
                         axs_err, 'xyuv')
        fig_config(fig_cons, 'Consistency plots')
        fig_config(fig_err, 'Error plots')

    def fac_gt_2d(self, data: StateCV, **_) -> float:
        kwargs = dict(
            color=self.mode_colors[data.prev_mode],
            label=f'gt s={self.mode_names[data.prev_mode]}')
        data_out = (data.x, data.y)
        return data_out, kwargs

    def fac_gt_clutter_2d(self, data: Sequence[MeasPos], **_):
        data_out = [(d.x, d.y)for d in data if d.isclutter]
        return data_out, dict()

    def fac_gt_meas_2d(self, data: Sequence[MeasPos], **_):
        data_out = [(d.x, d.y)for d in data if not d.isclutter]
        return data_out, dict()

    def fac_est_meas_2d_tp(self, t: float, data: Sequence[MeasPos], **_):
        data_out = [(d.x, d.y) for (i, d) in enumerate(data)
                    if not d.isclutter and i in self.gated_indices[t]]
        return data_out, dict()

    def fac_est_meas_2d_tn(self, t: float, data: Sequence[MeasPos], **_):
        data_out = [(d.x, d.y) for (i, d) in enumerate(data)
                    if d.isclutter and i not in self.gated_indices[t]]
        return data_out, dict()

    def fac_est_meas_2d_fp(self, t: float, data: Sequence[MeasPos], **_):
        data_out = [(d.x, d.y) for (i, d) in enumerate(data)
                    if d.isclutter and i in self.gated_indices[t]]
        return data_out, dict()

    def fac_est_meas_2d_fn(self, t: float, data: Sequence[MeasPos], **_):
        data_out = [(d.x, d.y) for (i, d) in enumerate(data)
                    if not d.isclutter and i not in self.gated_indices[t]]
        return data_out, dict()

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
