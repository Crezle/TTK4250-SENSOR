from typing import Any, Callable, Optional, Sequence, Union
from matplotlib import pyplot as plt
import numpy as np
from . import TimeSequence, MultiVarGauss, ConsistencyAnalysis, ConsistencyData
import itertools
import matplotlib as mpl

FactoryType = Callable[[float, Any, float, Any, str, str], tuple[tuple, dict]]
t = np.linspace(0, np.pi*2, 100)
circle_points = np.stack((np.cos(t), np.sin(t)), axis=0)


def gauss_points(gauss: MultiVarGauss):
    return gauss.mean[:, None] + gauss.cholesky @ circle_points


def do_field(func: str, ax: plt.Axes, tseq: TimeSequence,
             y: Union[str, Sequence[str]] = None,
             fac: Optional[FactoryType] = None,
             x: str = None,
             **kwargs):
    if y is not None and not isinstance(y, (str, int)):
        for field in y:
            do_field(func, ax=ax, tseq=tseq, y=field, fac=fac, x=x, **kwargs)
        return

    if callable(fac):
        plots = dict()
        prev_list = None
        nextitems = itertools.islice(tseq.items(), 1, None)
        for (t, data), (t_next, data_next) in zip(tseq.items(), nextitems):
            data_out, plot_kwargs = fac(t=t, data=data, x=x, y=y,
                                        t_next=t_next, data_next=data_next)
            current_list = plots.setdefault(tuple(plot_kwargs.items()), [])
            current_list.append(data_out)
            add_nans = ['plot', 'fill_between']
            if (prev_list is not None
                    and func in add_nans
                    and current_list is not prev_list):
                prev_list.append(current_list[-1])
                prev_list.append(np.full_like(prev_list[-1], np.nan))
            prev_list = current_list

    else:
        plots = {tuple(): np.stack([
            tseq.field_as_array(x) if x else tseq.times,
            tseq.field_as_array(y)], axis=1)}

    for kwarg_tuple, data in plots.items():
        plot_kwargs = kwargs.copy()
        plot_kwargs.update(dict(kwarg_tuple))
        if label := plot_kwargs.pop('label', None):
            plot_kwargs['label'] = (label.replace('@y', str(y) or '')
                                    .replace('@x', str(x) or '')
                                    .replace('@', str(y) or ''))

        data = np.array(data).swapaxes(0, 1)
        getattr(ax, func)(*data, **plot_kwargs)


def plot_field(ax: plt.Axes, tseq: TimeSequence,
               y: Union[str, Sequence[str]] = None,
               fac: Optional[FactoryType] = None,
               x: str = None,
               **kwargs):
    do_field('plot', ax, tseq, y, fac, x, **kwargs)


def scatter_field(ax: plt.Axes, tseq: TimeSequence,
                  y: Union[str, Sequence[str]] = None,
                  fac: Optional[FactoryType] = None,
                  x: str = None,
                  **kwargs):
    do_field('scatter', ax, tseq, y, fac, x, **kwargs)


def fill_between_field(ax: plt.Axes, tseq: TimeSequence,
                       y: Union[str, Sequence[str]] = None,
                       fac: Optional[FactoryType] = None,
                       x: str = None,
                       **kwargs):
    assert fac is not None, "fill_between_field requires a factory"
    do_field('fill_between', ax, tseq, y, fac, x, **kwargs)


def show_consistency(analysis: ConsistencyAnalysis,
                     axs_nis: Sequence[plt.Axes], axs_nees: Sequence[plt.Axes],
                     fields_nis: Sequence[str], fields_nees: Sequence[str],
                     **kwargs):

    def add_stuff(ax, data: ConsistencyData):
        ax.set_yscale('log')
        sym = rf"$\chi^2_{data.dof}$"
        labels = [f"{sym} {data.alpha:.0%} interval ({data.in_interval:.1%} inside)",
                  f"{sym} median ({data.above_median:.1%} above)",
                  None]
        colors = ['tab:orange', 'tab:green', 'tab:orange']
        for val, label, color in zip(data.thresholds, labels, colors):
            ax.axhline(val, ls='--', label=label, color=color)
        ax.legend()

    for ax, field in zip(axs_nis, fields_nis):
        data = ConsistencyAnalysis.get_nis(analysis, field)
        ax.plot(data.tseq.times, data.tseq.values, label='NIS')
        ax.set_title(f"NIS for {','.join(field)}")
        add_stuff(ax, data)

    for ax, field in zip(axs_nees, fields_nees):
        data = ConsistencyAnalysis.get_nees(analysis, field)
        ax.plot(data.tseq.times, data.tseq.values, label='NEES')
        ax.set_title(f"NEES for {','.join(field)}")
        add_stuff(ax, data)


def ax_config(ax, x_label=None, y_label=None, title=None, aspect=None,
              legend=True, xlim=None, ylim=None):
    if x_label:
        assert (xlabl := ax.get_xlabel()) == '' or xlabl == x_label
        ax.set_xlabel(x_label)
    if y_label:
        assert (ylabl := ax.get_ylabel()) == '' or ylabl == y_label
        ax.set_ylabel(y_label)
    if aspect:
        ax.set_aspect(aspect)
    if legend:
        ax.legend()
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def fig_config(fig, window_title):
    fig.canvas.set_window_title(window_title)
    fig.set_tight_layout(True)
