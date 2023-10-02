

from matplotlib import pyplot as plt
import logging


from mixtures import task1_show_gaussians
from plotting import PlotterIMM
from tuning import (x0_est, filter_imm,
                    sim_imm, sensor_model_sim)
from senfuslib import ConsistencyAnalysis, TimeSequence


def imm():
    gt, zs = sim_imm.get_gt_and_meas()
    x_est_upds, x_est_preds, z_est_preds = filter_imm.run(x0_est, zs)

    x_est_gauss = TimeSequence((t, v.reduce())for t, v in x_est_upds.items())
    z_est_gauss = TimeSequence((t, v.reduce())for t, v in z_est_preds.items())
    ca = ConsistencyAnalysis(gt, zs, x_est_gauss, z_est_gauss)
    plotter = PlotterIMM(ca, gt, zs, x_est_upds, sim_imm)
    plotter.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s:%(message)s')

    task1_show_gaussians()
    imm()
    plt.show()
