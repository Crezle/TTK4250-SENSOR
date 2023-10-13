import logging
from matplotlib import pyplot as plt
from plotting import PlotterPDAF
from senfuslib import ConsistencyAnalysis, TimeSequence, GaussianMixture
from tuning import (x0_est,
                    sim_imm, sensor_model_sim,
                    filter_pdaf
                    )


def imm():
    gt, zs = sim_imm.get_gt_and_meas()
    (x_est_upds, x_est_preds, z_est_preds,
     gated_indices_tseq) = filter_pdaf.run(x0_est, zs)

    # ca = None
    # x_est_upds = None

    def reduce(x: GaussianMixture): return x.reduce()

    ca = ConsistencyAnalysis(gt, None, x_est_upds, z_est_preds)

    plotter = PlotterPDAF(ca, gt, zs, x_est_upds, sim_imm, gated_indices_tseq)
    plotter.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s:%(message)s')

    imm()
    plt.show()
