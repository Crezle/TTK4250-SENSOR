from pathlib import Path
import numpy as np
from scipy.io import loadmat
from mytypes import TimeSequence, Measurement2d

from .sample_CT_trajectory import sample_CT_trajectory
from tuning import SimParams

from zlib import crc32
data_path = Path(__file__).parents[2].joinpath("data/data_for_ekf.mat")


def load_data(sim_params: SimParams
              ) -> tuple[TimeSequence[np.ndarray],
                         TimeSequence[Measurement2d]]:

    if seed := sim_params.seed:
        # random seed can be set for repeatability
        np.random.seed(crc32(seed.encode('utf-8')))

    # inital state distribution
    P0 = np.diag([50, 50, 10, 10, np.pi / 4]) ** 2

    # get data
    x_gt_data, z_data = sample_CT_trajectory(sim_params)

    times = np.round(np.arange(sim_params.N_data) * sim_params.dt, 5)

    assert len(times) == len(x_gt_data) == len(z_data)

    ground_truth = TimeSequence(zip(times, x_gt_data))
    measures = TimeSequence(
        zip(times, (Measurement2d(z, sim_params.dt) for z in z_data)))
    return ground_truth, measures
