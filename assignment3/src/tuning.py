import numpy as np
from mytypes import MultiVarGauss
from dataclasses import dataclass


@dataclass
class SimParams:
    sigma_a: float = 0.25  # acceleration noise
    sigma_z: float = 3  # measurement noise
    sigma_omega: float = 3.1416 / 20  # rotation noise
    x0 = [0, 0, 1, 1, 0]  # initial state
    P0 = np.diag([50, 50, 10, 10, np.pi / 4]) ** 2  # initial covariance
    N_data: int = 1000  # number of measurements
    dt: float = 0.1  # time step between measurements
    seed: str = 'sensorfusion'  # random seed for data generation


@dataclass
class EKFParams:
    sigma_a: float = 5.  # acceleration noise
    sigma_z: float = 3.1  # measurement noise
    init_state: MultiVarGauss = None  # initial state, None -> initialization
