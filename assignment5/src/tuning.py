import numpy as np
from senfuslib import MultiVarGauss, Simulator, GaussianMixture, TimeSequence
from copy import deepcopy

from models import ModelCT, ModelCV, ModelImm
from states import StateCV
from sensors import SensorPosClutter, SensorPos
from filter import FilterPDA


def sensor_setter(sensor: SensorPosClutter, gts: TimeSequence[StateCV]):

    x_min, x_max = gts.get_min_max(lambda state: state.x)
    x_diff = x_max - x_min
    sensor.x_min = x_min - x_diff*0.1
    sensor.x_max = x_max + x_diff*0.1

    y_min, y_max = gts.get_min_max(lambda state: state.y)
    y_diff = y_max - y_min
    sensor.y_min = y_min - y_diff*0.1
    sensor.y_max = y_max + y_diff*0.1


"""Simulation parameters"""
x0_sim = MultiVarGauss[StateCV](StateCV(0, 0, 6, 0),
                                np.diag([0.1, 0.1, 0.1, 0.1]))

dynamic_model_sim = ModelImm(
    models=[ModelCV(std_vel=1e-1),
            ModelCT(std_vel=1e-1, rate=0.3),
            ModelCT(std_vel=1e-1, rate=-0.5)],
    hold_times=np.array([10, 6, 6]),
    jump_mat=np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]))

sensor_model_sim = SensorPosClutter(sensor=SensorPos(std_pos=1.1),
                                    prob_detect=0.7,
                                    clutter_density=0.0001)

sim_imm = Simulator(dynamic_model=dynamic_model_sim,
                    sensor_model=sensor_model_sim,
                    sensor_setter=sensor_setter,
                    init_state=MultiVarGauss(
                        x0_sim.mean.with_new_meta(prev_mode=0),
                        x0_sim.cov),
                    dt=0.1,
                    end_time=80,
                    seed=None
                    )


"""Filter parameters"""
x0_est = deepcopy(x0_sim)
sensor_model_filter = deepcopy(sensor_model_sim)
dynamic_model_filter = ModelCV(std_vel=1.8) # Originally 0.8

filter_pdaf = FilterPDA(
    dynamic_model=dynamic_model_filter,
    sensor_model=sensor_model_filter,
    gate_prob=0.95) # Originally 0.99
