from eskf import ESKF
from models import ModelIMU
from sensors import SensorGNSS
from utils.dataloader import load_drone_params
from states import EskfState, NominalState, ErrorState, RotationQuaterion
from senfuslib import MultiVarGauss
import numpy as np
from config import fname_data_sim, fname_data_real

accm_corr, gyro_corr, lever_arm = load_drone_params(fname_data_sim)

"""Everything below here can be altered"""
start_time_sim = 0.  # Start time, set to None for full time
end_time_sim = 300  # End time in seconds, set to None to use all data

imu_min_dt_sim = None  # IMU is sampled at 100 Hz, use to downsample
gnss_min_dt_sim = None  # GPS is sampled at 1 Hz, use this to downsample

imu_sim = ModelIMU(
    accm_std=1.167e-3,   # Accelerometer standard deviation, TUNABE     (DEFAULT VALUE: 1.167e-3)
    accm_bias_std=4e-3,  # Accelerometer bias standard deviation        (DEFAULT VALUE: 4e-3)
    accm_bias_p=1e-16,  # Accelerometer inv time constant see (10.57)   (DEFAULT VALUE: 1e-16)

    gyro_std=4.36e-5,  # Gyro standard deviation                    (DEFAULT VALUE: 4.36e-5)
    gyro_bias_std=5e-5,  # Gyro bias standard deviation             (DEFAULT VALUE: 5e-5)
    gyro_bias_p=1e-16,  # Gyro inv time constant see (10.57)        (DEFAULT VALUE: 1e-16)

    accm_correction=accm_corr,  # Accelerometer correction matrix   (DEFAULT VALUE: accm_corr)
    gyro_correction=gyro_corr,  # Gyro correction matrix            (DEFAULT VALUE: gyro_corr)
)

gnss_sim = SensorGNSS(
    gnss_std_ne=0.3,  # GNSS standard deviation in North and East   (DEFAULT VALUE: 0.3)
    gnss_std_d=0.5,  # GNSS standard deviation in Down              (DEFAULT VALUE: 0.5)
    lever_arm=lever_arm,  # antenna position relative to origin
)


x_est_init_nom_sim = NominalState(
    pos=np.array([0.2, 0, -5]),  # position                         (DEFAULT VALUE: np.array([0.2, 0, -5]))
    vel=np.array([20, 0, 0]),  # velocity                           (DEFAULT VALUE: np.array([20, 0, 0]))
    ori=RotationQuaterion.from_euler([0, 0, 0]),  # orientation     (DEFAULT VALUE: RotationQuaterion.from_euler([0, 0, 0]))
    accm_bias=np.zeros(3),  # accelerometer bias                    (DEFAULT VALUE: np.zeros(3))
    gyro_bias=np.zeros(3),  # gyro bias                             (DEFAULT VALUE: np.zeros(3))
)

x_err_init_std_sim = np.repeat(repeats=3, a=[
    2,  # position                                                  (DEFAULT VALUE: 2)
    0.1,  # velocity                                                (DEFAULT VALUE: 0.1)
    np.deg2rad(5),  # angle vector                                  (DEFAULT VALUE: np.deg2rad(5))
    0.01,  # accelerometer bias                                     (DEFAULT VALUE: 0.01)
    0.001  # gyro bias                                              (DEFAULT VALUE: 0.001)
])


"""Dont change anything below here"""
x_est_init_err_sim = MultiVarGauss[ErrorState](  # Don't change this
    np.zeros(15),  # Don't change this
    np.diag(x_err_init_std_sim**2))  # Don't change this


eskf_sim = ESKF(imu_sim, gnss_sim)  # Don't change this
x_est_init_sim = EskfState(x_est_init_nom_sim, x_est_init_err_sim)
