from eskf import ESKF
from models import ModelIMU
from sensors import SensorGNSS
from utils.dataloader import load_drone_params
from states import EskfState, NominalState, ErrorState, RotationQuaterion
from senfuslib import MultiVarGauss
import numpy as np
from config import fname_data_real, fname_data_real

accm_corr, gyro_corr, lever_arm = load_drone_params(fname_data_real)

"""Everything below here can be altered"""
start_time_real = 200.  # Start time, set to None for full time
end_time_real = 600  # End time in seconds, set to None to use all data

imu_min_dt_real = None  # IMU is sampled at ~250 Hz, use to downsample
gnss_min_dt_real = None  # GPS is sampled at ~10 Hz, use this to downsample

imu_real = ModelIMU(
    accm_std=1.167e-3,  # Accelerometer standard deviation, TUNABE
    accm_bias_std=4e-3,  # Accelerometer bias standard deviation
    accm_bias_p=1e-16,  # Accelerometer inv time constant see (10.57)

    gyro_std=4.36e-5,  # Gyro standard deviation
    gyro_bias_std=5e-5,  # Gyro bias standard deviation
    gyro_bias_p=1e-16,  # Gyro inv time constant see (10.57)

    accm_correction=accm_corr,  # Accelerometer correction matrix
    gyro_correction=gyro_corr,  # Gyro correction matrix
)

gnss_real = SensorGNSS(
    gnss_std_ne=0.3,  # GNSS standard deviation in North and East
    gnss_std_d=0.5,  # GNSS standard deviation in Down
    lever_arm=lever_arm,  # antenna position relative to origin
)


x_est_init_nom_real = NominalState(
    pos=np.array([1, 0, -1.5]),  # position
    vel=np.array([0, 0, 0]),  # velocity
    ori=RotationQuaterion.from_euler([0, 0, np.deg2rad(90)]),  # orientation
    accm_bias=np.zeros(3),  # accelerometer bias
    gyro_bias=np.zeros(3),  # gyro bias
)

x_err_init_std_real = np.repeat(repeats=3, a=[
    1,  # position
    0.05,  # velocity
    np.deg2rad(10),  # angle vector
    0.01,  # accelerometer bias
    0.001  # gyro bias
])


"""Dont change anything below here"""
x_est_init_err_real = MultiVarGauss[ErrorState](  # Don't change this
    np.zeros(15),  # Don't change this
    np.diag(x_err_init_std_real**2))  # Don't change this


eskf_real = ESKF(imu_real, gnss_real)  # Don't change this
x_est_init_real = EskfState(x_est_init_nom_real, x_est_init_err_real)
