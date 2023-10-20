from pathlib import Path


data_dir = Path(__file__).parents[1].joinpath('data')
cache_dir = Path(__file__).parents[1].joinpath('cache')
cache_dir.mkdir(parents=True, exist_ok=True)
fname_data_sim = data_dir.joinpath('task_simulation.mat')
fname_data_real = data_dir.joinpath('task_real.mat')


RUN = 'sim'  # 'sim' or 'real', decided which data and tuning parameters to use
DEBUG = True and __debug__  # decedes is sanity checks is performed in classes
PLOT_MIN_DT = 0.1  # Minimum time between points in plot
