from pathlib import Path

DEBUG = True and __debug__  # set to False for speedup

sim_output_dir = Path(__file__).parents[1] / 'data' / 'cache'
sim_output_dir.mkdir(exist_ok=True, parents=True)
