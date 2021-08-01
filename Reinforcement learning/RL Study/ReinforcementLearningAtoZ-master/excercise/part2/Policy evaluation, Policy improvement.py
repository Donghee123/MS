import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorized_dp import TensorDP
from utils.grid_visualization import visualize_value_function, visualize_policy
from hdhenvs.gridworld import GridworldEnv
    
np.random.seed(0)

nx = 5
ny = 5
env = GridworldEnv([nx,ny])

dp_agent = TensorDP()
dp_agent.set_env(env)


dp_agent.reset_policy()
info_vi = dp_agent.value_iteration(compute_pi=True)

figsize_mul = 10
steps = info_vi['converge']

fig, ax = plt.subplots(nrows=steps,ncols=2, figsize= (steps * figsize_mul * 0.5, figsize_mul*3))

for i in range(steps):
    visualize_value_function(ax[i][0], info_vi['v'][i], nx,ny)
    visualize_policy(ax[i][1], info_vi['pi'][i], nx,ny)





