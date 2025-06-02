import time

import gymnasium as gym
import numpy as np

from stable_baselines3 import Dopa
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
import runDA_plotloss

dirs = ['/Users/kimchm/Documents/GitHub/stable-baselines3/logs/dopa/LunarLander-v3_1/']

results_plotter.plot_results(dirs, num_timesteps=None, x_axis="timesteps", task_name='LunarLander')

