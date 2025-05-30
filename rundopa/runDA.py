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

import runDA_plotloss

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

# # Create the vectorized environment
# env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
# model = A2C("MlpPolicy", env, verbose=0)

# By default, we use a DummyVecEnv as it is usually faster (cf doc)
env_id = "CartPole-v1"
# env_id = "LunarLander-v3"
num_cpu = 10  # Number of processes to use
vec_env = make_vec_env(env_id, n_envs=num_cpu, monitor_dir="./logs/")
# vec_env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

"""
lr_rlnet   = 5e-6, 7e-4
lr_dopanet = 1e-5, 1e-5
meta_steps =  200,  200

n_meta_steps = 60 (344s)
"""
normalize_values     = False
learning_rate_rl     = 7e-4 # 7e-4
learning_rate_dopa   = 1e-4 # 1e-4
n_meta_steps         = 1
trackers_window_size = 100 * num_cpu
net_arch             = dict(pi=[64, 64], vf=[64, 64], re=[64,64], 
                            v2d=[64], r2d=[64], d2d=[64], da=[64,64], td=[64,64,64,1])        
model = Dopa("MlpPolicy", vec_env, verbose=0, gae_lambda=0.0, n_steps=1, tracker_window_size = trackers_window_size,
             n_meta_steps =n_meta_steps, 
             learning_rate=learning_rate_rl, learning_rate_dopa=learning_rate_dopa,
             net_arch = net_arch,
             normalize_values=normalize_values)

# model = A2C("MlpPolicy", vec_env, verbose=0)

# We create a separate environment for evaluation
eval_env = gym.make(env_id)

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Before training - Mean reward: {mean_reward} +/- {std_reward:.2f}")

n_timesteps =  20 * vec_env.num_envs * 1000

# Multiprocessed RL Training
start_time = time.time()
model.learn(n_timesteps)
total_time_multi = time.time() - start_time

print(
    f"\nTook {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS"
)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"After training - Mean reward: {mean_reward} +/- {std_reward:.2f}")


# runDA_plotloss.plot_loss()

"""
# show the trained agents
vec_env = model.get_env()
obs = vec_env.reset()
idx = 0
for i in range(500):
    print(idx)
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    idx += 1
    time.sleep(0.1)
"""
