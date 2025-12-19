import time

import gymnasium as gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable


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

env_id = "CartPole-v1"
num_cpu = 10  # Number of processes to use
# # Create the vectorized environment
# env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
# model = A2C("MlpPolicy", env, verbose=0)

# By default, we use a DummyVecEnv as it is usually faster (cf doc)
vec_env = make_vec_env(env_id, n_envs=num_cpu)

"""
learning_rate=7e-4 (default)
learning_rate=5e-6 (does not work)
"""
model = A2C("MlpPolicy", vec_env, verbose=0, gae_lambda=0.0, n_steps=1, learning_rate=7e-4)
# model = A2C("MlpPolicy", vec_env, verbose=0)

# We create a separate environment for evaluation
eval_env = gym.make(env_id)

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Before training - Mean reward: {mean_reward} +/- {std_reward:.2f}")

# Multiprocessed RL Training
n_timesteps =  20 * vec_env.num_envs * 1000
start_time = time.time()
model.learn(n_timesteps)
total_time_multi = time.time() - start_time

print(
    f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS"
)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"After training - Mean reward: {mean_reward} +/- {std_reward:.2f}")

# vec_env = model.get_env()
# obs = vec_env.reset()
# idx = 0
# for i in range(500):
#     print(idx)
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
#     idx += 1
#     time.sleep(0.1)
#     # VecEnv resets automatically
#     # if done:
#     # #   obs = vec_env.reset()
#     #   idx = 0
#     #   time.sleep(0.5)
