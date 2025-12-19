import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import Dopa, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
from stable_baselines3.common.monitor import Monitor
import runDA_plotloss

# def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
#     def _init() -> gym.Env:
#         env = gym.make(env_id)
#         env.reset(seed=seed + rank)
#         return env
#     set_random_seed(seed)
#     return _init

def main():
    env_id = "CartPole-v1"
    num_cpu = 10  # Number of processes to use

    # Vectorized environment with SubprocVecEnv
    vec_env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    normalize_values     = False
    learning_rate_rl     = 7e-4
    learning_rate_dopa   = 1e-4
    n_meta_steps         = 1
    trackers_window_size = 100 * num_cpu
    net_arch = dict(pi=[64, 64], vf=[64, 64], re=[64,64], 
                    v2d=[64], r2d=[64], d2d=[64], da=[64,64], td=[64,64,64,1])        

    model = Dopa(
        "MlpPolicy", vec_env, verbose=0, gae_lambda=0.0, n_steps=1,
        tracker_window_size=trackers_window_size,
        n_meta_steps=n_meta_steps,
        learning_rate=learning_rate_rl,
        learning_rate_dopa=learning_rate_dopa,
        net_arch=net_arch,
        normalize_values=normalize_values
    )

    eval_env = gym.make(env_id)

    # Evaluate before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Before training - Mean reward: {mean_reward} +/- {std_reward:.2f}")

    n_timesteps = 40 * vec_env.num_envs * 1000

    # Train
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time
    print(f"\nTook {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS")

    # Evaluate after training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"After training - Mean reward: {mean_reward} +/- {std_reward:.2f}")

    # Optionally: plot loss
    # runDA_plotloss.plot_loss()

if __name__ == "__main__":
    main()