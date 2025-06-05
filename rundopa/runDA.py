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
# env = SubprocVecEnv([make_env(env_id, i) for i in range(num_env)])
# model = A2C("MlpPolicy", env, verbose=0)


dopa_kwargs = {
    "policy":            "MlpPolicy",
    "env":               None,
    "verbose":           0,
    "gae_lambda":        0.0,
    "n_steps":           1,
    "n_meta_steps":      1,
    "learning_rate":     7e-4,
    "learning_rate_dopa":1e-4,
    # Dopa‐specific custom args:
    "net_arch":          {"pi": [64, 64], "vf": [64, 64], "re":[64,64], "v2d":[64], "r2d":[64], "d2d":[64], "da":[64,64], "td":[64,64,64,1]},
    "normalize_values":  False,
    "tracker_window_size": 1000,
}
save_path          = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/trainedmodel/'
num_env            = 10  # Number of processes to use
n_timesteps        = 10 * num_env * 1000
env_id_meta        = "CartPole-v1" #"LunarLander-v3"
env_id_rl          = "LunarLander-v3"

#----------------------#
#    Meta learning
#----------------------#
# env for meta learning
vec_env_meta       = make_vec_env(env_id_meta, n_envs=num_env)
dopa_kwargs["env"] = vec_env_meta
traintype_meta     = True
meta_model         = Dopa(traintype_meta=traintype_meta, **dopa_kwargs)

#--- Before learning ---#
# We create a separate environment for evaluation
eval_env = gym.make(env_id_meta)
mean_reward, std_reward = evaluate_policy(meta_model, eval_env, n_eval_episodes=10)
print(f"Before training - Mean reward: {mean_reward} +/- {std_reward:.2f}")

#--- Meta learning ---#
start_time = time.time()
meta_model.learn(n_timesteps)
total_time_multi = time.time() - start_time
print(f"\nTook {total_time_multi:.2f}s for meta learning")
# save trained model
meta_model.save(save_path + 'dopa')


#----------------------#
#    Reinforcement learning
#----------------------#
# env for RL
vec_env_rl         = make_vec_env(env_id_rl, n_envs=num_env)
dopa_kwargs["env"] = vec_env_rl
traintype_meta     = False
rl_model           = Dopa(traintype_meta=traintype_meta, **dopa_kwargs)
# Use the trained TD network
meta_model_sd = meta_model.policy.mlp_extractor.state_dict()
rl_model_sd   = rl_model.policy.mlp_extractor.state_dict()
for layer in range(len(dopa_kwargs["net_arch"]["td"])):
    rl_model_sd['td_net.'+str(2*layer)+'.weight'] = meta_model_sd['td_net.'+str(2*layer)+'.weight'].clone()
    rl_model_sd['td_net.'+str(2*layer)+'.bias']   = meta_model_sd['td_net.'+str(2*layer)+'.bias'].clone()    
rl_model.policy.mlp_extractor.load_state_dict(rl_model_sd)

#--- Reinforcement learning ---#
start_time = time.time()
rl_model.learn(n_timesteps)
total_time_multi = time.time() - start_time
print(f"\nTook {total_time_multi:.2f}s for reinforcement learning")

#--- After learning ---#
eval_env = gym.make(env_id_rl)
mean_reward, std_reward = evaluate_policy(rl_model, eval_env, n_eval_episodes=100)
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
