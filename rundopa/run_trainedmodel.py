import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import Dopa
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

load_dopa = False
load_a2c  = True
num_env            = 10  # Number of processes to use
env_id_meta        = "LunarLander-v3" #"LunarLander-v3"
env_id_rl          = "LunarLander-v3" #CartPole-v1
eval_env           = gym.make(env_id_rl)

path               = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/trainedmodel/'
path_envs          = env_id_meta + '_' + env_id_rl + '/'
path_to_log        = path + path_envs + '1' + '/'
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
    "net_arch":          {"pi": [64, 64], "vf": [64, 64], "re":[64,64], "td":[64,64,64,1]},
    "normalize_values":  False,
    "tracker_window_size": 1000,
    "train_envs":        {"meta":env_id_meta, "rl":env_id_rl},
    "log_path":          path_to_log,
    "traintype_meta":    False
}
model_path_dopa = '/Users/kimchm/Documents/RL/trainedmodel/LunarLander-v3_LunarLander-v3/dopa/LunarLander-v3_2/best_model.zip'
dopa_env       = make_vec_env(env_id_meta, n_envs=num_env)
dopa_kwargs["env"] = dopa_env
model_dopa_dummy         = Dopa(**dopa_kwargs)
model_dopa = model_dopa_dummy.load(model_path_dopa)
mean_reward_dopa, std_reward_dopa = evaluate_policy(model_dopa, eval_env, n_eval_episodes=50)
print("=="*20)
print("Dopa")
print(f"Mean reward: {mean_reward_dopa} +/- {std_reward_dopa:.2f}")



# model_path_a2c = '/Users/kimchm/Documents/GitHub/rl-baselines3-zoo/logs/a2c/LunarLander-v3_1/LunarLander-v3.zip'
model_path_a2c = '/Users/kimchm/Documents/RL/trainedmodel/LunarLander-v3/a2c/LunarLander-v3_1/best_model.zip'
a2c_env = make_vec_env(env_id_meta, n_envs=num_env)    
model_a2c_dummy = A2C("MlpPolicy",a2c_env)
model_a2c = model_a2c_dummy.load(model_path_a2c)
mean_reward_a2c, std_reward_a2c = evaluate_policy(model_a2c, eval_env, n_eval_episodes=50)
print("=="*20)
print("A2C")
print(f"Mean reward: {mean_reward_a2c} +/- {std_reward_a2c:.2f}")



obs = dopa_env.reset()
for i in range(1000):
    # print(i)
    action, _state = model_dopa.predict(obs, deterministic=True)
    obs, reward, done, info = dopa_env.step(action)
    dopa_env.render("human")
    # # VecEnv resets automatically
    # if done:
    #   obs = dopa_env.reset()
    
    
obs = a2c_env.reset()
for i in range(1000):
    # print(i)
    action, _state = model_a2c.predict(obs, deterministic=True)
    obs, reward, done, info = a2c_env.step(action)
    a2c_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
        
        
        
        
        
# LunarLander-v2:
#   n_envs: 8
#   n_timesteps: !!float 2e5
#   policy: 'MlpPolicy'
#   gamma: 0.995
#   n_steps: 5
#   learning_rate: lin_0.00083
#   ent_coef: 0.00001
