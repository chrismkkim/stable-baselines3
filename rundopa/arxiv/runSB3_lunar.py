import gymnasium as gym
import time
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn

env_id = "LunarLander-v3"
num_cpu = 8  # Number of processes to use
vec_env = make_vec_env(env_id, n_envs=num_cpu, monitor_dir="./logs/")
# env = gym.make("LunarLander-v2", render_mode="rgb_array")
linear_schedule = get_linear_fn(start=0.00083, end=0.0, end_fraction=1.0)

start_time = time.time()
model = A2C("MlpPolicy", vec_env, verbose=1, gamma=0.995, n_steps=1, learning_rate=linear_schedule, ent_coef=1e-5)
model.learn(total_timesteps=2e5)
total_time_multi = time.time() - start_time
print(
    f"\nTook {total_time_multi:.2f}s for multiprocessed version"
)


# Random Agent, before training
eval_env = gym.make(env_id)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(500):
#     # print(i)
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
#     # VecEnv resets automatically
#     # if done:
#     #   obs = vec_env.reset()
    
# LunarLander-v2:
#   n_envs: 8
#   n_timesteps: !!float 2e5
#   policy: 'MlpPolicy'
#   gamma: 0.995
#   n_steps: 5
#   learning_rate: lin_0.00083
#   ent_coef: 0.00001

