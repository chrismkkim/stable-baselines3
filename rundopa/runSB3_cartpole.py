import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)


# Random Agent, before training
eval_env = gym.make("CartPole-v1")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    # print(i)
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
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
