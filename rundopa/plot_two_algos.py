import numpy as np
import results_plotter
import matplotlib.pyplot as plt
import os

# path_to_log_dopa = ['/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/dopa/CartPole-v1_1/']
# path_to_log_a2c = ['/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/a2c/CartPole-v1_1/']
# results_plotter.plot_results(path_to_log_dopa, num_timesteps=None, x_axis="timesteps", task_name='CartPole')
# results_plotter.plot_results(path_to_log_a2c, num_timesteps=None, x_axis="timesteps", task_name='CartPole')

# halftime = 4e5
topk    = 25
nagents = 50

env_cart  = 'CartPole-v1'
env_lunar = 'LunarLander-v3'

meta_train_type = 'Cart-Cart'
# meta_train_type = 'Lunar-Lunar'
# meta_train_type = 'Cart-Lunar'
# meta_train_type = 'Lunar-Cart'

if meta_train_type == 'Cart-Cart':
    env_id_meta = env_cart
    env_id_rl   = env_cart
    meta_log    = '2'
elif meta_train_type == 'Lunar-Lunar':
    env_id_meta = env_lunar
    env_id_rl   = env_lunar
    meta_log    = '2'
elif meta_train_type == 'Cart-Lunar':
    env_id_meta = env_cart
    env_id_rl   = env_lunar
    meta_log    = '1'
elif meta_train_type == 'Lunar-Cart':
    env_id_meta = env_lunar
    env_id_rl   = env_cart
    meta_log    = '1'


pathA2C = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/' + env_id_rl + '/'
pathDA = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/'  + env_id_meta  + '_' + env_id_rl + '/'

pathA2c_all = [pathA2C + str(i+1) + '/a2c/'  + env_id_rl + '_' + '1'      + '/' for i in range(nagents)]
pathDA_all  = [pathDA  + str(i+1) + '/dopa/' + env_id_rl + '_' + meta_log + '/' for i in range(nagents)]

rew_a2c               = results_plotter.collect_topk(pathA2c_all, nagents, topk, num_timesteps=None, x_axis="timesteps", frac="all")
rew_dopa              = results_plotter.collect_topk(pathDA_all,  nagents, topk, num_timesteps=None, x_axis="timesteps", frac="all")
# rew_meta_dopa_rl_dopa = results_plotter.collect_topk(pathDA_to_dopa, nagents, topk, num_timesteps=None, x_axis="timesteps", frac="half", halftime=halftime)
# rew_meta_td_rl_dopa   = results_plotter.collect_topk(pathTD_to_dopa, nagents, topk, num_timesteps=None, x_axis="timesteps", frac="half", halftime=halftime)

path = '/Users/kimchm/Documents/GitHub/stable-baselines3/rundopa/figure/'

plt.figure()
plt.plot(np.mean(rew_a2c,axis=0), label='A2C')
plt.plot(np.mean(rew_dopa,axis=0), label='TDnet')
plt.legend()
plt.xlabel('episodes')
plt.ylabel('reward')
plt.title(meta_train_type)
plt.tight_layout()
plt.savefig(path + meta_train_type + '.png', dpi=300)
plt.show()




x=1

