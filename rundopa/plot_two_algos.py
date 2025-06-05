import numpy as np
import results_plotter
import matplotlib.pyplot as plt

# path_to_log_dopa = ['/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/dopa/CartPole-v1_1/']
# path_to_log_a2c = ['/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/a2c/CartPole-v1_1/']
# results_plotter.plot_results(path_to_log_dopa, num_timesteps=None, x_axis="timesteps", task_name='CartPole')
# results_plotter.plot_results(path_to_log_a2c, num_timesteps=None, x_axis="timesteps", task_name='CartPole')

halftime = 4e5
topk    = 10
nagents = 50
env_id = 'LunarLander-v3_1'
pathDA = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/dopa/'
pathTD = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/td/'
pathDA_to_a2c  = [pathDA + str(i+1) + '/a2c/' + env_id + '/' for i in range(nagents)]
pathDA_to_dopa = [pathDA + str(i+1) + '/dopa/' + env_id + '/' for i in range(nagents)]
pathTD_to_dopa = [pathTD + str(i+1) + '/dopa/' + env_id + '/' for i in range(nagents)]

rew_a2c               = results_plotter.collect_topk(pathDA_to_a2c, nagents, topk, num_timesteps=None, x_axis="timesteps", frac="all")
rew_meta_dopa_rl_dopa = results_plotter.collect_topk(pathDA_to_dopa, nagents, topk, num_timesteps=None, x_axis="timesteps", frac="half", halftime=halftime)
rew_meta_td_rl_dopa   = results_plotter.collect_topk(pathTD_to_dopa, nagents, topk, num_timesteps=None, x_axis="timesteps", frac="half", halftime=halftime)

plt.figure()
plt.plot(np.mean(rew_a2c,axis=0), label='A2C')
plt.plot(np.mean(rew_meta_dopa_rl_dopa,axis=0), label='DA_DA')
plt.plot(np.mean(rew_meta_td_rl_dopa,axis=0), label='DA_TD')
plt.legend()
plt.xlabel('episodes')
plt.ylabel('reward')
plt.tight_layout()
plt.show()




x=1

