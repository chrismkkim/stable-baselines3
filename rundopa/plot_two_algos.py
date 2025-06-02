import numpy as np
import results_plotter


# path_to_log_dopa = ['/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/dopa/CartPole-v1_1/']
# path_to_log_a2c = ['/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/a2c/CartPole-v1_1/']
# results_plotter.plot_results(path_to_log_dopa, num_timesteps=None, x_axis="timesteps", task_name='CartPole')
# results_plotter.plot_results(path_to_log_a2c, num_timesteps=None, x_axis="timesteps", task_name='CartPole')

path_to_log_dopa = ['/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/dopa/LunarLander-v3_5/']
path_to_log_a2c = ['/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/a2c/LunarLander-v3_5/']
results_plotter.plot_results(path_to_log_a2c, num_timesteps=None, x_axis="timesteps", task_name='LunarLander', frac="all", figsize=[8,2])
results_plotter.plot_results(path_to_log_dopa, num_timesteps=None, x_axis="timesteps", task_name='LunarLander', frac="half", figsize=[8,2])



x=1

