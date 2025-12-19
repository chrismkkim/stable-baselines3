import numpy as np
import results_plotter
import matplotlib.pyplot as plt
import os
from utils import TDinterpolate
import Plots.tdnet_reward
import importlib

plt.style.use(os.path.join(os.path.dirname(__file__), 'pyplot_setting.mplstyle'))

env_cart  = 'CartPole-v1'
env_lunar = 'LunarLander-v3'
env_bipedal = 'BipedalWalker-v3'

# env_id = env_cart
# env_id = env_lunar
env_id = env_bipedal

if env_id == env_cart:
    None
elif env_id == env_lunar:
    None
elif env_id == env_bipedal:
    env_id_rl   = env_bipedal
    basetime = 10000
    sims_list  = np.arange(100)
    env_list   = [60]
    unit_list  = [128]    
    # interpolation_constant = 0.25
    # inter_list = np.linspace(0.0, 2.0, num=int(2.0/interpolation_constant + 1))
    inter_list = np.array([0.0,1.0])

nsims    = len(sims_list)
nenv     = len(env_list)
nunit    = len(unit_list)
evalnum  = 20


pathfig = '/Users/kimchm/Documents/GitHub/stable-baselines3/rundopa/figure/'
pathdata = '/Users/kimchm/Documents/RL/biowulf/'
pathsave = '/Users/kimchm/Documents/RL/biowulf/' + env_id + '/saved/'
pathclass = 'TDinterpolateClass.npy'

args = {
    "sims_list": sims_list,
    "env_list": env_list,
    "unit_list": unit_list,
    "inter_list":inter_list,
    "evalnum": evalnum,
    "env_id_rl": env_id_rl,
    "pathdata": pathdata,
    "pathsave": pathsave,
    "basetime": basetime,
    "pathclass": pathclass,
}

tdnetRew = TDinterpolate(**args)
tdnetRew.save_data()

# plt_tdnet_reward = Plots.tdnet_reward.gen(tdnetRew, pathclass, pathfig, False)

# plt_tdnet_reward.env_vs_unit(vmin=0,vmax=100,cmap='jet',env_list_val=env_list_val,unit_list_val=unit_list_val)


# plt_tdnet_reward.reward_in_time()
