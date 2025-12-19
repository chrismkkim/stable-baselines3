import numpy as np
import results_plotter
import matplotlib.pyplot as plt
import os
from utils import TDnetReward
import Plots.tdnet_reward
import importlib

plt.style.use(os.path.join(os.path.dirname(__file__), 'pyplot_setting.mplstyle'))

env_cart  = 'CartPole-v1'
env_lunar = 'LunarLander-v3'
env_bipedal = 'BipedalWalker-v3'

# env_id = env_cart
# env_id = env_lunar
# env_id, hyperparm = env_bipedal, 'full'
env_id, hyperparm = env_bipedal, 'compact'

if env_id == env_cart:
    env_id_meta = env_cart
    env_id_rl   = env_cart
    meta_log    = '2'
    basetime = 5000
    nreset = 15
elif env_id == env_lunar:
    env_id_meta = env_lunar
    env_id_rl   = env_lunar
    meta_log    = '2'
    basetime = 10000
    nreset = 10    
    sims_list  = np.arange(20)
    env_list   = [10, 20, 30, 40, 50, 60]
    unit_list  = [16, 32, 64, 128, 256, 384, 512]
    env_list_val = ['10', '20', '30', '40', '50', '60']
    unit_list_val = ['16', '32', '64', '128', '256', '384', '512']    
    pathclass = 'TDnetRewardClass.npy'
    pathparam = 'TDnetRewardBestParm.npy'
    
elif env_id == env_bipedal:
    if hyperparm == 'full':
        env_id_meta = env_bipedal
        env_id_rl   = env_bipedal
        meta_log    = '2'
        basetime = 10000
        nreset = 10    
        nsims = 20
        sims_list  = np.arange(nsims)
        env_list   = [10, 20, 30, 40, 50, 60]
        unit_list  = [16, 32, 64, 128, 256, 384]    
        env_list_val = ['10', '20', '30', '40', '50', '60']
        unit_list_val = ['16', '32', '64', '128', '256', '384']
        pathclass = 'TDnetRewardClass.npy'
        pathparam = 'TDnetRewardBestParm.npy'
    if hyperparm == 'compact':
        env_id_meta = env_bipedal
        env_id_rl   = env_bipedal
        meta_log    = '2'
        basetime = 10000
        nreset = 10    
        nsims = 100
        sims_list  = np.arange(nsims)
        env_list   = [10, 20, 30, 40, 50, 60]
        unit_list  = [16, 32, 64, 128]    
        env_list_val = ['10', '20', '30', '40', '50', '60']
        unit_list_val = ['16', '32', '64', '128']
        pathclass = 'TDnetRewardClass-compact.npy'
        pathparam = 'TDnetRewardBestParm-compact.npy'        


nsims    = len(sims_list)
nenv     = len(env_list)
nunit    = len(unit_list)
evalnum  = 20


pathfig = '/Users/kimchm/Documents/GitHub/stable-baselines3/rundopa/figure/'
pathdata = '/Users/kimchm/Documents/RL/biowulf/'
pathsave = '/Users/kimchm/Documents/RL/biowulf/' + env_id + '/saved/'

args = {
    "sims_list": sims_list,
    "env_list": env_list,
    "unit_list": unit_list,
    "evalnum": evalnum,
    "env_id_rl": env_id_rl,
    "pathdata": pathdata,
    "pathsave": pathsave,
    "basetime": basetime,
    "pathclass": pathclass,
    "pathparam": pathparam,
}

tdnetRew = TDnetReward(**args)
tdnetRew.save_data()

plt_tdnet_reward = Plots.tdnet_reward.gen(tdnetRew, pathclass, pathfig, False)

plt_tdnet_reward.env_vs_unit(vmin=0,vmax=100,cmap='jet',env_list_val=env_list_val,unit_list_val=unit_list_val)


plt_tdnet_reward.reward_in_time()
