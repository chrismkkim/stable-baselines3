import numpy as np
import results_plotter
import matplotlib.pyplot as plt
import os
from utils import A2CReward
import Plots.a2c_reward
import importlib

plt.style.use(os.path.join(os.path.dirname(__file__), 'pyplot_setting.mplstyle'))

# env_id = 'CartPole-v1'
# env_id = 'LunarLander-v3'
# env_id, hyperparam = 'BipedalWalker-v3', 'full'
env_id, hyperparam = 'BipedalWalker-v3', 'compact'
path_data = '/Users/kimchm/Documents/RL/biowulf/' + env_id + '/a2c/' 
nsims = 100
sims_list = np.arange(nsims)
if env_id == 'CartPole-v1':
    basetime = 5000
    nreset = 15
elif env_id == 'LunarLander-v3':
    basetime = 10000
    nreset = 10
    envs_list    = [10, 20, 30, 40, 50]
    lr_list      = [0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002]
    ent_list   = ['0.000005', '0.00005', '0.0005', '0.005', '0.05', '0.5', '5.0']    
    lr_list_val = ['8e-4', '10e-4', '12e-4', '14e-4', '16e-4', '18e-4', '20e-4']
    ent_list_val = ['5e-6', '5e-5', '5e-4', '5e-3', '5e-2', '5e-1', '5e0']    
    pathclass = 'A2CRewardClass.npy'
    pathparam = 'A2CRewardBestParm.npy'
    
elif env_id == 'BipedalWalker-v3':
    if hyperparam == 'full':
        basetime = 16000
        nreset = 10
        envs_list    = [10, 20, 30, 40, 50, 60]
        lr_list = [0.00002, 0.00004, 0.00006, 0.00008, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001]
        ent_list = ['0.000005', '0.000015', '0.00005', '0.00015', '0.0005', '0.0015', '0.005']
        lr_list_val = ['2e-5', '4e-5', '6e-5', '8e-5', '1e-4', '2e-4', '4e-4', '6e-4', '8e-4', '1e-3']
        ent_list_val = ['5e-6', '15e-6', '5e-5', '15e-5', '5e-4', '15e-4', '5e-3']
        pathclass = 'A2CRewardClass.npy'
        pathparam = 'A2CRewardBestParm.npy'
    
    if hyperparam == 'compact':
        basetime = 16000
        nreset = 10
        envs_list    = [10, 20, 30, 40, 50, 60]
        lr_list = [0.0006]
        ent_list = ['0.00015']
        lr_list_val = ['6e-4']
        ent_list_val = ['15e-5']
        pathclass = 'A2CRewardClass-compact.npy'
        pathparam = 'A2CRewardBestParm-compact.npy'
    
    # lr_list      = [0.0006, 0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.0018]
    # ent_list   = ['0.00005', '0.00015', '0.0005', '0.0015', '0.005', '0.015', '0.05']
    # lr_list_val = ['6e-4', '8e-4', '10e-4', '12e-4', '14e-4', '16e-4', '18e-4']
    # ent_list_val = ['5e-5', '15e-5', '5e-4', '15e-4', '5e-3', '15e-3', '5e-2']
    

nsims    = len(sims_list)
nenv     = len(envs_list)
nlr      = len(lr_list)
nent     = len(ent_list)
evalnum  = 20


pathfig = '/Users/kimchm/Documents/GitHub/stable-baselines3/rundopa/figure/'
pathdata = '/Users/kimchm/Documents/RL/biowulf/'
pathsave = '/Users/kimchm/Documents/RL/biowulf/' + env_id + '/saved/'

args = {
    "sims_list": sims_list,
    "env_list": envs_list,
    "lr_list": lr_list,
    "ent_list": ent_list,
    "evalnum": evalnum,
    "env_id_rl": env_id,
    "basetime": basetime,
    "pathdata": pathdata,
    "pathsave": pathsave,
    "pathclass": pathclass,
    "pathparam": pathparam,
}

a2cRew = A2CReward(**args)
a2cRew.save_data()

plt_a2c_reward = Plots.a2c_reward.gen(a2cRew, pathclass, pathfig, False)

# vmin, vmax = 50, 150 # lunar
vmin, vmax = -100, 0 # bipedal
plt_a2c_reward.lr_vs_entropy(vmin=vmin,vmax=vmax,cmap='jet',lr_list_val=lr_list_val,ent_list_val=ent_list_val)

plt_a2c_reward.reward_in_time()
