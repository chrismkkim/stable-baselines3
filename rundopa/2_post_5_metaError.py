import numpy as np
import results_plotter
import matplotlib.pyplot as plt
import os
from utils import A2CReward
import Plots.compare_a2c_tdnet
import importlib

plt.style.use(os.path.join(os.path.dirname(__file__), 'pyplot_setting.mplstyle'))

# env_id  = 'CartPole-v1'
# env_id = 'LunarLander-v3'
env_id = 'BipedalWalker-v3'

pathfig = '/Users/kimchm/Documents/GitHub/stable-baselines3/rundopa/figure/a2c_vs_tdnet/'
pathsave = '/Users/kimchm/Documents/RL/biowulf/' + env_id + '/saved/' 
pathclass_a2c = 'A2CRewardClass.npy'
pathclass_tdnet = 'TDnetRewardClass.npy'

a2cRew = np.load(pathsave + pathclass_a2c, allow_pickle=True)
tdnetRew = np.load(pathsave + pathclass_tdnet, allow_pickle=True)

# a2c:   nenv x nsims
# tdnet: nenv x nsims x nunit
a2c   = a2cRew.getEnv_with_optimal_lr_and_entropy()
tdnet = tdnetRew.adjust_shape()


a2c, a2c_avg, a2c_sem = a2cRew.get_averages()
tdnet, \
tdnet_avg,     tdnet_sem, \
tdnet_nn_avg,  tdnet_nn_sem, \
tdnet_env_avg, tdnet_env_sem, \
tdnet_best_nn, tdnet_best_avg, tdnet_best_sem = tdnetRew.get_averages()

nenv = 60
nunit = 128

eid = np.where(np.array(tdnetRew.env_list) == nenv)[0][0]
uid = np.where(np.array(tdnetRew.unit_list) == nunit)[0][0]
tdnet_agents = tdnet[eid,:,uid]
sortidx = np.argsort(tdnet_agents)[::-1]



plt.figure()
plt.plot(tdnet_agents, marker='o')
plt.tight_layout()


ntimesteps = 160000
nenv = 60
nsim = 20
metaerr_norm = np.zeros((nsim,ntimesteps,nenv))
metaerr_raw = np.zeros((nsim,ntimesteps,nenv))
dopa    = np.zeros((nsim,ntimesteps,nenv))
adv     = np.zeros((nsim,ntimesteps,nenv))
pathrlerror = '/Users/kimchm/Documents/RL/biowulf/' + env_id + '/rlerror/rl_files'
for sid in range(nsim):
    simid = sortidx[sid]
    pathvalue = pathrlerror + '/' + str(simid) + '/dopa/' + 'rl_values.npy'
    pathdopa  = pathrlerror + '/' + str(simid) + '/dopa/' + 'rl_dopa.npy'
    pathadv   = pathrlerror + '/' + str(simid) + '/dopa/' + 'rl_advantages.npy'
    _value = np.load(pathvalue, allow_pickle=True)
    _dopa = np.load(pathdopa, allow_pickle=True)
    _adv  = np.load(pathadv, allow_pickle=True)    
    metaerr_norm[simid] = (_dopa - _adv)/_adv
    metaerr_raw[simid] = _dopa - _adv
    dopa[simid]    = _dopa
    adv[simid]     = _adv


plt.figure(figsize=(4,1.5))
for sim in range(tdnetRew.nsims):
    plt.plot(tdnetRew.tdnet_reward[sortidx[sim],eid,uid,:])
plt.plot(np.mean(tdnetRew.tdnet_reward[sortidx[:5],eid,uid,:],axis=0), c='k', lw=2)
plt.xlabel('time')
plt.ylabel('reward')
plt.tight_layout()


def mvavg(x, wid):
    xavg = np.zeros_like(x)
    nstep = x.shape[0]
    for i in range(nstep):
        Lidx = np.max([0,i-wid])
        Ridx = np.min([nstep,i+wid])
        xavg[i] = np.mean(x[Lidx:Ridx])
    return xavg

envi = 0
plt.figure(figsize=(6,4))
for sid in range(6):
    plt.subplot(3,2,sid+1)
    metaerr_avg = mvavg(np.abs(metaerr[sid,:,envi]),500)
    plt.plot(np.log10(np.abs(metaerr[sid,:,envi])), marker='.', linestyle='')
    plt.plot(np.log10(metaerr_avg))
    plt.axhline(-2, color='r', linestyle='--')
    plt.title(str(sid) + ': ' + str(tdnet_agents[sortidx[sid]]),fontsize=5)
plt.tight_layout()


plt.figure(figsize=(5,4))
for sid in range(20):
    plt.subplot(5,4,sid+1)
    # metaerr_avg = mvavg(np.abs(metaerr[sid,:,envi]),500)
    # plt.hist(np.log10(np.abs(metaerr[sid,:,envi])), bins=100, range=(-7,2), histtype='step', density=True)
    plt.hist(metaerr_norm[sid,:,envi], bins=100, range=(-0.05,0.05), histtype='step', density=True)
    plt.title(str(sid) + ': ' + str(tdnet_agents[sortidx[sid]]),fontsize=5)
plt.tight_layout()



sid  = 0
envi = 1
plt.figure(figsize=(10,3))
plt.subplot(211)
plt.plot(dopa[sid,:,envi], marker='.', label='dopa')
plt.plot(adv[sid,:,envi], marker='.', label='adv', alpha=0.3)
# plt.plot(value[:,envi], marker='.', label='value')
plt.legend()

plt.subplot(212)
# metaerr_avg = mvavg(np.abs(metaerr_norm[sid,:,envi]),500)
# plt.plot(np.log10(np.abs(metaerr_norm[sid,:,envi])), marker='.', linestyle='')
plt.plot(metaerr_raw[sid,:,envi], marker='.', linestyle='')
# plt.plot(np.log10(metaerr_avg))
plt.tight_layout()
