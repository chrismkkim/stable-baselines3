import time

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def plot_loss():
        
    num_envs = 10
    stepinc = 100

    dirname = "/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/"
    loss_meta = np.loadtxt(dirname + "traindata/loss_meta.txt")
    loss_rl = np.loadtxt(dirname + "traindata/loss_rl.txt")
    adva = np.loadtxt(dirname + "traindata/adva.txt")
    value = np.loadtxt(dirname + "traindata/value.txt")
    next_value = np.loadtxt(dirname + "traindata/next_value.txt")
    dones = np.loadtxt(dirname + "traindata/dones.txt")
    reward = np.loadtxt(dirname + "traindata/reward.txt")
    # time_switch = np.loadtxt(dirname + "traindata/time_switch.txt")
    time_ = np.loadtxt(dirname + "traindata/time.txt")
    
    
    # adva_calc = reward + (1-dones) * next_value - value
    
    
    
    def movavg(x, wid=20):
        nstep = x.shape[0]
        xavg = np.zeros_like(x)
        for i in range(nstep):
            Lidx = np.max([i-wid,0])
            Ridx = i
            if i == 0:
                xavg[i] = x[i]
            else:
                xavg[i] = np.mean(x[Lidx:Ridx])
        return xavg

    loss_meta = np.sqrt(loss_meta)
    loss_meta_avg = movavg(loss_meta)
    loss_rl_avg = movavg(loss_rl)
        
    # time_switch_ = np.copy(time_switch)
    # time_switch_ = np.insert(time_switch_, 0, 0)
    # time_switch_ = np.append(time_switch_, time_[-1])

    # dopa_rate = 1e-5
    # factor = 0.1
    plt.figure(figsize=(10,8))
    ax1 = plt.subplot(311)
    # if isinstance(time_switch, float):
    #     ax1.axvline(time_switch, c='r', linestyle='--')
    # else:
    #     for ti in range(len(time_switch_)):
    #         if (ti % 2 == 0) and (ti + 1 < len(time_switch_)):
    #             ax1.axvspan(time_switch_[ti], time_switch_[ti+1], color='gray', alpha=0.2, ec='None')
    ax1.plot(time_, np.log10(loss_meta), marker='.', linestyle='')
    ax1.axhline(-2, color='b', linestyle='--')
    plt.plot(time_, np.log10(loss_meta_avg))
    ax1.set_ylabel('Meta Loss')

    ax2 = plt.subplot(312, sharex=ax1)
    # if isinstance(time_switch, float):
    #     ax2.axvline(time_switch, c='r', linestyle='--')
    # else:
    #     for ti in range(len(time_switch_)):    
    #         if (ti % 2 == 1) and (ti + 1 < len(time_switch_)):
    #             ax2.axvspan(time_switch_[ti], time_switch_[ti+1], color='gray', alpha=0.2, ec='None')
    ax2.plot(time_, loss_rl, marker='.', linestyle='')
    ax2.plot(time_, loss_rl_avg)
    # ax2.set_ylim([0.6, 1.0])
    ax2.set_ylabel('RL Loss')
    ax2.set_xlabel('epoch')

    ax3 = plt.subplot(313, sharex=ax1)
    ax3.plot(time_, adva, marker='.', linestyle='')
    # ax3.plot(time_, value, marker='.', linestyle='')
    # ax3.plot(time_, next_value, marker='.', linestyle='', color='r')
    # if isinstance(time_switch, float):
    #     ax3.axvline(time_switch, c='r', linestyle='--')
    # else:
    #     for ti in range(len(time_switch_)):    
    #         if (ti % 2 == 1) and (ti + 1 < len(time_switch_)):
    #             ax3.axvspan(time_switch_[ti], time_switch_[ti+1], color='gray', alpha=0.2, ec='None')            
    ax3.set_ylabel('Value / Advantages')
    ax3.set_xlabel('epoch')
    # plt.ylim([-0.2,1.2])
    plt.tight_layout()
    plt.show()
    # plt.savefig(dirname + 'figure/meta_dopa.png', dpi=100)
        
    x=1