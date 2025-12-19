import matplotlib.pyplot as plt
import numpy as np
from utils import A2CReward
import os

plt.style.use(os.path.join(os.path.dirname(__file__), 'pyplot_setting.mplstyle'))


class gen:
    
    def __init__(
        self,
        pathfig: str,
        env_id: str, 
        a2cRew,
        tdnetRew,
        a2c: np.array, 
        a2c_avg: np.array, 
        a2c_sem: np.array,
        tdnet: np.array,
        tdnet_avg: np.array, 
        tdnet_sem: np.array,
        tdnet_nn_avg: np.array, 
        tdnet_nn_sem: np.array,
        tdnet_env_avg: np.array, 
        tdnet_env_sem: np.array,
        tdnet_best_nn: np.array, 
        tdnet_best_avg: np.array, 
        tdnet_best_sem: np.array):
        
        self.pathfig = pathfig
        self.env_id = env_id
        self.a2cRew = a2cRew
        self.tdnetRew = tdnetRew
        self.a2c = a2c
        self.a2c_avg = a2c_avg
        self.a2c_sem = a2c_sem
        self.tdnet = tdnet
        self.tdnet_avg = tdnet_avg
        self.tdnet_sem = tdnet_sem
        self.tdnet_nn_avg = tdnet_nn_avg
        self.tdnet_nn_sem = tdnet_nn_sem
        self.tdnet_env_avg = tdnet_env_avg
        self.tdnet_env_sem = tdnet_env_sem
        self.tdnet_best_nn = tdnet_best_nn
        self.tdnet_best_avg = tdnet_best_avg
        self.tdnet_best_sem = tdnet_best_sem
                
                
        if self.env_id == "LunarLander-v3":
            env_rng      = self.a2cRew.nenv
            ylim         = [80,200]
            reward_range = (-50,300)
        elif self.env_id == "BipedalWalker-v3":
            env_rng      = self.tdnetRew.nenv
            ylim         = [-100,150]
            reward_range = (-150,300)
        self.env_rng      = env_rng
        self.ylim         = ylim
        self.reward_range = reward_range

    def compare_all(self):    

        plt.figure(figsize=(6,1.3))
        for envi in range(self.a2cRew.nenv):
            plt.subplot(1,self.a2cRew.nenv,envi+1)
            plt.fill_between(np.arange(self.tdnetRew.nunit), self.tdnet_avg[envi] - self.tdnet_sem[envi], self.tdnet_avg[envi] + self.tdnet_sem[envi], fc='C0', ec='None', alpha=0.3) 
            plt.plot(np.arange(self.tdnetRew.nunit), self.tdnet_avg[envi], marker='o', c='C0', label='TDnet')
            plt.fill_between(np.arange(self.tdnetRew.nunit), self.a2c_avg[envi] - self.a2c_sem[envi], self.a2c_avg[envi] + self.a2c_sem[envi], fc='r', ec='None', alpha=0.3) 
            plt.axhline(self.a2c_avg[envi], color='r', linestyle='--', label='A2C')
            plt.xticks(np.arange(self.tdnetRew.nunit), self.tdnetRew.unit_list, rotation=45, fontsize=5)
            plt.yticks(fontsize=5)
            plt.title('nenv ' + str(self.a2cRew.env_list[envi]), fontsize=5)
            plt.ylim(self.ylim)
            if envi == 0:
                plt.ylabel('reward', fontsize=5)
                plt.legend(frameon=False, fontsize=5)
            plt.xlabel('num unit', fontsize=5)
        plt.tight_layout()
        # plt.savefig(self.pathfig + self.env_id + '_compare_all.pdf')        
        
        
    def compare_env(self):
        xrg = self.env_rng
        plt.figure(figsize=(2.3,1.5))
        # plt.fill_between(a2cRew.env_list, tdnet_best_avg - tdnet_best_sem, tdnet_best_avg + tdnet_best_sem, fc='C1', ec='None', alpha=0.3) 
        # plt.plot(a2cRew.env_list, tdnet_best_avg, c='C1', label='TDnet Best', marker='o')
        plt.fill_between(self.tdnetRew.env_list[:xrg], self.tdnet_env_avg[:xrg] - self.tdnet_env_sem[:xrg], self.tdnet_env_avg[:xrg] + self.tdnet_env_sem[:xrg], fc='C0', ec='None', alpha=0.3) 
        plt.plot(self.tdnetRew.env_list[:xrg], self.tdnet_env_avg[:xrg], c='C0', label='TDnet', marker='o')
        plt.fill_between(self.a2cRew.env_list, self.a2c_avg - self.a2c_sem, self.a2c_avg + self.a2c_sem, fc='r', ec='None', alpha=0.3) 
        plt.plot(self.a2cRew.env_list, self.a2c_avg, c='r', label='A2C', marker='o')
        plt.legend(frameon=False, bbox_to_anchor=[1,1])
        plt.xticks(self.a2cRew.env_list)
        plt.xlabel('num env')
        plt.ylabel('reward')
        plt.tight_layout()
        # plt.savefig(self.pathfig + self.env_id + '_compare_env.pdf')

    def compare_hist(self):

        plt.figure(figsize=(2.3,1.5))
        plt.hist(self.tdnet.flatten(), bins=20, range=self.reward_range, histtype='step', density=True, color='C0', label='TDnet')
        plt.hist(self.a2c.flatten(), bins=20, range=self.reward_range, histtype='bar', density=True, color='r', label='A2C', alpha=0.5)
        plt.xlabel('reward')
        plt.ylabel('agent density')
        plt.legend(frameon=False, bbox_to_anchor=[1,0.5])
        plt.tight_layout()
        # plt.savefig(self.pathfig + self.env_id + '_compare_hist.pdf')

        
        