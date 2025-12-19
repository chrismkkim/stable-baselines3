import matplotlib.pyplot as plt
import numpy as np
from utils import TDnetReward
import os

plt.style.use(os.path.join(os.path.dirname(__file__), 'pyplot_setting.mplstyle'))


class gen:
    
    def __init__(
        self,
        data: TDnetReward,
        pathdata: str,
        pathfig: str,
        load_data: bool,
        ):
        
        self.pathdata = pathdata
        self.pathfig = pathfig
        self.load_data = load_data
        
        if self.load_data:
            self.load_tdnetReward_class()
        else:
            self.tdnetRew = data
        
    def load_tdnetReward_class(self):
        self.tdnetRew = np.load(self.pathdata, allow_pickle=True)                                    
                        
    def env_vs_unit(self,vmin,vmax,cmap,env_list_val,unit_list_val):
        plt.figure(figsize=(3,2.5))
        plt.imshow(self.tdnetRew.max_reward, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='lower', interpolation='gaussian')
        for j in range(self.tdnetRew.nenv):
            for i in range(self.tdnetRew.nunit):
                plt.annotate(str(np.floor(self.tdnetRew.max_reward[j,i]).astype(int)),xy=(i-0.25,j-0.15),color='white', fontsize=6)
        plt.xlabel('Network unit')
        plt.ylabel('Environment')
        plt.xticks(ticks=np.arange(len(unit_list_val)), labels=unit_list_val, rotation=45)
        plt.yticks(ticks=np.arange(len(env_list_val)), labels=env_list_val)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(self.pathfig + 'tdnet_reward.pdf')

    def reward_in_time(self):
        tinc = self.tdnetRew.basetime * self.tdnetRew.nenv * 10 / self.tdnetRew.evalnum
        tvec = np.arange(self.tdnetRew.evalnum) * tinc
         
        stepone = 2
        topk = -2
        env_ix = self.tdnetRew.topk_param_idx[0][topk]
        unit_ix = self.tdnetRew.topk_param_idx[1][topk]
        plt.figure(figsize=(2.0,1.3))
        for sid in np.arange(5,self.tdnetRew.nsims):
            plt.plot(tvec[stepone:], self.tdnetRew.tdnet_reward[sid,env_ix,unit_ix,stepone:], marker='o', lw=0.5, ms=1.5, mec='None')
        plt.plot(tvec[stepone:], np.mean(self.tdnetRew.tdnet_reward[:,env_ix,unit_ix,stepone:],axis=0), color='k',lw=1.5)    
        plt.xlabel('time step')
        plt.ylabel('mean reward')
        plt.tight_layout()
        plt.savefig(self.pathfig + 'tdnet_reward_in_time.pdf')
        
        