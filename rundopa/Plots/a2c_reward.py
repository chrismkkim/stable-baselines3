import matplotlib.pyplot as plt
import numpy as np
from utils import A2CReward
import os

plt.style.use(os.path.join(os.path.dirname(__file__), 'pyplot_setting.mplstyle'))


class gen:
    
    def __init__(
        self,
        data: A2CReward,
        pathdata: str,
        pathfig: str,
        load_data: bool,
        ):
        
        self.pathdata = pathdata
        self.pathfig = pathfig
        self.load_data = load_data
        
        if self.load_data:
            self.load_a2cReward_class()
        else:
            self.a2cRew = data
        
    def load_a2cReward_class(self):
        self.a2cRew = np.load(self.pathdata, allow_pickle=True)                                    
                        
    def lr_vs_entropy(self,vmin,vmax,cmap,lr_list_val,ent_list_val):
        for eid in range(self.a2cRew.nenv):
            plt.figure(figsize=(3,2.5))
            # plt.title('Fixed LR, ' + 'nenv ' + str(self.a2cRew.env_list[eid]))
            lrmode = 0
            plt.imshow(self.a2cRew.max_reward[eid,:,:], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='lower', interpolation='gaussian')
            for i in range(self.a2cRew.nent):
                for j in range(self.a2cRew.nlr):
                    plt.annotate(str(np.floor(self.a2cRew.max_reward[eid,j,i]).astype(int)),xy=(i-0.25,j-0.15),color='white', fontsize=6)
            plt.xlabel('Entropy')
            plt.ylabel('Learning rate')
            plt.xticks(ticks=np.arange(len(ent_list_val)), labels=ent_list_val, rotation=45)
            plt.yticks(ticks=np.arange(len(lr_list_val)), labels=lr_list_val)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(self.pathfig + 'a2c_reward_nenv_' + str(self.a2cRew.env_list[eid]) + '_fixedlr.pdf')

    def reward_in_time(self):
        tinc = self.a2cRew.basetime * self.a2cRew.nenv * 10 / self.a2cRew.evalnum
        tvec = np.arange(self.a2cRew.evalnum) * tinc
         
        eid = self.a2cRew.nenv-1
        lr_ix = self.a2cRew.topk_param_idx[0][-1]
        ent_ix = self.a2cRew.topk_param_idx[1][-1]
        plt.figure(figsize=(2.0,1.3))
        for sid in np.arange(5,self.a2cRew.nsims):
            plt.plot(tvec[1:], self.a2cRew.a2c_reward[sid,eid,lr_ix,ent_ix,1:], marker='o', lw=0.5, ms=1.5, mec='None')
        plt.plot(tvec[1:], np.mean(self.a2cRew.a2c_reward[:,eid,lr_ix,ent_ix,1:],axis=0), color='k',lw=1.5)    
        plt.xlabel('time step')
        plt.ylabel('mean reward')
        plt.tight_layout()
        plt.savefig(self.pathfig + 'a2c_reward_in_time.pdf')
        
        