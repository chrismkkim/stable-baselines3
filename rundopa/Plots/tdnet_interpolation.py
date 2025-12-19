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
        tdnetInterRew,
        a2c: np.array, 
        a2c_avg: np.array, 
        a2c_sem: np.array,
        tdnet: np.array,
        tdnet_avg: np.array, 
        tdnet_sem: np.array,
        tdinter: np.array,
        tdinter_avg: np.array, 
        tdinter_sem: np.array        
        ):
        
        self.pathfig = pathfig
        self.env_id = env_id
        self.a2cRew = a2cRew
        self.tdnetRew = tdnetRew
        self.tdnetInterRew = tdnetInterRew
        self.a2c = a2c
        self.a2c_avg = a2c_avg
        self.a2c_sem = a2c_sem
        self.tdnet = tdnet
        self.tdnet_avg = tdnet_avg
        self.tdnet_sem = tdnet_sem
        self.tdinter = tdinter
        self.tdinter_avg = tdinter_avg
        self.tdinter_sem = tdinter_sem                                
        self.select_tdnet_with_nenv60_nunit128()
        
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

    def select_tdnet_with_nenv60_nunit128(self):
        nenv60 = 60
        nunit128 = 128
        eid = np.where(np.array(self.tdnetRew.env_list) == nenv60)[0][0]
        uid = np.where(np.array(self.tdnetRew.unit_list) == nunit128)[0][0]
        
        self.a2c_e60_avg        = self.a2c_avg[eid]
        self.a2c_e60_sem        = self.a2c_sem[eid]
        self.tdnet_e60_u128     = self.tdnet[eid,:,uid]
        self.tdnet_e60_u128_avg = self.tdnet_avg[eid,uid]
        self.tdnet_e60_u128_sem = self.tdnet_sem[eid,uid]        

    def compare_tdnet_interpolation(self):    

        plt.figure(figsize=(2.3,1.5))
        plt.fill_between(self.tdnetInterRew.inter_list, self.a2c_e60_avg - self.a2c_e60_sem, self.a2c_e60_avg + self.a2c_e60_sem, color='r', alpha=0.3, ec='None')
        plt.axhline(self.a2c_e60_avg, color='r', label='A2C', linestyle='--')
        plt.fill_between(self.tdnetInterRew.inter_list, self.tdnet_e60_u128_avg - self.tdnet_e60_u128_sem, self.tdnet_e60_u128_avg + self.tdnet_e60_u128_sem, color='C0', alpha=0.3, ec='None')
        plt.axhline(self.tdnet_e60_u128_avg, color='C0', label='TDnet', linestyle='--')
        plt.fill_between(self.tdnetInterRew.inter_list, self.tdinter_avg - self.tdinter_sem, self.tdinter_avg + self.tdinter_sem, color='C1', alpha=0.3, ec='None')
        plt.plot(self.tdnetInterRew.inter_list, self.tdinter_avg, marker='o', color='C1', label='Interp')
        plt.legend(bbox_to_anchor=[1,1])        
        plt.xlabel('interpolation')
        plt.ylabel('reward')
        # plt.xticks([0,1,2])
        plt.tight_layout()
        # plt.savefig(self.pathfig + self.env_id + '_compare_interpolation.pdf')        
        
        