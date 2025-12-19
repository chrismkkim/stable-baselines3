import numpy as np
import results_plotter
import matplotlib.pyplot as plt
import os
import pickle

class RLerror:
    def __init__(
        self,
        adv: np.ndarray,
        dopa: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        rewards: np.ndarray,
        raw_rewards: np.ndarray,
        dones: np.ndarray
        ):
        
        self.adv = adv
        self.dopa = dopa
        self.values = values
        self.next_values = next_values
        self.rewards = rewards
        self.raw_rewards = raw_rewards
        self.dones = dones
        self.nsteps  = dones.shape[0]
        self.nenvs = dones.shape[1]
        
        self.states()
        self.errors()
        self.episode_reward()
        self.episode_err()
        self.episode_err_by_states()
        # self.remove_early_episodes()
        
    def states(self) -> None:
        self.state_nonterm = 1 - self.dones
        self.state_trunc   = (self.rewards != self.raw_rewards).astype(float)
        self.state_term    = (1 - self.state_trunc) * self.dones
        
    def errors(self) -> None:
        self.err         = (self.dopa - self.adv) / self.values
        self.err_nonterm = self.err * self.state_nonterm
        self.err_term    = self.err * self.state_term
        self.err_trunc   = self.err * self.state_trunc
        
    def episode_reward(self) -> None:        
        epis_reward = [[] for _ in range(self.nenvs)]
        for envi in range(self.nenvs):
            reward_envi    = self.raw_rewards[:,envi]
            t_done_envi = np.where(self.dones[:,envi] == 1)[0]
            for ix, done in enumerate(t_done_envi):
                if ix == 0:
                    epis_reward[envi].append(reward_envi[:done+1])
                else:
                    epis_reward[envi].append(reward_envi[last_done+1:done+1])
                last_done = done
        self.epis_reward = epis_reward                        
        self._rewards_in_time()
        
    def _rewards_in_time(self):
        rewards_in_time = []
        for envi in range(self.nenvs):
            nepis    = len(self.epis_reward[envi])
            rew_in_t = np.zeros((2,nepis))
            tcum     = 0
            for epi in range(nepis):
                duration        = len(self.epis_reward[envi][epi])
                tcum           += duration
                reward_epi      = np.sum(self.epis_reward[envi][epi])
                rew_in_t[:,epi] = np.array([tcum, reward_epi])
            rewards_in_time.append(rew_in_t)
        self.rewards_in_time = rewards_in_time
            
    def episode_err(self) -> None:        
        epis_err = [[] for _ in range(self.nenvs)]
        for envi in range(self.nenvs):
            err_envi    = self.err[:,envi]
            t_done_envi = np.where(self.dones[:,envi] == 1)[0]
            for ix, done in enumerate(t_done_envi):
                if ix == 0:
                    epis_err[envi].append(err_envi[:done+1])
                else:
                    epis_err[envi].append(err_envi[last_done+1:done+1])
                last_done = done
        self.epis_err = epis_err                        

    def episode_err_by_states(self) -> None:
        epis_err_nonterm = [[] for _ in range(self.nenvs)]
        epis_err_term    = [[] for _ in range(self.nenvs)]
        epis_err_len     = [[] for _ in range(self.nenvs)]
        epis_err_last30  = [[] for _ in range(self.nenvs)]
        for envi in range(self.nenvs):
            nepis = len(self.epis_err[envi])
            for epi in range(nepis):
                epis_err_nonterm[envi].append(self.epis_err[envi][epi][:-1])
                epis_err_term[envi].append(self.epis_err[envi][epi][-1]) 
                
                _epi_len = self.epis_err[envi][epi].shape[0]
                epis_err_len[envi].append(_epi_len)                
                Lidx = np.min([-2, -int(0.3*_epi_len)])
                epis_err_last30[envi].append(self.epis_err[envi][epi][Lidx:-1])
        self.epis_err_nonterm = epis_err_nonterm
        self.epis_err_term    = epis_err_term
        self.epis_err_last30  = epis_err_last30
        self.epis_err_len     = epis_err_len
                                
    def compute_avg_reward(self, tinc):
        ninc = int(self.nsteps / tinc)
        reward  = [[] for _ in range(ninc)]
        for envi in range(self.nenvs):
            time = self.rewards_in_time[envi][0,:]
            rew  = self.rewards_in_time[envi][1,:]
            tidx = (time/ tinc).astype(int)
            for ix, ti in enumerate(tidx):
                if ti == ninc:
                    ti = ti-1
                reward[ti].append(rew[ix])
            
        reward_avg = np.zeros(ninc)
        reward_sem = np.zeros(ninc)
        for ti in range(ninc):
            cnt = len(reward[ti])
            if cnt > 0:
                reward_avg[ti] = np.mean(reward[ti])
                reward_sem[ti] = np.std(reward[ti]) / np.sqrt(cnt)
        return reward_avg, reward_sem
                                
    def compute_mean_err(self,frac_time:float):
        _nonterm = np.array([])
        _term    = np.array([])
        _last30  = np.array([])
        for envi in range(self.nenvs):
            _epis_start   = self._find_start_episode(envi, frac_time)
            _nonterm_envi = np.concat(self.epis_err_nonterm[envi][_epis_start:])
            _term_envi    = np.array(self.epis_err_term[envi][_epis_start:])
            _last30_envi  = np.concat(self.epis_err_last30[envi][_epis_start:])
            
            _nonterm = np.append(_nonterm, _nonterm_envi)
            _term    = np.append(_term, _term_envi)
            _last30  = np.append(_last30, _last30_envi)            
        return np.mean(np.abs(_nonterm)), np.mean(np.abs(_term)), np.mean(np.abs(_last30)), np.std(np.abs(_nonterm)), np.std(np.abs(_term)), np.std(np.abs(_last30))
    
    def _find_start_episode(self, envi, frac_time):
        nepis = len(self.epis_err_nonterm[envi])
        epis_len = np.zeros(nepis)
        for epi in range(nepis):
            epis_len[epi] = self.epis_err_nonterm[envi][epi].shape[0]
        epis_cumlen = np.cumsum(epis_len)
        epis_start  = np.where(epis_cumlen > frac_time * epis_cumlen[-1])[0][0]
        # epis_start  = epis_cumlen[idx_start]
        return epis_start
        
    def _check_if_episodes_are_correct(self):        
        agent0 = 0
        x = np.array([])
        nepi = len(self.epis_err_nonterm[agent0])
        for epi in range(nepi):
            x = np.append(x,self.epis_err_nonterm[agent0][epi])    
            
        y = self.err_nonterm[:,agent0]
        y = y[np.abs(y)>0]
        y = y[:len(x)]            
        assert np.all(x == y)    
        
    def remove_early_episodes(self) -> None:
        cutoff_nonterm = [[] for _ in range(self.nenvs)]
        cutoff_term    = [[] for _ in range(self.nenvs)]
        cutoff_last30  = [[] for _ in range(self.nenvs)]        
        for envi in range(self.nenvs):
            cutoff_epi           = int(0.0*self.epis_err_nonterm[envi].shape[0])
            cutoff_nonterm[envi] = self.epis_err_nonterm[envi][cutoff_epi:]
            cutoff_term[envi]    = self.epis_err_term[envi][cutoff_epi:]
            cutoff_last30[envi]  = self.epis_err_last30[envi][cutoff_epi:]
        self.cutoff_nonterm = cutoff_nonterm
        self.cutoff_term    = cutoff_term
        self.cutoff_last30  = cutoff_last30
        
        
class AnalyzeErr:
    def __init__(
        self,
        err_nonterm: np.ndarray,
        err_term: np.ndarray,
        err_last30: np.ndarray,
        std_nonterm: np.ndarray,
        std_term: np.ndarray,
        std_last30: np.ndarray,
        reward: np.ndarray,
        ):
        
        self.nsims       = err_nonterm.shape[0]
        self.nenvs       = err_nonterm.shape[1]
        self.nunits      = err_nonterm.shape[2]
        self.ninc        = reward.shape[3]
        self.err_nonterm = err_nonterm
        self.err_term    = err_term
        self.err_last30  = err_last30
        self.std_nonterm = std_nonterm
        self.std_term    = std_term
        self.std_last30  = std_last30
        self.reward      = reward
                
        self.outliers        = []
        self.avg_err_nonterm = np.zeros((self.nenvs,self.nunits))
        self.avg_err_term    = np.zeros((self.nenvs,self.nunits))
        self.avg_err_last30  = np.zeros((self.nenvs,self.nunits))
        self.avg_std_nonterm = np.zeros((self.nenvs,self.nunits))
        self.avg_std_term    = np.zeros((self.nenvs,self.nunits))
        self.avg_std_last30  = np.zeros((self.nenvs,self.nunits))
        self.reward_avg      = np.zeros((self.nenvs,self.nunits,self.ninc))
        self.reward_sem      = np.zeros((self.nenvs,self.nunits,self.ninc))
        self.reward_time_avg = np.zeros((self.nenvs,self.nunits))
        
        self.avg_err_outliers_removed()
        self.avg_rew_outliers_removed()
        
    def avg_err_outliers_removed(self):
        for env in range(self.nenvs):
            for unit in range(self.nunits):
                idx_outliers1 = self._remove_outliers(self.err_nonterm[:,env,unit])
                idx_outliers2 = self._remove_outliers(self.std_nonterm[:,env,unit])
                idx_outliers  = np.append(idx_outliers1, idx_outliers2)
                err_nonterm_outliers_removed   = np.delete(self.err_nonterm[:,env,unit], idx_outliers)
                err_term_outliers_removed      = np.delete(self.err_term[:,env,unit],    idx_outliers)
                err_last30_outliers_removed    = np.delete(self.err_last30[:,env,unit],  idx_outliers)                
                std_nonterm_outliers_removed   = np.delete(self.std_nonterm[:,env,unit], idx_outliers)
                std_term_outliers_removed      = np.delete(self.std_term[:,env,unit],    idx_outliers)
                std_last30_outliers_removed    = np.delete(self.std_last30[:,env,unit],  idx_outliers)
                
                self.avg_err_nonterm[env,unit] = np.mean(err_nonterm_outliers_removed)
                self.avg_err_term[env,unit]    = np.mean(err_term_outliers_removed)
                self.avg_err_last30[env,unit]  = np.mean(err_last30_outliers_removed)
                self.avg_std_nonterm[env,unit] = np.mean(std_nonterm_outliers_removed)
                self.avg_std_term[env,unit]    = np.mean(std_term_outliers_removed)
                self.avg_std_last30[env,unit]  = np.mean(std_last30_outliers_removed)
                if len(idx_outliers) > 0:
                    self.outliers.append([env,unit,idx_outliers])                    
                
    def _remove_outliers(self, xsims):
        # idx_outliers = []
        # for i in range(self.nsims):
        #     big_gaps = np.log10(xsims[i]) - np.log10(xsims) > 1.5
        #     if np.sum(big_gaps) > self.nsims * 0.7:
        #         idx_outliers.append(i)
        idx_outliers = np.where(xsims > 1)[0]
        return idx_outliers
                
    def avg_rew_outliers_removed(self):
        for env in range(self.nenvs):
            for unit in range(self.nunits):
                idx_outliers1                  = self._remove_outliers(self.err_nonterm[:,env,unit])
                idx_outliers2                  = self._remove_outliers(self.std_nonterm[:,env,unit])
                idx_outliers                   = np.append(idx_outliers1, idx_outliers2)
                rew_outliers_removed           = np.delete(self.reward[:,env,unit,:], idx_outliers, axis=0)
                nsims                          = rew_outliers_removed.shape[0]
                self.reward_avg[env,unit]      = np.mean(rew_outliers_removed,axis=0)
                self.reward_sem[env,unit]      = np.std(rew_outliers_removed,axis=0) / np.sqrt(nsims)
                self.reward_time_avg[env,unit] = np.mean(rew_outliers_removed[:,-int(0.1*self.ninc):])
                

class AnalyzeReward:
    def __init__(
        self,
        err_nonterm: np.ndarray,
        std_nonterm: np.ndarray,
        reward: np.ndarray,
        ):
        
        self.nsims       = err_nonterm.shape[0]
        self.nenvs       = err_nonterm.shape[1]
        self.nlr         = err_nonterm.shape[2]
        self.nent        = err_nonterm.shape[3]
        self.ninc        = reward.shape[-1]
        self.err_nonterm = err_nonterm
        self.std_nonterm = std_nonterm
        self.reward      = reward
                
        self.outliers        = []
        self.reward_avg      = np.zeros((self.nenvs,self.nlr,self.nent,self.ninc))
        self.reward_sem      = np.zeros((self.nenvs,self.nlr,self.nent,self.ninc))
        self.reward_time_avg = np.zeros((self.nenvs,self.nlr,self.nent))
        
        self.avg_rew_outliers_removed()             
                
    def _remove_outliers(self, xsims):
        idx_outliers = np.where(xsims > 1)[0]
        return idx_outliers
                
    def avg_rew_outliers_removed(self):
        for eid in range(self.nenvs):
            for rid in range(self.nlr):
                for cid in range(self.nent):
                    idx_outliers1                     = self._remove_outliers(self.err_nonterm[:,eid,rid,cid])
                    idx_outliers2                     = self._remove_outliers(self.std_nonterm[:,eid,rid,cid])
                    idx_outliers                      = np.append(idx_outliers1, idx_outliers2)
                    rew_outliers_removed              = np.delete(self.reward[:,eid,rid,cid,:], idx_outliers, axis=0)
                    nsims                             = rew_outliers_removed.shape[0]
                    self.reward_avg[eid,rid,cid]      = np.mean(rew_outliers_removed,axis=0)
                    self.reward_sem[eid,rid,cid]      = np.std(rew_outliers_removed,axis=0) / np.sqrt(nsims)
                    self.reward_time_avg[eid,rid,cid] = np.mean(rew_outliers_removed[:,-int(0.1*self.ninc):])
                

class A2CReward:
    def __init__(
        self,
        sims_list: np.ndarray,
        env_list: list,
        lr_list: list,
        ent_list: list,
        evalnum: float,
        env_id_rl: str,
        basetime: float,
        pathdata: str,
        pathsave: str,
        pathclass: str,
        pathparam: str,
        ):
        
        self.env_list     = env_list
        self.lr_list      = lr_list
        self.ent_list     = ent_list
        self.nsims        = len(sims_list)
        self.nenv         = len(env_list)
        self.nlr          = len(lr_list)
        self.nent         = len(ent_list)
        self.evalnum      = evalnum
        self.env_id       = env_id_rl
        self.basetime     = basetime        
        
        self.pathdata     = pathdata
        self.pathsave     = pathsave
        self.pathclass    = pathclass
        self.pathparam    = pathparam
        
        self.a2c_reward = np.zeros((self.nsims, self.nenv, self.nlr, self.nent, self.evalnum))
        
        self.load_biowulf_a2c_data()
        
        self.find_max_reward()
        
        topk_frac = 0.1
        self.find_topk_hyperparam(topk_frac=topk_frac)
        
        # self.save_data()
                
    def load_biowulf_a2c_data(self):
        for eid in range(self.nenv):
            for rid in range(self.nlr):
                for cid in range(self.nent):
                    for sid in range(self.nsims):
                        _nenv    = self.env_list[eid]
                        _lr      = self.lr_list[rid]
                        _ent     = self.ent_list[cid]
                        # pathhyper = 'env_' + str(_nenv) + '_lr_' + str(_lr) + '_ent_' + _ent
                        pathhyper = 'env_' + str(_nenv) + '_lr_' + '{:.5f}'.format(_lr).rstrip('0') + '_ent_' + _ent
                        # "{:.5f}".format(x).rstrip('0')
                        pathA2C   = self.pathdata + self.env_id + '/a2c/evaluations/' + pathhyper + '/' + str(sid) + '/a2c/' + self.env_id + '_1/'
                        _reward = np.load(pathA2C + 'evaluations.npz', allow_pickle=True)['results']
                        _reward_mean = np.mean(_reward,axis=1)
                        if len(_reward_mean) < self.evalnum:
                            print('eid ', eid, ', rid ', rid, ', cid', cid, ', sid', sid)
                            gap = self.evalnum - len(_reward_mean)
                            _reward_extended = np.hstack((_reward_mean,_reward_mean[-gap:]))
                            self.a2c_reward[sid,eid,rid,cid,:]  = _reward_extended
                        else:
                            self.a2c_reward[sid,eid,rid,cid,:] = np.mean(_reward,axis=1) # mean in episodes, max in time
                            
    def find_max_reward(self):
        # stepone = 1 # will skip first time point
        # self.max_reward_epis = np.max(self.a2c_reward[:,:,:,:,stepone:],axis=-1) # max in time
        self.max_reward_epis = np.max(self.a2c_reward,axis=-1) # max in time
        self.max_reward = np.mean(self.max_reward_epis[:,:,:,:],axis=0) # mean in sims
        
    def find_topk_hyperparam(self, topk_frac):
        topk               = int(topk_frac * self.nlr * self.nent)
        max_env            = self.nenv-1 # nenv = 50
        _reward_env        = self.max_reward[max_env,:,:] # LR vs. Ent
        # first index: LR, second index: Ent
        self.topk_param_idx = np.unravel_index(np.argsort(_reward_env.flatten())[-topk:], _reward_env.shape)

    def find_max_reward_param_idx(self, _nenv):
        env_idx      = np.where(np.array(self.env_list) == _nenv)[0][0]
        _reward_env  = self.max_reward[env_idx,:,:] # LR vs. Ent
        # first index: LR, second index: Ent
        rid, cid = np.unravel_index(np.argsort(_reward_env.flatten())[-1], _reward_env.shape)
        return rid, cid
        
    def save_data(self):        
        path_to_class = self.pathsave + self.pathclass
        path_to_param = self.pathsave + self.pathparam
        
        ntopk = self.topk_param_idx[0].shape[0]
        self.topk_param = np.zeros((ntopk,2))
        for i in range(ntopk):
            self.topk_param[i,0] = self.lr_list[self.topk_param_idx[0][i]]
            self.topk_param[i,1] = self.ent_list[self.topk_param_idx[1][i]]
                    
        with open(path_to_class, "wb") as f:
            pickle.dump(self,f)
        
        with open(path_to_param, "wb") as f:
            pickle.dump(self.topk_param,f)
            
    def _get_optimal_lr_and_entropy(self):
        # LR and Ent indices selected for nenv = nenv_max
        nenv_max        = self.env_list[-1]
        rid, cid        = self.find_max_reward_param_idx(nenv_max)
        return rid, cid
        
    def getEnv_with_optimal_lr_and_entropy(self):
        
        a2c   = np.zeros((self.nenv, self.nsims))
        rid, cid = self._get_optimal_lr_and_entropy()
        for eid in range(self.nenv):
            a2cRew_max_sims = self.max_reward_epis[:,eid,rid,cid]
            a2c[eid] = a2cRew_max_sims
        return a2c
    
    def get_averages(self):
        #--- all data ---#
        #   * a2c: nenv x nsims
        a2c = self.getEnv_with_optimal_lr_and_entropy()
        
        #--- average over sims ---#
        #   * a2c_avg: nenv
        a2c_avg = np.mean(a2c, axis=1)
        a2c_sem = np.std(a2c, axis=1) / np.sqrt(self.nsims)
        
        return a2c, a2c_avg, a2c_sem
        

class TDnetReward:
    def __init__(
        self,
        sims_list: np.ndarray,
        env_list: list,
        unit_list: list,
        evalnum: float,
        env_id_rl: str,
        basetime: float,
        pathdata: str,
        pathsave: str,
        pathclass: str,
        pathparam: str,
        ):
        
        self.env_list     = env_list
        self.unit_list    = unit_list
        self.nsims        = len(sims_list)
        self.nenv         = len(env_list)
        self.nunit        = len(unit_list)
        self.evalnum      = evalnum
        self.env_id       = env_id_rl
        self.basetime     = basetime        
        
        self.pathdata     = pathdata
        self.pathsave     = pathsave
        self.pathclass    = pathclass
        self.pathparam    = pathparam
        
        self.tdnet_reward = np.zeros((self.nsims, self.nenv, self.nunit, self.evalnum))
        
        self.load_biowulf_tdnet_data()
        
        self.find_max_reward()
        
        topk_frac = 0.1
        self.find_topk_hyperparam(topk_frac=topk_frac)
        
        # self.save_data()
                
    def load_biowulf_tdnet_data(self):
        for eid in range(self.nenv):
            for uid in range(self.nunit):
                for sid in range(self.nsims):
                    _nenv    = self.env_list[eid]
                    _nunit   = self.unit_list[uid]
                    pathhyper = 'env_' + str(_nenv) + '_unit_' + str(_nunit)
                    pathTDnet = self.pathdata + self.env_id + '/tdnet/evaluations/' + pathhyper + '/' + str(sid) + '/dopa/' + self.env_id + '_2/'
                    if os.path.isfile(pathTDnet + 'evaluations.npz'):
                        # _reward:      ntime x nepis
                        # tdnet_reward: nsim  x nenv  x nunit x ntime
                        _reward = np.load(pathTDnet + 'evaluations.npz', allow_pickle=True)['results']        
                        if len(np.mean(_reward,axis=1)) == self.evalnum:                  
                            self.tdnet_reward[sid,eid,uid,:] = np.mean(_reward,axis=1) # mean in episodes
                        else:
                            self.tdnet_reward[sid,eid,uid,:] = -999 # dummy number
                            
    def find_max_reward(self):
        # tdnet_reward:    nsim x nenv x nunit x ntime
        # max_reward_epis: nsim x nenv x nunit
        # max_reward:             nenv x nunit
        stepone = 2 # will skip first time point
        self.max_reward_epis = np.max(self.tdnet_reward[:,:,:,stepone:],axis=-1) # max in time
        self.max_reward = np.mean(self.max_reward_epis[:,:,:],axis=0) # mean in sims
        
    def find_topk_hyperparam(self, topk_frac):
        topk               = int(topk_frac * self.nenv * self.nunit)
        self.topk_param_idx = np.unravel_index(np.argsort(self.max_reward.flatten())[-topk:], self.max_reward.shape)
        
    def save_data(self):        
        path_to_class = self.pathsave + self.pathclass
        path_to_param = self.pathsave + self.pathparam
        
        ntopk = self.topk_param_idx[0].shape[0]
        self.topk_param = np.zeros((ntopk,2))
        for i in range(ntopk):
            self.topk_param[i,0] = self.env_list[self.topk_param_idx[0][i]]
            self.topk_param[i,1] = self.unit_list[self.topk_param_idx[1][i]]
                    
        with open(path_to_class, "wb") as f:
            pickle.dump(self,f)
        
        with open(path_to_param, "wb") as f:
            pickle.dump(self.topk_param,f)
                                        
    def adjust_shape(self):    
        # shape after the swap: nenv x nsims x nunit
        tdnet = np.swapaxes(self.max_reward_epis, 0, 1)    
        return tdnet

    def get_averages(self):
        #--- all the data ---#
        #   * tdnet: nenv x nsims x nunit
        tdnet = self.adjust_shape()
        
        #--- average over sims ---#
        #   * tdnet_avg: nenv x nunit
        tdnet_avg = np.mean(tdnet,axis=1)
        tdnet_sem = np.std(tdnet,axis=1) / np.sqrt(self.nsims)
        
        #--- average over nn / env ---#
        #   * tdnet_nn_avg:  nunit
        #   * tdnet_env_avg: nenv
        tdnet_nn_avg  = np.mean(tdnet_avg, axis=0)
        tdnet_nn_sem  =  np.std(tdnet_avg, axis=0) / np.sqrt(tdnet_avg.shape[0])
        tdnet_env_avg = np.mean(tdnet_avg, axis=1)
        tdnet_env_sem =  np.std(tdnet_avg, axis=1) / np.sqrt(tdnet_avg.shape[1])
        
        #--- best nn ---#
        #   * tdnet_best_nn:  nenv x nsims
        #   * tdnet_best_avg: nenv
        tdnet_best_nn = np.zeros((self.nenv, self.nsims))
        for envi in range(self.nenv):
            best_nn             = np.argmax(tdnet_avg[envi,:])
            tdnet_best_nn[envi] = tdnet[envi,:,best_nn]
        tdnet_best_avg = np.mean(tdnet_best_nn, axis=1)
        tdnet_best_sem =  np.std(tdnet_best_nn, axis=1) / np.sqrt(self.nsims)
        
        return tdnet, \
                tdnet_avg, tdnet_sem, \
                tdnet_nn_avg, tdnet_nn_sem, \
                tdnet_env_avg, tdnet_env_sem, \
                tdnet_best_nn, tdnet_best_avg, tdnet_best_sem
                
                

class TDinterpolate:
    def __init__(
        self,
        sims_list: np.ndarray,
        env_list: list,
        unit_list: list,
        inter_list: list,
        evalnum: float,
        env_id_rl: str,
        basetime: float,
        pathdata: str,
        pathsave: str,
        pathclass: str,
        ):
        
        self.env_list     = env_list
        self.unit_list    = unit_list
        self.inter_list   = inter_list
        self.nsims        = len(sims_list)
        self.nenv         = len(env_list)
        self.nunit        = len(unit_list)
        self.ninter       = len(inter_list)
        self.evalnum      = evalnum
        self.env_id       = env_id_rl
        self.basetime     = basetime        
        
        self.pathdata     = pathdata
        self.pathsave     = pathsave
        self.pathclass    = pathclass
        
        self.tdnet_reward = np.zeros((self.nsims, self.ninter, self.evalnum))
        
        self.load_biowulf_tdnet_data()
        
        self.find_max_reward()
        
        # self.save_data()
                
    def load_biowulf_tdnet_data(self):
        for iid in range(self.ninter):
            for sid in range(self.nsims):
                _nenv    = self.env_list[0]
                _nunit   = self.unit_list[0]
                _ninter  = self.inter_list[iid]
                pathhyper = 'env_' + str(_nenv) + '_unit_' + str(_nunit) + '_inter_' + str(_ninter)
                pathTDnet = self.pathdata + self.env_id + '/tdnet/interpolate/evaluations/' + pathhyper + '/' + str(sid) + '/dopa/' + self.env_id + '_2/'
                # pathTDnet = self.pathdata + self.env_id + '/tdnet/interpolate/evaluations/previous/' + pathhyper + '/' + str(sid) + '/dopa/' + self.env_id + '_2/'
                if os.path.isfile(pathTDnet + 'evaluations.npz'):
                    # _reward:      ntime x nepisode
                    # tdnet_reward: nsim  x ninterpolation x ntime
                    _reward = np.load(pathTDnet + 'evaluations.npz', allow_pickle=True)['results']                        
                    self.tdnet_reward[sid,iid,:] = np.mean(_reward,axis=1) # mean in episodes
                else:
                    print('inter ' + str(iid) + ', sim ' + str(sid) + ' does not exist')
                            
    def find_max_reward(self):
        # tdnet_reward:    nsim x ninterpolation x ntime
        # max_reward_epis: nsim x ninterpolation
        # max_reward:             ninterpolation
        self.max_reward_epis = np.max(self.tdnet_reward,axis=-1) # max in time
        self.max_reward = np.mean(self.max_reward_epis,axis=0)   # mean in sims        
        
    def save_data(self):        
        path_to_class = self.pathsave + self.pathclass        
        with open(path_to_class, "wb") as f:
            pickle.dump(self,f)
                                        
    def adjust_shape(self):    
        # shape after the swap: ninterpolation x nsims
        tdnet = np.swapaxes(self.max_reward_epis, 0, 1)    
        return tdnet

    def get_averages(self):
        #--- all the data ---#
        #   * tdnet: ninterpolation x nsims
        tdnet = self.adjust_shape()
        
        #--- average over sims ---#
        #   * tdnet_avg: ninterpolation
        tdnet_avg = np.mean(tdnet,axis=1)
        tdnet_sem = np.std(tdnet,axis=1) / np.sqrt(self.nsims)
        
        return tdnet, tdnet_avg, tdnet_sem
        