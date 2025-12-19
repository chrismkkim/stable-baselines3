import numpy as np
import results_plotter
import matplotlib.pyplot as plt
import os
from utils import RLerror, AnalyzeErr

plt.style.use(os.path.join(os.path.dirname(__file__), 'pyplot_setting.mplstyle'))

env_cart  = 'CartPole-v1'
env_lunar = 'LunarLander-v3'

# meta_train_type = 'Cart-Cart'
meta_train_type = 'Lunar-Lunar'
# meta_train_type = 'Cart-Lunar'
# meta_train_type = 'Lunar-Cart'

if meta_train_type == 'Cart-Cart':
    env_id_meta = env_cart
    env_id_rl   = env_cart
    meta_log    = '2'
    path_data = '/Users/kimchm/Documents/RL/biowulf/cartpole/n_rlnet_reset_15/' 
    basetime = 5000
    nreset = 15
elif meta_train_type == 'Lunar-Lunar':
    env_id_meta = env_lunar
    env_id_rl   = env_lunar
    meta_log    = '2'
    path_data = '/Users/kimchm/Documents/RL/biowulf/lunar/tdnet/' 
    basetime = 10000
    nreset = 10
elif meta_train_type == 'Cart-Lunar':
    env_id_meta = env_cart
    env_id_rl   = env_lunar
    meta_log    = '1'
elif meta_train_type == 'Lunar-Cart':
    env_id_meta = env_lunar
    env_id_rl   = env_cart
    meta_log    = '1'

pathfig = '/Users/kimchm/Documents/GitHub/stable-baselines3/rundopa/figure/'

sims_list  = [0,1,2,3,4,5,6,7,8,9]
envs_list  = [1, 4, 7, 10, 20, 30, 40]
# units_list = [16, 32, 64, 128, 256, 384, 512, 640, 768]
units_list = [16, 32, 64, 128, 256, 384, 512]
# units_list = [64, 128, 256, 384, 512, 640, 768]
# units_list = [16, 32, 64, 128, 256]
nsims  = len(sims_list)
nenvs  = len(envs_list)
nunits = len(units_list)
tinc   = 50
ninc  = int(basetime/tinc)
rlerr_trace      = [np.zeros((nsims,nunits,basetime,envi)) for envi in envs_list]
rlerr_trace_term = [np.zeros((nsims,nunits,basetime,envi)) for envi in envs_list]
rlerr_trace_nonterm = [np.zeros((nsims,nunits,basetime,envi)) for envi in envs_list]
err_list_nonterm = np.zeros((nsims,nenvs,nunits))
err_list_term    = np.zeros((nsims,nenvs,nunits))
err_list_last30  = np.zeros((nsims,nenvs,nunits))
std_list_nonterm = np.zeros((nsims,nenvs,nunits))
std_list_term    = np.zeros((nsims,nenvs,nunits))
std_list_last30  = np.zeros((nsims,nenvs,nunits))
reward_list      = np.zeros((nsims,nenvs,nunits,ninc))
metaerr_list     = [np.zeros((nsims,nunits,basetime*nreset,envi)) for envi in envs_list]
metavalue_list   = [np.zeros((nsims,nunits,basetime*nreset,envi)) for envi in envs_list]
# reward_sem_list    = np.zeros((nsims,nenvs,nunits,ninc))
for eid in range(nenvs):
    for uid in range(nunits):
        for six, sid in enumerate(sims_list):
            pathTask        = path_data  + env_id_meta  + '_' + env_id_rl + '/' 
            pathParam       = 'env_' + str(envs_list[eid]) + '_unit_' + str(units_list[uid])
            pathSim         = '/' + str(sid)
            pathDA          = pathTask + pathParam + pathSim
            pathDA_all      = [pathDA + '/dopa/' + env_id_rl + '_' + meta_log + '/']            

            if os.path.isfile(pathDA + '/dopa/' + 'rl_advantages.npy'):
                x_adv           = np.load(pathDA + '/dopa/' + 'rl_advantages.npy', allow_pickle=True)
                x_dopa          = np.load(pathDA + '/dopa/' + 'rl_dopa.npy', allow_pickle=True)
                x_values        = np.load(pathDA + '/dopa/' + 'rl_values.npy', allow_pickle=True)
                x_next_values   = np.load(pathDA + '/dopa/' + 'rl_next_values.npy', allow_pickle=True)
                x_rewards       = np.load(pathDA + '/dopa/' + 'rl_rewards.npy', allow_pickle=True)
                x_raw_rewards   = np.load(pathDA + '/dopa/' + 'rl_raw_rewards.npy', allow_pickle=True)
                x_dones         = np.load(pathDA + '/dopa/' + 'rl_dones.npy', allow_pickle=True)    
                x_metaerr       = np.load(pathDA + '/dopa/' + 'meta_err_nonterm.npy', allow_pickle=True)    
                x_metavalue     = np.load(pathDA + '/dopa/' + 'meta_values.npy', allow_pickle=True)    

                rlerr = RLerror(x_adv, x_dopa, x_values, x_next_values, x_rewards, x_raw_rewards, x_dones)             

                avgerr_nonterm, avgerr_term, avgerr_last30, \
                stderr_nonterm, stderr_term, stderr_last30 = rlerr.compute_mean_err(frac_time=0.1)
                
                reward_avg, reward_sem = rlerr.compute_avg_reward(tinc)
                
                err_list_nonterm[six,eid,uid] = avgerr_nonterm
                err_list_term[six,eid,uid]    = avgerr_term
                err_list_last30[six,eid,uid]  = avgerr_last30
                std_list_nonterm[six,eid,uid] = stderr_nonterm
                std_list_term[six,eid,uid]    = stderr_term
                std_list_last30[six,eid,uid]  = stderr_last30
                
                reward_list[six,eid,uid] = reward_avg
                # reward_sem_list[six,eid,uid] = reward_sem
                
                rlerr_trace[eid][six,uid,:] = rlerr.err
                rlerr_trace_term[eid][six,uid,:] = rlerr.err_term
                rlerr_trace_nonterm[eid][six,uid,:] = rlerr.err_nonterm
                
                metaerr_list[eid][six,uid,:]   = x_metaerr
                metavalue_list[eid][six,uid,:] = x_metavalue
            else:
                print(uid)

anlyzErr = AnalyzeErr(err_list_nonterm, err_list_term, err_list_last30,
                        std_list_nonterm, std_list_term, std_list_last30,
                        reward_list)


#------ load a2c reward --------#
a2c_env_list = [10,20,30,40]
a2c_nenvs = len(a2c_env_list)
a2c_reward = np.zeros((nsims,a2c_nenvs,ninc))
for envi in range(a2c_nenvs):
    for sid in range(nsims):
        _nenvs = a2c_env_list[envi]
        # pathA2C = '/Users/kimchm/Documents/RL/biowulf/a2c_cartpole/' + env_id_rl + '/' + 'env_' + str(_nenvs) + '/' + str(sid) + '/a2c/' + env_id_rl + '_1/'
        pathA2C = '/Users/kimchm/Documents/RL/biowulf/lunar/a2c/' + env_id_rl + '/' + 'env_' + str(_nenvs) + '/' + str(sid) + '/a2c/' + env_id_rl + '_1/'
        _, a2c_reward[sid,envi], _ = results_plotter.load_rlzoo(pathA2C,  x_axis="timesteps", nenvs=_nenvs, ninc=ninc, tinc=tinc)
a2c_reward_avg = np.mean(a2c_reward,axis=0)


#---- meta error analysis ------#
metaerr_avg = np.zeros((nenvs,nreset))
plt.figure(figsize=(2.2,1.8))
for eid in range(nenvs):
    tmp = np.mean(np.abs(metaerr_list[eid][:,uid,:]),axis=(0,2))
    metaerr_split = np.array_split(tmp, nreset)
    _metaerr_avg = np.array([split.mean() for split in metaerr_split])
    metaerr_avg[eid,:] = _metaerr_avg
    plt.plot(np.log10(_metaerr_avg), label=str(envs_list[eid]))
plt.plot(np.log10(np.mean(metaerr_avg,axis=0)), c='k', marker='o', lw=2.0, ms=2.5)        
plt.plot(np.log10(np.mean(metaerr_avg,axis=0)), c='w', marker='o', lw=0.5)            
plt.legend(bbox_to_anchor=[1,1], title='nenv')
plt.ylim([-2.5,0])
plt.xlabel('num reset')
plt.ylabel('meta error')
plt.tight_layout()
plt.savefig(pathfig + 'meta_err_lunar.pdf')


metavalue_avg = np.zeros((nenvs,nreset))
plt.figure(figsize=(2.2,1.8))
for eid in range(nenvs):
    tmp = np.mean(np.abs(metavalue_list[eid][:,uid,:]),axis=(0,2))
    metavalue_split = np.array_split(tmp, nreset)
    _metavalue_avg = np.array([split.mean() for split in metavalue_split])
    metavalue_avg[eid,:] = _metavalue_avg
    plt.plot(np.log10(_metavalue_avg), label=str(envs_list[eid]))
plt.plot(np.log10(np.mean(metavalue_avg,axis=0)), c='k', marker='o', lw=2.0, ms=2.5)        
plt.plot(np.log10(np.mean(metavalue_avg,axis=0)), c='w', marker='o', lw=0.5)            
plt.legend(bbox_to_anchor=[1,1], title='nenv')
# plt.ylim([-2.5,0])
plt.xlabel('num reset')
plt.ylabel('values (meta train)')
plt.tight_layout()
plt.savefig(pathfig + 'meta_value_lunar.pdf')



# def mvavg(x,wid):
#     xavg = np.zeros_like(x)
#     tlen = x.shape[0]
#     for t in range(tlen):
#         Lidx = np.max([0,t-wid])
#         Ridx = np.min([tlen-1,t+wid])
#         xavg[t] = np.mean(x[Lidx:Ridx])
#     return xavg    
        
# # meta error during training    
# sid = 0
# uid = 4
# aid = 0
# plt.figure(figsize=(4,4))
# for eid in range(nenvs):
#     nenv = envs_list[eid]
#     plt.subplot(4,2,eid+1)
#     err = np.abs(metaerr_list[eid][sid,uid,:,aid])
#     err_avg = mvavg(err, wid=100)
#     plt.plot(np.log10(err))
#     plt.plot(np.log10(err_avg))
#     plt.axhline(-2,color='r', linestyle='--')
#     for i in range(nreset):
#         plt.axvline(basetime*i, color="gray", linestyle="--")
#     plt.title('env ' + str(nenv), fontsize=5)
# plt.tight_layout()


# eid = 4
# six = 0
# uid = 3
# aid = 1
# plt.figure(figsize=(8,3))
# tidx = np.where(np.abs(rlerr_trace_term[eid][six,uid,:,aid])>0)[0]
# plt.plot(np.log10(np.abs(rlerr_trace_nonterm[eid][six,uid,:,aid])))
# plt.plot(tidx,np.log10(np.abs(rlerr_trace_term[eid][six,uid,tidx,aid])), marker='o', linestyle='')
# plt.tight_layout()

# np.mean(np.abs(rlerr_trace_nonterm[eid][six,uid,:,aid]))

# tidx = np.where(np.abs(rlerr_trace_term[eid][six,uid,:,aid])>0)[0]
# np.mean(np.abs(rlerr_trace_term[eid][six,uid,tidx,aid]))




#-------- network size vs. error  -------#
env10 = 3

#--- non-terminal states ---#
plt.figure(figsize=(2.0,1.3))
for eid in np.arange(env10,nenvs):
# for eid in np.arange(nenvs):    
    plt.plot(units_list, np.log10(anlyzErr.avg_err_nonterm[eid,:]), label=str(envs_list[eid]), marker='o')
plt.plot(units_list, np.log10(np.mean(anlyzErr.avg_err_nonterm[env10:,:],axis=0)), c='k', marker='o', lw=2.0, ms=2.5)        
plt.plot(units_list, np.log10(np.mean(anlyzErr.avg_err_nonterm[env10:,:],axis=0)), c='w', marker='o', lw=0.5)        
plt.legend(frameon=False, bbox_to_anchor=[1, 1], title='nenvs')  
# plt.ylim([-2.5, -1])
plt.xlabel('network size')
plt.ylabel('log (mean error)')
plt.tight_layout()  
plt.savefig(pathfig + 'ntwk_vs_avgerr_nonterm.pdf')


plt.figure(figsize=(2.0,1.3))
for eid in np.arange(env10,nenvs):    
    plt.plot(units_list, np.log10(anlyzErr.avg_std_nonterm[eid,:]), label=str(envs_list[eid]), marker='o')
plt.plot(units_list, np.log10(np.mean(anlyzErr.avg_std_nonterm[env10:,:],axis=0)), c='k', marker='o', lw=2.0, ms=2.5)        
plt.plot(units_list, np.log10(np.mean(anlyzErr.avg_std_nonterm[env10:,:],axis=0)), c='w', marker='o', lw=0.5)            
# plt.ylim([-3.5, 0.5])
plt.legend(frameon=False, bbox_to_anchor=[1, 1], title='nenvs')    
plt.xlabel('network size')
plt.ylabel('log (std of error)')    
plt.tight_layout()
plt.savefig(pathfig + 'ntwk_vs_stderr_nonterm.pdf')
# plt.show()


#--- terminal states---
plt.figure(figsize=(2.0,1.3))
for eid in np.arange(env10,nenvs):
    plt.plot(units_list, np.log10(anlyzErr.avg_err_term[eid,:]), label=str(envs_list[eid]), marker='o')
plt.plot(units_list, np.log10(np.mean(anlyzErr.avg_err_term[env10:,:],axis=0)), c='k', marker='o', lw=2.0, ms=2.5)        
plt.plot(units_list, np.log10(np.mean(anlyzErr.avg_err_term[env10:,:],axis=0)), c='w', marker='o', lw=0.5)        
plt.legend(frameon=False, bbox_to_anchor=[1, 1], title='nenvs')  
plt.ylim([-2, 1.2])
plt.xlabel('network size')
plt.ylabel('log (mean error)')
plt.tight_layout()  
# plt.savefig(pathfig + 'ntwk_vs_avgerr_term.pdf')


plt.figure(figsize=(2.0,1.3))
for eid in np.arange(env10,nenvs):    
    plt.plot(units_list, np.log10(anlyzErr.avg_std_term[eid,:]), label=str(envs_list[eid]), marker='o')
plt.plot(units_list, np.log10(np.mean(anlyzErr.avg_std_term[env10:,:],axis=0)), c='k', marker='o', lw=2.0, ms=2.5)        
plt.plot(units_list, np.log10(np.mean(anlyzErr.avg_std_term[env10:,:],axis=0)), c='w', marker='o', lw=0.5)            
plt.ylim([-2, 1.2])
plt.legend(frameon=False, bbox_to_anchor=[1, 1], title='nenvs')    
plt.xlabel('network size')
plt.ylabel('log (std of error)')    
plt.tight_layout()
# plt.savefig(pathfig + 'ntwk_vs_stderr_term.pdf')
# plt.show()




#-------- error vs. num of environments -------#
unit64 =2
plt.figure(figsize=(2.0,1.3))
for uid in np.arange(unit64,nunits):
    plt.plot(envs_list, np.log10(anlyzErr.avg_err_nonterm[:,uid]), label=str(units_list[uid]), marker='o')
plt.plot(envs_list, np.log10(np.mean(anlyzErr.avg_err_nonterm[:,unit64:],axis=1)), c='k', marker='o', lw=2.0, ms=2.5)
plt.plot(envs_list, np.log10(np.mean(anlyzErr.avg_err_nonterm[:,unit64:],axis=1)), c='w', marker='o', lw=0.5)    
# plt.ylim([-2.5, 0])
plt.xlabel('num of envs')
plt.ylabel('log (mean error)')
plt.legend(frameon=False, bbox_to_anchor=[1, 1], title='ntwk')    
plt.tight_layout()
plt.savefig(pathfig + 'nenvs_vs_avgerr_nonterm.pdf')


unit64 =2
plt.figure(figsize=(2.0,1.3))
for uid in np.arange(unit64,nunits):
    plt.plot(envs_list, np.log10(anlyzErr.avg_std_nonterm[:,uid]), label=str(units_list[uid]), marker='o')
plt.plot(envs_list, np.log10(np.mean(anlyzErr.avg_std_nonterm[:,unit64:],axis=1)), c='k', marker='o', lw=2.0, ms=2.5)            
plt.plot(envs_list, np.log10(np.mean(anlyzErr.avg_std_nonterm[:,unit64:],axis=1)), c='w', marker='o', lw=0.5)        
# plt.ylim([-3.5, 0.5])
plt.legend(frameon=False, bbox_to_anchor=[1, 1], title='ntwk')    
plt.xlabel('num of envs')
plt.ylabel('log (std of error)')
plt.tight_layout()
plt.savefig(pathfig + 'nenvs_vs_stderr_nonterm.pdf')
# plt.show()



plt.figure(figsize=(2.0,1.3))
for uid in np.arange(unit64,nunits):
    plt.plot(envs_list, np.log10(anlyzErr.avg_err_term[:,uid]), label=str(units_list[uid]), marker='o')
plt.plot(envs_list, np.log10(np.mean(anlyzErr.avg_err_term[:,unit64:],axis=1)), c='k', marker='o', lw=2.0, ms=2.5)
plt.plot(envs_list, np.log10(np.mean(anlyzErr.avg_err_term[:,unit64:],axis=1)), c='w', marker='o', lw=0.5)    
plt.ylim([-2.5, 1.5])
plt.xlabel('num of envs')
plt.ylabel('log (mean error)')
plt.legend(frameon=False, bbox_to_anchor=[1, 1], title='ntwk')    
plt.tight_layout()
# plt.savefig(pathfig + 'nenvs_vs_avgerr_term.pdf')


plt.figure(figsize=(2.0,1.3))
for uid in np.arange(unit64,nunits-1):
    plt.plot(envs_list, np.log10(anlyzErr.avg_std_term[:,uid]), label=str(units_list[uid]), marker='o')
plt.plot(envs_list, np.log10(np.mean(anlyzErr.avg_std_term[:,unit64:],axis=1)), c='k', marker='o', lw=2.0, ms=2.5)            
plt.plot(envs_list, np.log10(np.mean(anlyzErr.avg_std_term[:,unit64:],axis=1)), c='w', marker='o', lw=0.5)        
plt.ylim([-2.5, 1.5])
plt.legend(frameon=False, bbox_to_anchor=[1, 1], title='ntwk')    
plt.xlabel('num of envs')
plt.ylabel('log (std of error)')
plt.tight_layout()
# plt.savefig(pathfig + 'nenvs_vs_stderr_term.pdf')
# plt.show()






plt.figure(figsize=(2.3,2.0))
# plt.axhline(-1, color='gray', linestyle='--')
# plt.axvline(-1, color='gray', linestyle='--')
plt.plot(np.log10(np.mean(anlyzErr.avg_err_nonterm,axis=1)), np.log10(np.mean(anlyzErr.avg_std_nonterm[:,unit64:],axis=1)), c='k', marker='o', ms=2, label='nenv')        
# plt.plot(np.log10(np.mean(anlyzErr.avg_err_term,axis=1)), np.log10(np.mean(anlyzErr.avg_std_term,axis=1)), c='r', marker='o', label='term')        
envi = 0
plt.annotate('nenv',xy=(np.log10(np.mean(anlyzErr.avg_err_nonterm,axis=1))[envi], np.log10(np.mean(anlyzErr.avg_std_nonterm[:,unit64:],axis=1))[envi]), xytext=(np.log10(np.mean(anlyzErr.avg_err_nonterm,axis=1))[envi]-0.2, np.log10(np.mean(anlyzErr.avg_std_nonterm,axis=1))[envi]+0.5), color='b')
for envi in range(nenvs):
    plt.annotate(str(envs_list[envi]),xy=(np.log10(np.mean(anlyzErr.avg_err_nonterm,axis=1))[envi], np.log10(np.mean(anlyzErr.avg_std_nonterm[:,unit64:],axis=1))[envi]), xytext=(np.log10(np.mean(anlyzErr.avg_err_nonterm,axis=1))[envi]-0.2, np.log10(np.mean(anlyzErr.avg_std_nonterm,axis=1))[envi]+0.1), color='b')
    # plt.annotate(str(envs_list[envi]),xy=(np.log10(np.mean(anlyzErr.avg_err_nonterm,axis=1))[envi], np.log10(np.mean(anlyzErr.avg_std_nonterm,axis=1))[envi]), xytext=(np.log10(np.mean(anlyzErr.avg_err_nonterm,axis=1))[envi], np.log10(np.mean(anlyzErr.avg_std_nonterm,axis=1))[envi]), color='k')
# for envi in range(nenvs):
#     plt.annotate(str(envs_list[envi]),xy=(np.log10(np.mean(anlyzErr.avg_err_term,axis=1))[envi], np.log10(np.mean(anlyzErr.avg_std_term,axis=1))[envi]), xytext=(np.log10(np.mean(anlyzErr.avg_err_term,axis=1))[envi]+0.1, np.log10(np.mean(anlyzErr.avg_std_term,axis=1))[envi]), color='r')
# plt.legend(frameon=False)
# plt.xlim([-3.0, 0.5])
# plt.ylim([-3.0, 0.5])
plt.xlabel('log (mean error)')
plt.ylabel('log (std of error)')
plt.tight_layout()
# plt.savefig(pathfig + 'err_mean_vs_std.pdf')





#----- non-terminal vs terminal states ----#
xmin=-3.5 #-3, -3.5
xmax=1.5 #1.5, 0.4

idline = np.linspace(xmin,xmax,num=10)
plt.figure(figsize=(2.8,1.5))
plt.subplot(121)
plt.plot(np.log10(anlyzErr.avg_err_nonterm).flatten(), np.log10(anlyzErr.avg_err_term).flatten(), c='k', marker='o', mfc='None', linestyle='')
plt.plot(idline, idline, color='gray', linestyle='--')
plt.xlabel('non-terminated')
plt.ylabel('terminated')
plt.xlim([xmin,xmax])
plt.ylim([xmin,xmax])
plt.xticks([-2,0])
plt.yticks([-2,0])
plt.title('Mean error', fontsize=8)
plt.subplot(122)
plt.plot(np.log10(anlyzErr.avg_std_nonterm).flatten(), np.log10(anlyzErr.avg_std_term).flatten(), c='k', marker='o', mfc='None', linestyle='')
plt.plot(idline, idline, color='gray', linestyle='--')
plt.xlabel('non-terminated')
plt.ylabel('terminated')
plt.xlim([xmin,xmax])
plt.ylim([xmin,xmax])
plt.xticks([-2,0])
plt.yticks([-2,0])
plt.title('Std of error', fontsize=8)
plt.tight_layout()
plt.savefig(pathfig + 'term_vs_nonterm.pdf')


#---- reward ----#
plt.figure()
plt.plot(np.log10(anlyzErr.avg_err_nonterm), anlyzErr.reward_time_avg, marker='o', c='k', linestyle='')
# plt.plot(np.log10(anlyzErr.avg_std_nonterm), anlyzErr.reward_time_avg, marker='o', c='k', linestyle='')
plt.xlabel('meta error')
plt.ylabel('reward')
plt.tight_layout()
plt.savefig(pathfig + 'reward_vs_error_lunear.pdf')


tvec = np.arange(ninc) * tinc
uid = 3
plt.figure(figsize=(5.5,1.8))
plt.subplot(141)
dopa_eid = 3
a2c_eid = 0
plt.plot(tvec, anlyzErr.reward_avg[dopa_eid,uid], label='TD net', c='k')
plt.plot(tvec, a2c_reward_avg[a2c_eid], label='A2C', c='r')
plt.legend(frameon=False)
# plt.ylim([0,300])
plt.xlabel('time steps')
plt.ylabel('reward')
plt.title('nenv' + str(envs_list[dopa_eid]), fontsize=8)

plt.subplot(142)
dopa_eid = 4
a2c_eid = 1
plt.plot(tvec, anlyzErr.reward_avg[dopa_eid,uid], label=str(envs_list[dopa_eid]), c='k')
plt.plot(tvec, a2c_reward_avg[a2c_eid], label=str(a2c_env_list[a2c_eid]), c='r')
# plt.ylim([0,300])
plt.title('nenv' + str(envs_list[dopa_eid]), fontsize=8)

plt.subplot(143)
dopa_eid = 5
a2c_eid = 2
plt.plot(tvec, anlyzErr.reward_avg[dopa_eid,uid], label=str(envs_list[dopa_eid]), c='k')
plt.plot(tvec, a2c_reward_avg[a2c_eid], label=str(a2c_env_list[a2c_eid]), c='r')
# plt.ylim([0,300])
plt.title('nenv' + str(envs_list[dopa_eid]), fontsize=8)

plt.subplot(144)
dopa_eid = 6
a2c_eid = 3
plt.plot(tvec, anlyzErr.reward_avg[dopa_eid,uid], label=str(envs_list[dopa_eid]), c='k')
plt.plot(tvec, a2c_reward_avg[a2c_eid], label=str(a2c_env_list[a2c_eid]), c='r')
# plt.ylim([0,300])
plt.title('nenv' + str(envs_list[dopa_eid]), fontsize=8)
plt.tight_layout()
plt.savefig(pathfig + 'reward_vs_time.pdf')















# #----- distribution of std of error ----#
# distribution_std_nonterm = np.std(std_list_nonterm,axis=0)
# distribution_std_term    = np.std(std_list_term,axis=0)


# eid = 1
# plt.figure(figsize=(8,4))
# for uid in range(nunits):
#     plt.subplot(121)
#     plt.plot(units_list[uid]*np.ones(nsims), np.log10(std_list_nonterm[:,eid,uid]), c='k', label='n_envs_' + str(envs_list[eid]), marker='o', linestyle='')
#     plt.plot(units_list[uid], np.log10(np.mean(std_list_nonterm[:,eid,uid])), c='r', label='n_envs_' + str(envs_list[eid]), marker='x', ms=10, linestyle='')
#     plt.xlabel('network size')
#     plt.ylabel('log(Std of error)')
#     # plt.ylim([-3,-1])
# # plt.legend()
# plt.title('Example distr of std of err: nenvs ' + str(envs_list[eid]))
# for eid in range(nenvs):    
#     plt.subplot(122)
#     plt.plot(units_list, np.log10(distribution_std_nonterm[eid,:]), label='n_envs_' + str(envs_list[eid]), marker='o')
#     plt.xlabel('network size')
#     plt.ylabel('log(Std of error)')
# # plt.ylim([-0.005, 0.05])
# plt.legend()    
# plt.title('Distribution of std')
# plt.tight_layout()
# plt.savefig(pathfig + 'distribution_std_nonterm.png')
# plt.show()


# eid = 0
# plt.figure(figsize=(8,4))
# for uid in range(nunits):
#     plt.subplot(121)
#     plt.plot(units_list[uid]*np.ones(nsims), np.log10(std_list_term[:,eid,uid]), c='k', label='n_envs_' + str(envs_list[eid]), marker='o', linestyle='')
#     plt.plot(units_list[uid], np.log10(np.mean(std_list_term[:,eid,uid])), c='r', label='n_envs_' + str(envs_list[eid]), marker='x', ms=10, linestyle='')
#     plt.xlabel('network size')
#     plt.ylabel('log(Std of error)')
#     # plt.ylim([-3,-1])
# # plt.legend()
# plt.title('Example distr of std of err: nenvs ' + str(envs_list[eid]))
# for eid in range(nenvs):    
#     plt.subplot(122)
#     plt.plot(units_list, np.log10(distribution_std_term[eid,:]), label='n_envs_' + str(envs_list[eid]), marker='o')
#     plt.xlabel('network size')
#     plt.ylabel('log(Std of error)')
# # plt.ylim([-0.005, 0.05])
# plt.legend()    
# plt.title('Distribution of std')
# plt.tight_layout()
# plt.savefig(pathfig + 'distribution_std_term.png')
# plt.show()
















# plt.figure(figsize=(4,4))
# for eid in range(nenvs):
#     plt.plot(units_list, np.log10(anlyzErr.avg_std_last30[eid,:]), label='env ' + str(envs_list[eid]), marker='o')
#     plt.xlabel('log10(network size)')
#     plt.ylabel('log10(TD net error)')
# plt.tight_layout()
# plt.legend()
# plt.savefig(pathfig + 'ntwksize_rl_stderr_last30.png')














# plt.figure(figsize=(8,8))
# for eid in range(nenvs):
#     plt.subplot(2,2,eid+1)
#     plt.plot(units_list, np.log10(anlyzErr.avg_err_nonterm[eid,:]), label='mean', marker='o')
#     plt.plot(units_list, np.log10(anlyzErr.avg_std_nonterm[eid,:]), label='std all', marker='o')
#     plt.xlabel('log10(network size)')
#     plt.ylabel('log10(TD net error)')
#     plt.title('env ' + str(envs_list[eid]))
# plt.tight_layout()
# plt.legend()
# plt.savefig(pathfig + 'ntwksize_rl_stderr.png')



# plt.figure(figsize=(15,8))
# for envi in range(4):
#     plt.subplot(4,1,envi+1)
#     idx_nonterm = np.where(np.abs(rlerr.err_nonterm[:,envi]) > 0)[0]
#     idx_term = np.where(np.abs(rlerr.err_term[:,envi]) > 0)[0]
#     plt.plot(idx_nonterm, np.log10(np.abs(rlerr.err_nonterm[idx_nonterm,envi])), marker='.', c='C0', linestyle='', label='non-term state')
#     plt.plot(idx_term, np.log10(np.abs(rlerr.err_term[idx_term,envi])), marker='.', c='C1', linestyle='', label='term state')
#     plt.axhline(-1, color='gray', linestyle='--')
#     plt.axhline(-2, color='gray', linestyle='--')
#     plt.xlabel('time step')
#     plt.ylabel('log10(TD net error)')
#     if envi==0:
#         plt.legend()
# plt.tight_layout()
# plt.savefig(pathfig + 'ntwksize_rlerr_trace.png')





# plt.figure(figsize=(8,8))
# for eid in range(nenvs):    
#     plt.subplot(2,2,eid+1)
#     plt.plot(units_list, np.log10(anlyzErr.avg_err_nonterm[eid,:]), label='nonterm', marker='o')
#     plt.plot(units_list, np.log10(anlyzErr.avg_err_last30[eid,:]), label='last30', marker='o')
#     plt.xlabel('log10(network size)')
#     plt.ylabel('TD net error')
#     plt.title('env ' + str(envs_list[eid]))
# # plt.ylim([-0.005, 0.05])
# plt.tight_layout()
# plt.legend()    
# plt.savefig(pathfig + 'ntwksize_rl_error_last30.png')





















# env0 = 3
# sid0 = 1
# agent0 = 0

# plt.figure(figsize=(10,10))
# for uid in range(nunits):
#     plt.subplot(4,1,uid+1)
#     abs_nonterm = np.abs(rlerr_trace_nonterm[env0][sid0,uid,:,agent0])
#     abs_term    = np.abs(rlerr_trace_term[env0][sid0,uid,:,agent0])
#     idx_nonterm = np.where(abs_nonterm>0)[0]
#     idx_term    = np.where(abs_term>0)[0]
#     plt.plot(idx_nonterm, np.log10(abs_nonterm[idx_nonterm]), marker='.', linestyle='', c='C0')
#     plt.plot(idx_term,    np.log10(abs_term[idx_term]), marker='.', linestyle='', c='C1')
#     # plt.plot(rlerr_trace_term[env0][sid0,uid,:,env0])
#     # plt.ylim([-0.2, 0.2])
#     plt.title('network size ' + str(units_list[uid]) + ', nenvs ' + str(envs_list[env0]))
#     plt.tight_layout()






# # rlerr_trace: nsims x nunits x ntimesteps x nenvs
# avg_rlerr_trace = np.log10(np.mean(rlerr_trace, axis=3))


# env0 = 1
# sid0 = 0
# agent0 = 1
# for uid in range(nunits):
#     plt.figure(figsize=(10,10))
#     for sid0 in range(5):
#         plt.subplot(5,1,sid0+1)
#         plt.plot(np.log10(rlerr_trace[env0][sid0,uid,:,agent0]))
#         plt.axhline(-2, color='gray', linestyle='--')
#         # plt.plot(avg_rlerr_trace[sid0,uid,:])
#         # plt.ylim([-5,2])
#     plt.tight_layout()




# rlerr_trace_nonterm = rlerr_trace - rlerr_trace_term
# avg_rlerr_nonterm = np.log10(np.mean(np.abs(rlerr_trace_nonterm)[:,3,50000:,:]))
# std_rlerr_nonterm = np.log10(np.std(np.abs(rlerr_trace_nonterm)[:,3,50000:,:]))













# eid = 1
# # uid = 0
# # sid = 2

# nsteps = 50000
# metaloss = np.zeros((nsims,nunits,nsteps))
# for sid in range(nsims):
#     # for eid in range(nenvs):
#     for uid in range(nunits):
#     # for eid in range(nenvs):    
#         pathTask        = '/Users/kimchm/Documents/RL/biowulf/'  + env_id_meta  + '_' + env_id_rl + '/' 
#         pathParam       = 'env_' + str(envs_list[eid]) + '_unit_' + str(units_list[uid])
#         pathSim         = '/' + str(sid)
#         pathDA          = pathTask + pathParam + pathSim
#         meta_time     = np.loadtxt(pathDA + '/dopa/' + 'meta_time.txt')
#         meta_lossmeta = np.loadtxt(pathDA + '/dopa/' + 'meta_lossmeta.txt')

#         metaloss[sid,uid,:] = meta_lossmeta



# def movavg(x, wid):
#     nsteps = x.shape[1]
#     xavg   = np.zeros_like(x)
#     for i in range(nsteps):
#         Lidx = np.max([0,i-wid])
#         Ridx = i+1
#         xavg[:,i] = np.mean(x[:,Lidx:Ridx],axis=1)
#     return xavg

# avg_metaloss = np.mean(metaloss,axis=0)

# mvavg_metaloss = movavg(avg_metaloss, wid=1000)



# plt.figure(figsize=(10,10))
# for uid in range(nunits):
#     plt.subplot(4,1,uid+1)
#     plt.plot(meta_time, np.log10(avg_metaloss)[uid,:])
#     plt.plot(meta_time, np.log10(mvavg_metaloss)[uid,:])
#     plt.axhline(-2, color='gray', linestyle='--')
#     plt.title('network size ' + str(units_list[uid]))    
#     plt.ylim([-4,1])
# plt.tight_layout()
# # plt.savefig(pathfig + 'ntwksize_meta_error_trace.png')



# plt.figure(figsize=(10,5))
# for uid in range(nunits):
#     # plt.plot(meta_time, np.log10(avg_metaloss)[uid,:])
#     plt.plot(meta_time, np.log10(mvavg_metaloss)[uid,:])
#     # plt.axhline(-2, color='gray', linestyle='--')
#     plt.ylim([-3,1])
#     plt.title('network size ' + str(units_list[uid]))
# plt.tight_layout()

