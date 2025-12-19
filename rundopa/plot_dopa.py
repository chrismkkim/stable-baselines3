import numpy as np
import results_plotter
import matplotlib.pyplot as plt
import os
from utils import RLerror

topk    = 20
nsims = 20

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
elif meta_train_type == 'Lunar-Lunar':
    env_id_meta = env_lunar
    env_id_rl   = env_lunar
    meta_log    = '2'
elif meta_train_type == 'Cart-Lunar':
    env_id_meta = env_cart
    env_id_rl   = env_lunar
    meta_log    = '1'
elif meta_train_type == 'Lunar-Cart':
    env_id_meta = env_lunar
    env_id_rl   = env_cart
    meta_log    = '1'


# pathA2C = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs/' + env_id_rl + '/'
pathDA        = '/Users/kimchm/Documents/RL/trainedmodel/'  + env_id_meta  + '_' + env_id_rl + '/'
pathDA_all    = [pathDA + '/dopa/' + env_id_rl + '_' + meta_log + '/']
# rew_dopa, time_reward = results_plotter.collect_topk(pathDA_all,  nsims, topk, num_timesteps=None, x_axis="timesteps", frac="all")
metaerr_nonterm = np.load(pathDA + '/dopa/' + 'meta_err_nonterm.npy', allow_pickle=True)
metaerr_term    = np.load(pathDA + '/dopa/' + 'meta_err_term.npy', allow_pickle=True)
meta_values     = np.load(pathDA + '/dopa/' + 'meta_values.npy', allow_pickle=True)
x_adv           = np.load(pathDA + '/dopa/' + 'rl_advantages.npy', allow_pickle=True)
x_dopa          = np.load(pathDA + '/dopa/' + 'rl_dopa.npy', allow_pickle=True)
x_values        = np.load(pathDA + '/dopa/' + 'rl_values.npy', allow_pickle=True)
x_next_values   = np.load(pathDA + '/dopa/' + 'rl_next_values.npy', allow_pickle=True)
x_rewards       = np.load(pathDA + '/dopa/' + 'rl_rewards.npy', allow_pickle=True)
x_raw_rewards   = np.load(pathDA + '/dopa/' + 'rl_raw_rewards.npy', allow_pickle=True)
x_dones         = np.load(pathDA + '/dopa/' + 'rl_dones.npy', allow_pickle=True)

rew_by_envs = results_plotter.collect_agents_reward(pathDA_all)    
    
rlerr = RLerror(x_adv, x_dopa, x_values, x_next_values, x_rewards, x_raw_rewards, x_dones)





plt.figure(figsize=(10,10))
for envi in range(10):
    plt.subplot(5,2,envi+1)
    plt.axhline(0, color='gray', linestyle='--')
    plt.plot(np.cumsum(rew_by_envs[envi][1]), rew_by_envs[envi][0], marker='o')
    plt.xlabel('time steps')
    plt.ylabel('reward')
    plt.ylim([-300,300])
    plt.title(f'env{envi}')
plt.tight_layout()



def mvavg(x,wid):
    xavg = np.zeros_like(x)
    tlen = x.shape[0]
    for t in range(tlen):
        Lidx = np.max([0,t-wid])
        Ridx = np.min([tlen-1,t+wid])
        xavg[t] = np.mean(x[Lidx:Ridx])
    return xavg    
        
nreset = 10
tmp = np.mean(np.abs(metaerr_nonterm),axis=1)
metaerr_split = np.array_split(tmp, nreset)
metaerr_avg = np.array([split.mean() for split in metaerr_split])
metaerr_std = np.array([split.std() for split in metaerr_split])



plt.figure()
plt.plot(np.log10(metaerr_avg))
plt.plot(np.log10(metaerr_std))
plt.tight_layout()



meta_values_split = np.array_split(meta_values, nreset, axis=0)
meta_values_avg = np.array([split.mean(axis=0) for split in meta_values_split])
meta_values_std = np.array([split.std(axis=0) for split in meta_values_split])



plt.figure()
for agenti in range(10):
    plt.plot(meta_values_avg[:,agenti])
plt.tight_layout()


plt.figure()
for agenti in range(10):
    plt.plot(meta_values_std[:,agenti])
plt.plot(np.mean(meta_values_std,axis=1), c='k', lw=4, marker='o')
plt.tight_layout()


# plt.figure()
# # for i in np.arange(5,10):
# plt.hist(meta_values_split[6].flatten(), bins=100, range=(-200,100), histtype='step')
# plt.hist(meta_values_split[9].flatten(), bins=100, range=(-200,100), histtype='step')
# plt.tight_layout()



plt.figure()
plt.subplot(211)
plt.plot(np.mean(meta_values_avg,axis=1))
plt.subplot(212)
plt.plot(np.mean(meta_values_std,axis=1))
plt.tight_layout()



# plt.figure()
# for agenti in range(6):
#     plt.subplot(3,2,agenti+1)
#     plt.plot(meta_values[:,agenti])
#     avg_values = mvavg(meta_values[:,agenti],wid=250)
#     plt.plot(avg_values)
# plt.tight_layout()







agenti = 0
tidx = np.where(np.abs(metaerr_term[:,agenti]) > 0)[0]
plt.figure()
plt.plot(np.log10(np.abs(metaerr_nonterm[:,0])))
plt.plot(tidx, np.log10(np.abs(metaerr_term[tidx,agenti])),marker='.', linestyle='')
plt.tight_layout()


plt.figure()
for ix, agenti in enumerate(np.arange(5,9)):
    tidx = np.where(np.abs(rlerr.err_term[:,agenti]) > 0)[0]
    plt.subplot(2,2,ix+1)
    plt.plot(np.log10(np.abs(rlerr.err_nonterm[:,agenti])))
    plt.plot(tidx, np.log10(np.abs(rlerr.err_term[tidx,agenti])),marker='.', linestyle='')
plt.tight_layout()





plt.figure(figsize=(15,8))
for envi in range(5):
    plt.subplot(4,2,envi+1)
    idx_nonterm = np.where(np.abs(rlerr.err_nonterm[:,envi]) > 0)[0]
    idx_term = np.where(np.abs(rlerr.err_term[:,envi]) > 0)[0]
    plt.plot(idx_nonterm, np.log10(np.abs(rlerr.err_nonterm[idx_nonterm,envi])), marker='.', c='C0', linestyle='', label='non-term state')
    plt.plot(idx_term, np.log10(np.abs(rlerr.err_term[idx_term,envi])), marker='.', c='C1', linestyle='', label='term state')
    plt.axhline(-1, color='gray', linestyle='--')
    plt.axhline(-2, color='gray', linestyle='--')
    plt.xlabel('time step')
    plt.ylabel('log10(TD net error)')
    if envi==0:
        plt.legend()
plt.tight_layout()


idx_nonterm = np.where(np.abs(rlerr.err_nonterm.flatten()) > 0)[0]
idx_term = np.where(np.abs(rlerr.err_term.flatten()) > 0)[0]
plt.figure()
plt.hist(np.log10(np.abs(rlerr.err_nonterm.flatten())[idx_nonterm]), bins=100, range=(-6,0), color='C0', histtype='step', density=True, label='non-termimal state')
plt.hist(np.log10(np.abs(rlerr.err_term.flatten())[idx_term]), bins=100, range=(-6,0), color='C1', histtype='step', density=True, label='termimal state')
plt.xlabel('log10(TD net error)')
plt.ylabel('density')
plt.legend()
plt.tight_layout()



envi = 1
n_epi_envi = len(rlerr.epis_err[envi])
cnt=0
epi=0
plt.figure(figsize=(10,10))
while (cnt < 25) and (epi < n_epi_envi):
    if len(rlerr.epis_err[envi][epi]) > 100:
        plt.subplot(5,5,cnt+1)
        plt.plot(np.log10(np.abs(rlerr.epis_err[envi][epi])))
        # plt.xlim([0,510])
        plt.ylim([-5,1])
        plt.title(f'episode{epi}')
        plt.xlabel('time')
        plt.ylabel('log10 TD net err')
        cnt+=1
    epi+=1
plt.tight_layout()
    
    
    


# meta error
metaerr_nonterm[np.abs(metaerr_nonterm) == 0] = np.nan
trace1 = metaerr_nonterm[:,0]
tix    = np.where(np.isnan(trace1)==True)[0]

idxnan = np.where(np.isnan(metaerr_nonterm)==True)
avg_metaerr_nonterm = np.nanmean(metaerr_nonterm, axis=1)

plt.figure(figsize=(10,3))
plt.plot(np.log10(np.abs(avg_metaerr_nonterm)))
plt.tight_layout()


plt.figure()
plt.plot(np.log10(np.abs(trace1)))
plt.axhline(-2, color='gray', linestyle='--')
# for i in tix:
#     plt.axvline(i, color='gray')
plt.tight_layout()









# """
# Compare non-terminal and terminated errors
#     * This plot has interesting feature
# """
# plt.figure(figsize=(15,10))
# for i in range(5):
#     plt.subplot(5,1,i+1)
#     plt.plot(err_nonterm[:,i])
#     plt.plot(err_term[:,i])
#     plt.ylim([-0.1,0.15])
# plt.tight_layout()
# # plt.savefig(path + meta_train_type + '_relu_loss.png', dpi=300)
# # plt.show()




# agenti = 2
# plt.figure()
# ax1 = plt.subplot(211)
# # plt.plot(delta_flip[:,agenti])
# plt.plot(delta[:,agenti], marker='.')
# plt.plot(delta_flip2[:,agenti], marker='.')

# plt.subplot(212, sharex=ax1)
# plt.plot(delta_flip1[:,agenti], marker='.')
# plt.plot(delta_flip2[:,agenti], marker='.')
# plt.tight_layout()





# agents = results_plotter.collect_agents_reward(pathDA_all)
# nagents = len(agents)



# def plot_nonzeros(i, idx, err, clr):
#     num = np.sum(idx)
#     if num > 0:
#         plt.plot(i*np.ones(num), err[idx], marker='.', color=clr, linestyle='')
        
# def remove_large_elt(x):        
#     idx = np.abs(x) > 1
#     x[idx] = 0.0
#     return x
    
# def remove_zeros(x, start_t):
#     x = x[start_t:]
#     idx = np.abs(x) > 0
#     xall = x[idx]
#     return xall

# def movavg(x, wid):
#     nsteps = x.shape[0]
#     xavg   = np.zeros(nsteps)
#     for i in range(nsteps):
#         Lidx = np.max([0,i-wid])
#         Ridx = i+1
#         xavg[i] = np.mean(x[Lidx:Ridx])
#     return xavg

# """
# Check truncated states: This is the simplest code that compare RL Zoo and my rewards
#     * The order in which RL Zoo agent files are loaded are somewhat random.
#     *  - trunc_states contains correctly ordered agents
#     *  - agents contains randomly ordered agents
# """    
# t1 = []
# t2 = []
# trunc_states = (x_rewards > 1.0).astype(float)
# for i in range(nagents):
#     idx1 = np.where(trunc_states[:,i] > 0)[0]
#     idx2 = np.cumsum(agents[i][0,:])[agents[i][1,:]==500]
#     if len(idx1)>0:
#         t1.append(idx1)
#     if len(idx2) > 0:
#         t2.append(idx2-1)



    



# agenti = 1
# n_epi_agenti = len(episodes[agenti])
# plt.figure(figsize=(10,5))
# for epi in range(n_epi_agenti):    
#     if len(episodes[agenti][epi]) > 50:
#         plt.plot(episodes[agenti][epi])
# plt.ylim([-0.1,0.1])
# plt.tight_layout()
    
    
    
# start_epi = 10
# plt.figure(figsize=(10,5))
# plt.subplot(121)
# plt.title('non-terminal states')
# for agenti in range(nagents):
#     plt.plot(np.log10(episodes_length[agenti][start_epi:]), np.log10(np.abs(episodes_nonterm_err[agenti]))[start_epi:], c='k', marker='.', linestyle='')
# plt.ylim([-5,1])

# plt.subplot(122)    
# plt.title('terminated states')
# for agenti in range(nagents):
#     plt.plot(np.log10(episodes_length[agenti][start_epi:]), np.log10(np.abs(episodes_term_err[agenti]))[start_epi:], c='k', marker='.', linestyle='')
# plt.ylim([-5,1])
# plt.tight_layout()








# """
# Show non-terminal, terminated, truncated errors
# """
# plt.figure(figsize=(10,7))
# plt.subplot(311)
# plt.title('non terminal')
# for i in range(nsteps):
#     idx_non = err_nonterm[i] != 0
#     plot_nonzeros(i, idx_non, err_nonterm[i], 'k')
# plt.ylim([-0.1,0.1])

# plt.subplot(312)
# plt.title('terminated')
# for i in range(nsteps):
#     idx_trm = err_term[i] != 0
#     plot_nonzeros(i, idx_trm, err_term[i], 'b')
# plt.ylim([-0.1,0.1])

# plt.subplot(313)
# plt.title('truncated')
# for i in range(nsteps):
#     idx_trc = err_trunc[i] != 0
#     plot_nonzeros(i, idx_trc, err_trunc[i], 'r')
# plt.ylim([-0.1,0.1])
# plt.tight_layout()
# # plt.savefig(path + 'meta_loss_N256.png', dpi=300)





# #-----------------------------------------------



# rew_mean = np.mean(rew_dopa, axis=0)
# rew_avg = movavg(rew_mean, 10)
# log_meta_loss = np.log10(np.sqrt(meta_lossmeta))
# log_meta_loss_avg = movavg(log_meta_loss, 500)
# log_rl_loss = np.log10(np.sqrt(rl_lossmeta))
# log_rl_loss_avg = movavg(log_rl_loss, 500)


# meta_timedone = np.where(meta_done > 0)[0]
# rl_timedone   = np.where(rl_done > 0)[0]


# plt.figure(figsize=(10,8))
# plt.subplot(311)
# plt.plot(rew_mean, label='TDnet')
# plt.plot(rew_avg)
# plt.legend()
# plt.xlabel('episodes')
# plt.ylabel('reward')

# plt.subplot(312)
# plt.plot(meta_time, log_meta_loss)
# plt.plot(meta_time, log_meta_loss_avg)
# plt.xlabel('steps')
# plt.ylabel('meta loss')

# plt.subplot(313)
# plt.plot(rl_time, log_rl_loss)
# plt.plot(rl_time, log_rl_loss_avg)
# # plt.plot(time, rl_done, marker='.', linestyle='')
# plt.xlabel('steps')
# plt.ylabel('rl loss')

# plt.tight_layout()
# # plt.savefig(path + meta_train_type + '_relu.png', dpi=300)
# # plt.savefig(path + meta_train_type + '_tanh.png', dpi=300)
# plt.show()





# plt.figure(figsize=(10,8))
# plt.subplot(211)
# plt.plot(meta_time, log_meta_loss)
# # plt.plot(meta_time, meta_done, marker='.', linestyle='')
# for i in meta_timedone:
#     plt.axvline(meta_time[i], meta_done[i], color='gray', alpha=0.5)
# plt.xlabel('steps')
# plt.ylabel('meta loss')
# plt.subplot(212)
# plt.plot(rl_time, log_rl_loss)
# # plt.plot(time, rl_done, marker='.', linestyle='')
# for i in rl_timedone:
#     plt.axvline(rl_time[i], rl_done[i], color='gray', alpha=0.5)
# plt.xlabel('steps')
# plt.ylabel('rl loss')
# plt.tight_layout()
# # plt.savefig(path + meta_train_type + '_relu_loss.png', dpi=300)
# plt.show()






x=1

