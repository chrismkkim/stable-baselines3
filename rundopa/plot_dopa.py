import numpy as np
import results_plotter
import matplotlib.pyplot as plt
import os


topk    = 20
nagents = 20

env_cart  = 'CartPole-v1'
env_lunar = 'LunarLander-v3'

meta_train_type = 'Cart-Cart'
# meta_train_type = 'Lunar-Lunar'
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
pathDA_all    = [pathDA  + str(i+1) + '/dopa/' + env_id_rl + '_' + meta_log + '/' for i in range(nagents)]
rew_dopa, time_reward = results_plotter.collect_topk(pathDA_all,  nagents, topk, num_timesteps=None, x_axis="timesteps", frac="all")
meta_time     = np.loadtxt(pathDA + '1' + '/dopa/' + 'meta_time.txt')
meta_lossmeta = np.loadtxt(pathDA + '1' + '/dopa/' + 'meta_lossmeta.txt')
meta_done     = np.loadtxt(pathDA + '1' + '/dopa/' + 'meta_done.txt')
rl_time       = np.loadtxt(pathDA + '1' + '/dopa/' + 'rl_time.txt')
rl_lossmeta   = np.loadtxt(pathDA + '1' + '/dopa/' + 'rl_lossmeta.txt')
rl_done       = np.loadtxt(pathDA + '1' + '/dopa/' + 'rl_done.txt')

x_adv = np.load(pathDA + '1' + '/dopa/' + 'rl_advantages.npy', allow_pickle=True)
x_dopa = np.load(pathDA + '1' + '/dopa/' + 'rl_dopa.npy', allow_pickle=True)
x_values = np.load(pathDA + '1' + '/dopa/' + 'rl_values.npy', allow_pickle=True)
x_next_values = np.load(pathDA + '1' + '/dopa/' + 'rl_next_values.npy', allow_pickle=True)
x_rewards = np.load(pathDA + '1' + '/dopa/' + 'rl_rewards.npy', allow_pickle=True)
x_raw_rewards = np.load(pathDA + '1' + '/dopa/' + 'rl_raw_rewards.npy', allow_pickle=True)
x_dones = np.load(pathDA + '1' + '/dopa/' + 'rl_dones.npy', allow_pickle=True)


agents = results_plotter.collect_agents(pathDA_all)



def plot_nonzeros(i, idx, err, clr):
    num = np.sum(idx)
    if num > 0:
        plt.plot(i*np.ones(num), err[idx], marker='.', color=clr, linestyle='')
        
def remove_large_elt(x):        
    idx = np.abs(x) > 1
    x[idx] = 0.0
    return x
    
def remove_zeros(x, start_t):
    x = x[start_t:]
    idx = np.abs(x) > 0
    xall = x[idx]
    return xall
    
    
x_nonterm = 1 - x_dones
x_trunc = (x_rewards != x_raw_rewards).astype(float)
x_term = (1 - x_trunc) * x_dones

x_trunc_confirm = x_trunc * x_dones
assert np.all(x_trunc_confirm == x_trunc)
assert np.sum(x_trunc) == np.sum(x_trunc_confirm)


err = (x_dopa - x_adv) / x_values
err_nonterm = remove_large_elt(err * x_nonterm)
err_trunc = remove_large_elt(err * x_trunc)
err_term  = remove_large_elt(err * x_term)
nsteps = x_dones.shape[0]

err_nonterm_ = remove_zeros(err_nonterm, 2000)
err_trunc_   = remove_zeros(err_trunc, 2000)
err_term_    = remove_zeros(err_term, 2000)


    
    
"""
Check truncated states: This is the simplest code that compare RL Zoo and my rewards
    * The order in which RL Zoo agent files are loaded are somewhat random.
    *  - trunc_states contains correctly ordered agents
    *  - agents contains randomly ordered agents
"""    
t1 = []
t2 = []
trunc_states = (x_rewards > 1.0).astype(float)
for i in range(20):
    idx1 = np.where(trunc_states[:,i] > 0)[0]
    idx2 = np.cumsum(agents[i][0,:])[agents[i][1,:]==500]
    if len(idx1)>0:
        t1.append(idx1)
    if len(idx2) > 0:
        t2.append(idx2-1)


path = '/Users/kimchm/Documents/GitHub/stable-baselines3/rundopa/figure/'

"""
Show non-terminal, terminated, truncated errors
"""
plt.figure(figsize=(10,7))
plt.subplot(311)
plt.title('non terminal')
for i in range(nsteps):
    idx_non = err_nonterm[i] != 0
    plot_nonzeros(i, idx_non, err_nonterm[i], 'k')
plt.ylim([-0.1,0.1])

plt.subplot(312)
plt.title('terminated')
for i in range(nsteps):
    idx_trm = err_term[i] != 0
    plot_nonzeros(i, idx_trm, err_term[i], 'b')
plt.ylim([-0.1,0.1])

plt.subplot(313)
plt.title('truncated')
for i in range(nsteps):
    idx_trc = err_trunc[i] != 0
    plot_nonzeros(i, idx_trc, err_trunc[i], 'r')
plt.ylim([-0.1,0.1])
plt.tight_layout()
plt.savefig(path + 'meta_loss_N256.png', dpi=300)


"""
Compare non-terminal and terminated errors
    * This plot has interesting features
"""
plt.figure(figsize=(15,10))
for i in range(5):
    plt.subplot(5,1,i+1)
    plt.plot(err_nonterm[:,i+10])
    plt.plot(err_term[:,i+10])
    plt.ylim([-0.1,0.1])
plt.tight_layout()
# plt.savefig(path + meta_train_type + '_relu_loss.png', dpi=300)
# plt.show()



plt.figure(figsize=(10,8))
for i in range(4):
    plt.subplot(4,1,i+1)
    nsteps = err_nonterm.shape[0]
    for j in range(nsteps):
        if np.abs(err_nonterm[j,i+4]) > 0:
            plt.plot(j, np.log10(np.abs(err_nonterm[j,i+4])), marker='.', c='C0', linestyle='')
    for j in range(nsteps):
        if np.abs(err_term[j,i+4]) > 0:
            plt.plot(j, np.log10(np.abs(err_term[j,i+4])), marker='.', c='C1', linestyle='')
    plt.axhline(-1, color='gray', linestyle='--')
    plt.axhline(-2, color='gray', linestyle='--')
    # plt.ylim([-0.1,0.1])
plt.tight_layout()
# plt.savefig(path + meta_train_type + '_relu_loss.png', dpi=300)
# plt.show()





# plt.figure()
# plt.subplot(311)
# plt.hist(err_nonterm_, bins=100, color='k', density=True, histtype='step')
# plt.subplot(312)
# plt.hist(err_term_, bins=100, color='b', density=True, histtype='step')
# plt.subplot(313)
# plt.hist(err_trunc_, bins=100, color='r', density=True, histtype='step')
# plt.tight_layout()


#-----------------------------------------------


def movavg(x, wid):
    nsteps = x.shape[0]
    xavg   = np.zeros(nsteps)
    for i in range(nsteps):
        Lidx = np.max([0,i-wid])
        Ridx = i+1
        xavg[i] = np.mean(x[Lidx:Ridx])
    return xavg

rew_mean = np.mean(rew_dopa, axis=0)
rew_avg = movavg(rew_mean, 10)
log_meta_loss = np.log10(np.sqrt(meta_lossmeta))
log_meta_loss_avg = movavg(log_meta_loss, 500)
log_rl_loss = np.log10(np.sqrt(rl_lossmeta))
log_rl_loss_avg = movavg(log_rl_loss, 500)


meta_timedone = np.where(meta_done > 0)[0]
rl_timedone   = np.where(rl_done > 0)[0]



plt.figure(figsize=(10,8))
# plt.title(meta_train_type)

plt.subplot(311)
plt.plot(rew_mean, label='TDnet')
plt.plot(rew_avg)
plt.legend()
plt.xlabel('episodes')
plt.ylabel('reward')

plt.subplot(312)
plt.plot(meta_time, log_meta_loss)
plt.plot(meta_time, log_meta_loss_avg)
plt.xlabel('steps')
plt.ylabel('meta loss')

plt.subplot(313)
plt.plot(rl_time, log_rl_loss)
plt.plot(rl_time, log_rl_loss_avg)
# plt.plot(time, rl_done, marker='.', linestyle='')
plt.xlabel('steps')
plt.ylabel('rl loss')

plt.tight_layout()
# plt.savefig(path + meta_train_type + '_relu.png', dpi=300)
# plt.savefig(path + meta_train_type + '_tanh.png', dpi=300)
plt.show()



plt.figure(figsize=(10,8))
plt.subplot(211)
plt.plot(meta_time, log_meta_loss)
# plt.plot(meta_time, meta_done, marker='.', linestyle='')
for i in meta_timedone:
    plt.axvline(meta_time[i], meta_done[i], color='gray', alpha=0.5)
plt.xlabel('steps')
plt.ylabel('meta loss')
plt.subplot(212)
plt.plot(rl_time, log_rl_loss)
# plt.plot(time, rl_done, marker='.', linestyle='')
for i in rl_timedone:
    plt.axvline(rl_time[i], rl_done[i], color='gray', alpha=0.5)
plt.xlabel('steps')
plt.ylabel('rl loss')
plt.tight_layout()
# plt.savefig(path + meta_train_type + '_relu_loss.png', dpi=300)
plt.show()






x=1

