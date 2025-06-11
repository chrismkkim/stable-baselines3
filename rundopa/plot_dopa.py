import numpy as np
import results_plotter
import matplotlib.pyplot as plt
import os


topk    = 25
nagents = 50

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
rew_dopa      = results_plotter.collect_topk(pathDA_all,  nagents, topk, num_timesteps=None, x_axis="timesteps", frac="all")
time          = np.loadtxt(pathDA + '1' + '/dopa/' + 'meta_time.txt')
meta_lossmeta = np.loadtxt(pathDA + '1' + '/dopa/' + 'meta_lossmeta.txt')
rl_lossmeta   = np.loadtxt(pathDA + '1' + '/dopa/' + 'rl_lossmeta.txt')
rl_done       = np.loadtxt(pathDA + '1' + '/dopa/' + 'rl_done.txt')
meta_done     = np.loadtxt(pathDA + '1' + '/dopa/' + 'meta_done.txt')

# pathA2c_all = [pathA2C + str(i+1) + '/a2c/'  + env_id_rl + '_' + '1'      + '/' for i in range(nagents)]
# rew_a2c               = results_plotter.collect_topk(pathA2c_all, nagents, topk, num_timesteps=None, x_axis="timesteps", frac="all")

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

path = '/Users/kimchm/Documents/GitHub/stable-baselines3/rundopa/figure/'

meta_timedone = np.where(meta_done > 0)[0]
rl_timedone   = np.where(rl_done > 0)[0]

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.plot(time, log_meta_loss)
plt.plot(time, meta_done, marker='.', linestyle='')
for i in meta_timedone:
    plt.axvline(time[i], meta_done[i], color='gray', alpha=0.5)
plt.subplot(212)
plt.plot(time, log_rl_loss)
# plt.plot(time, rl_done, marker='.', linestyle='')
for i in rl_timedone:
    plt.axvline(time[i], rl_done[i], color='gray', alpha=0.5)
plt.xlabel('steps')
plt.ylabel('rl loss')
plt.tight_layout()
# plt.savefig(path + meta_train_type + '_relu_loss.png', dpi=300)
plt.show()



plt.figure(figsize=(10,8))
# plt.title(meta_train_type)

plt.subplot(311)
plt.plot(rew_mean, label='TDnet')
plt.plot(rew_avg)
plt.legend()
plt.xlabel('episodes')
plt.ylabel('reward')

plt.subplot(312)
plt.plot(time, log_meta_loss)
plt.plot(time, log_meta_loss_avg)
plt.xlabel('steps')
plt.ylabel('meta loss')

plt.subplot(313)
plt.plot(time, log_rl_loss)
plt.plot(time, log_rl_loss_avg)
# plt.plot(time, rl_done, marker='.', linestyle='')
plt.xlabel('steps')
plt.ylabel('rl loss')

plt.tight_layout()
# plt.savefig(path + meta_train_type + '_relu.png', dpi=300)
# plt.savefig(path + meta_train_type + '_tanh.png', dpi=300)
plt.show()




x=1

