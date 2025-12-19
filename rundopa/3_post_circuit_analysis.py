import numpy as np
import torch as th
import torch.nn as nn
import results_plotter
import matplotlib.pyplot as plt
import os
from utils import A2CReward
import Plots.a2c_reward
import importlib
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import Dopa
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import utils
from sklearn.linear_model import LinearRegression

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

plt.style.use(os.path.join(os.path.dirname(__file__), 'pyplot_setting.mplstyle'))

nenv               = 50
env_id_meta        = "LunarLander-v3" #"LunarLander-v3"
env_id_rl          = "LunarLander-v3" #CartPole-v1
eval_env           = gym.make(env_id_rl)

pathfig = '/Users/kimchm/Documents/GitHub/stable-baselines3/rundopa/figure/'
path               = '/Users/kimchm/Documents/RL/biowulf/lunar/tdnet/circuit/sim/'
# path_envs          = env_id_rl
path_to_log        = path + env_id_rl + '/'

# 500000

dopa_kwargs = {
    "policy":            "MlpPolicy",
    "n_envs":            nenv,
    "env":               env_id_rl,
    "verbose":           0,
    "gae_lambda":        0.0,
    "n_timesteps":       250000,
    "n_steps":           1,
    "n_meta_steps":      1,
    "learning_rate":     7e-4,
    "learning_rate_dopa":1e-4,
    # Dopa‐specific custom args:
    "net_arch":          {"pi": [64, 64], "vf": [64, 64], "re":[64,64], "td":[64,64,64,1]},
    "normalize_values":  False,
    "tracker_window_size": 1000,
    "train_envs":        {"meta":env_id_meta, "rl":env_id_rl},
    "log_path":          path_to_log,
    "traintype_meta":    False
}
path_model = '/Users/kimchm/Documents/RL/biowulf/lunar/tdnet/circuit/LunarLander_best_model/0/dopa/LunarLander-v3_2/best_model.zip'
model_dopa_dummy         = Dopa(**dopa_kwargs)
model_dopa = model_dopa_dummy.load(path_model)

dopa_env       = make_vec_env(env_id_rl, n_envs=nenv)
model_dopa.env = dopa_env
model_dopa.log_path = path_to_log
model_dopa.tensorboard_log = path_to_log
model_dopa._logger = utils.configure_logger(model_dopa.verbose, model_dopa.tensorboard_log, reset_num_timesteps=True)

model_dopa.simulate(model_dopa.n_timesteps)

# video of trained agents
dopa_env       = make_vec_env(env_id_meta, n_envs=nenv)
dopa_kwargs["env"] = dopa_env
dopa_kwargs["n_timesteps"] = 5000
mean_reward_dopa, std_reward_dopa, all_reward_dopa = evaluate_policy(model_dopa, eval_env, n_eval_episodes=50, return_episode_rewards=False)
obs = dopa_env.reset()
# for i in range(1000):
#     action, _state = model_dopa.predict(obs, deterministic=True)
#     obs, reward, done, info = dopa_env.step(action)
#     dopa_env.render("human")
    
    


# path_data = '/Users/kimchm/Documents/RL/biowulf/lunar/tdnet/circuit/rl_files/0/dopa/'
sim_raw_rewards = th.from_numpy(np.load(path_to_log + 'sim_raw_rewards.npy')).to(th.float32)
sim_rewards = th.from_numpy(np.load(path_to_log + 'sim_rewards.npy')).to(th.float32)
sim_values = th.from_numpy(np.load(path_to_log + 'sim_values.npy')).to(th.float32)
sim_next_values = th.from_numpy(np.load(path_to_log + 'sim_next_values.npy')).to(th.float32)
sim_dones = th.from_numpy(np.load(path_to_log + 'sim_dones.npy')).to(th.float32)
sim_dopa = th.from_numpy(np.load(path_to_log + 'sim_dopa.npy')).to(th.float32)




td_net = model_dopa.policy.mlp_extractor.td_net

nenv  = model_dopa.n_envs
nstep = sim_rewards.shape[0] - 1
nunit = td_net[0].out_features
nlayer_total = len(td_net)
nlayer = int((nlayer_total-1) / 2)
act = th.zeros(nenv,nunit,nlayer,nstep)
preAct = th.zeros(nenv,nunit,nlayer,nstep)
out = th.zeros(nenv,nstep)

for tix, ti in enumerate(np.arange(1,nstep+1)):       
    
    _rewards_t     = sim_rewards[ti].reshape(-1,1)
    _raw_rewards_t = sim_raw_rewards[ti].reshape(-1,1)
    _next_values_t = sim_next_values[ti].reshape(-1,1)
    _values_t      = sim_values[ti].reshape(-1,1)
    _dones_t       = sim_dones[ti].reshape(-1,1)
    _dopa_t        = sim_dopa[ti]

    trunc_idx                     = th.where(_raw_rewards_t != _rewards_t)[0]
    _next_values_t_mod            = _next_values_t.clone()
    _next_values_t_mod[trunc_idx] = (_rewards_t[trunc_idx] - _raw_rewards_t[trunc_idx]) / model_dopa.gamma
    _dones_t_mod                  = _dones_t.clone()
    _dones_t_mod[trunc_idx]       = th.tensor(1) - _dones_t[trunc_idx]

    _input_t = th.hstack([_raw_rewards_t, _next_values_t_mod, _values_t, th.tensor(1) - _dones_t_mod])
    dopa_t   = (td_net(_input_t)).flatten()
        
    x = _input_t.clone()
    for layer in range(nlayer_total):
        x = td_net[layer](x)        
        if isinstance(td_net[layer], nn.ReLU):
            layeri = int((layer-1) / 2)
            act[:,:,layeri,tix] = x
        elif layer < nlayer_total - 1:
            layeri = int(layer / 2)
            preAct[:,:,layeri,tix] = x
            
    out[:,tix] = x.flatten()
        
    assert th.all(th.isclose(dopa_t, _dopa_t, rtol=0, atol=1e-4))            
assert th.all(th.isclose(out, sim_dopa[1:,:].T, rtol=0, atol=1e-4))



#-------- linear regression -------#

# linear regression
r2 = np.zeros((nenv,nlayer,nunit))
coef = np.zeros((nenv,nlayer,nunit,3))
r2_dopa = np.zeros(nenv)

# envi = 2

tlim = 9900
actLin = np.zeros((nlayer,tlim,nenv,nunit))
actTlim = np.zeros((nenv,nunit,nlayer,tlim))
preActTlim = np.zeros((nenv,nunit,nlayer,tlim))
sim_dopa_notdones = np.zeros((tlim,nenv))
for envi in range(nenv):
    '''
    Exclude dones. If included, the linear approximation becomes worse.
    '''
    _notdones = ~(sim_dones[1:,envi] == 1)
    _rew = sim_raw_rewards[1:,envi][_notdones].detach().numpy().reshape(-1,1)
    _nval = sim_next_values[1:,envi][_notdones].detach().numpy().reshape(-1,1)
    _val = sim_values[1:,envi][_notdones].detach().numpy().reshape(-1,1)
    _variables = np.hstack([_rew, _nval, _val])

    sim_dopa_envi_notdones = sim_dopa[1:,envi][_notdones]
    
    linreg_dopa = LinearRegression(fit_intercept=False)
    linreg_dopa.fit(_variables, sim_dopa_envi_notdones)
    r2_dopa[envi] = linreg_dopa.score(_variables, sim_dopa_envi_notdones)
    for layer in range(nlayer):
        for unit in range(nunit):
            _act = act[envi,unit,layer,:][_notdones].detach().numpy()
            _preAct = preAct[envi,unit,layer,:][_notdones].detach().numpy()
            
            linreg = LinearRegression(fit_intercept=True)
            linreg.fit(_variables, _act)    
            r2[envi,layer,unit] = linreg.score(_variables, _act)    
            coef[envi,layer,unit,:] = linreg.coef_                    
            actLin[layer,:,envi,unit] = linreg.predict(_variables)[:tlim]
            actTlim[envi,unit,layer,:] = _act[:tlim]
            preActTlim[envi,unit,layer,:] = _preAct[:tlim]
    sim_dopa_notdones[:,envi] = sim_dopa_envi_notdones[:tlim]
                
                
tdnet_lastlayer_wgt = td_net[6].weight.detach().numpy().flatten()

layer0 = 0
layer1 = 1
layer2 = 2

relu = nn.ReLU()
actProp_layer1 = td_net[3](td_net[2](th.tensor(actLin[layer0]).to(th.float32))).detach().numpy()
actProp_layer2 = td_net[5](td_net[4](th.tensor(actLin[layer1]).to(th.float32))).detach().numpy()
actProp_output =      td_net[6](th.tensor(actLin[layer2]).to(th.float32)).detach().numpy()
preActProp_layer1 = (td_net[2](th.tensor(actLin[layer0]).to(th.float32))).detach().numpy()
preActProp_layer2 = (td_net[4](th.tensor(actLin[layer1]).to(th.float32))).detach().numpy()


err_layer1 = np.zeros((nenv,nunit))
err_layer2 = np.zeros((nenv,nunit))
err_output = np.zeros(nenv)
std_layer1 = np.zeros((nenv,nunit))
std_layer2 = np.zeros((nenv,nunit))
cor_layer1 = np.zeros((nenv,nunit))
cor_layer2 = np.zeros((nenv,nunit))
for envi in range(nenv):
    for uniti in range(nunit):
        std1 = np.std(preActTlim[envi,uniti,layer1,:])
        err_layer1[envi,uniti] = np.mean(np.abs(preActTlim[envi,uniti,layer1,:] - preActProp_layer1[:,envi,uniti])) / std1
        cor_layer1[envi,uniti] = np.corrcoef(preActTlim[envi,uniti,layer1,:],preActProp_layer1[:,envi,uniti])[0,1]

        std2 = np.std(preActTlim[envi,uniti,layer2,:])
        err_layer2[envi,uniti] = np.mean(np.abs(preActTlim[envi,uniti,layer2,:] - preActProp_layer2[:,envi,uniti])) / std2
        cor_layer2[envi,uniti] = np.corrcoef(preActTlim[envi,uniti,layer2,:],preActProp_layer2[:,envi,uniti])[0,1]
            
    stdout = np.std(sim_dopa_notdones[:,envi])
    err_output[envi] = np.sqrt(np.mean((sim_dopa_notdones[:,envi] - actProp_output[:,envi,0])**2)) / stdout



#----------- linearly approximate each neuron's pre-activation ----------#
# hidden layers
plt.figure(figsize=(2.3,1.5))
plt.hist(cor_layer1.flatten(), bins=20, range=(0,1), histtype='step', label='layer1', color='C0')
plt.hist(cor_layer2.flatten(), bins=20, range=(0,1), histtype='step', label='layer2', color='C1')
plt.axvline(np.mean(cor_layer1), color='C0', linestyle='--')
plt.axvline(np.mean(cor_layer2), color='C1', linestyle='--')
plt.legend(frameon=False, bbox_to_anchor=[1,1])
plt.xlabel('approx. accuracy')
plt.ylabel('neuron count')
plt.tight_layout()

# output layer (perfect match)
idline = np.arange(-50,50)
plt.figure()
for envi in range(nenv):
    plt.plot(sim_dopa_notdones[:,envi], actProp_output[:,envi,0], marker='.', linestyle='', c='k', alpha=0.2)
plt.plot(idline, idline, c='gray', linestyle='--')
plt.xlabel('TDnet (actual)')
plt.ylabel('TDnet (lin appr)')
plt.tight_layout()

        
envi = 11
# layer 1 single units
plt.figure(figsize=(6,4))
for uniti in range(nunit):
    ax = plt.subplot(8,8,uniti+1)
    plt.title(str(np.round(cor_layer1[envi,uniti],decimals=2)), fontsize=4)
    plt.plot(preActTlim[envi,uniti,layer1,:], lw=0.5)    
    plt.plot(preActProp_layer1[:,envi,uniti], lw=0.5)
    plt.axhline(0, color='gray', linestyle='--', alpha=1, lw=0.5)
    plt.xticks([])
    plt.yticks([])
    ax.axis('off')
plt.tight_layout()

# layer 2 single units
plt.figure(figsize=(6,4))
for uniti in range(nunit):
    ax = plt.subplot(8,8,uniti+1)
    plt.title(str(np.round(cor_layer2[envi,uniti],decimals=2)), fontsize=4)
    plt.plot(preActTlim[envi,uniti,layer2,:4000], lw=0.5)    
    plt.plot(preActProp_layer2[:4000,envi,uniti], lw=0.5)
    plt.axhline(0, color='gray', linestyle='--', alpha=1, lw=0.5)
    plt.xticks([])
    plt.yticks([])
    ax.axis('off')
plt.tight_layout()


#----- accuracy of linear regression -----#
plt.figure(figsize=(1.5,3.5))
plt.subplot(311)
plt.title('layer 0')
plt.hist(r2[:,0,:].flatten(), bins=20, range=(-1,1), histtype='step')
# plt.ylim([0,20])
plt.subplot(312)
plt.title('layer 1')
plt.hist(r2[:,1,:].flatten(), bins=20, range=(-1,1), histtype='step')
# plt.ylim([0,20])
plt.subplot(313)
plt.title('layer 2')
plt.hist(r2[:,2,:].flatten(), bins=20, range=(-1,1), histtype='step')
# plt.ylim([0,20])
plt.tight_layout()


#----- regression coefficients -----#
layer0=0
layer1=1
layer2=2
plt.figure(figsize=(1.5,3.5))
plt.subplot(311)
plt.plot(coef[:,layer0,:,0].flatten(), coef[:,layer0,:,1].flatten(), marker='o', linestyle='')
# plt.plot(coef[envi,layer0,:,1], coef[envi,layer0,:,2], marker='o', linestyle='')
# plt.plot(linreg_dopa.coef_[0], linreg_dopa.coef_[1], marker='x', ms=5, linestyle='')
plt.xlim([-0.4,0.6])
plt.ylim([-0.4,0.6])
plt.subplot(312)
plt.plot(coef[:,layer1,:,0].flatten(), coef[:,layer1,:,1].flatten(), marker='o', linestyle='')
# plt.plot(coef[envi,layer1,:,1], coef[envi,layer1,:,2], marker='o', linestyle='')
# plt.plot(linreg_dopa.coef_[0], linreg_dopa.coef_[1], marker='x', ms=5, linestyle='')
plt.xlim([-0.4,0.6])
plt.ylim([-0.4,0.6])
plt.subplot(313)
plt.plot(coef[:,layer2,:,0].flatten(), coef[:,layer2,:,1].flatten(), marker='o', linestyle='')
# plt.plot(coef[envi,layer2,:,1], coef[envi,layer2,:,2], marker='o', linestyle='')
# plt.plot(linreg_dopa.coef_[0], linreg_dopa.coef_[1], marker='x', ms=5, linestyle='')
plt.xlim([-0.4,0.6])
plt.ylim([-0.4,0.6])
plt.tight_layout()



#------ SVD on regression coeff and weight matrix ------#
envi = 0
plt.figure(figsize=(3.5,1.5))
plt.subplot(131)
plt.imshow(coef[envi,layer0,:,:], aspect='auto', vmin=-0.3, vmax=0.5)
plt.subplot(132)
plt.imshow(coef[envi,layer1,:,:], aspect='auto', vmin=-0.3, vmax=0.5)
plt.subplot(133)
plt.imshow(coef[envi,layer2,:,:], aspect='auto', vmin=-0.3, vmax=0.5)
plt.tight_layout()


U, S, Vt = np.linalg.svd(td_net[2].weight.detach().numpy())
cumsum2 = np.cumsum(S**2/np.sum(S**2))

U, S, Vt = np.linalg.svd(td_net[4].weight.detach().numpy())
cumsum4 = np.cumsum(S**2/np.sum(S**2))

plt.figure()
plt.plot(cumsum2)
plt.plot(cumsum4)
plt.tight_layout()



U0, S0, Vt0 = np.linalg.svd(coef[envi,layer0])
U1, S1, Vt1 = np.linalg.svd(coef[envi,layer1])
U1, S2, Vt2 = np.linalg.svd(coef[envi,layer2])
        
# plt.figure(figsize=(1.5,2.5))
# plt.subplot(211)
# plt.title('hidden layers')
# plt.hist(err_layer1.flatten(), bins=20, range=(0,1), histtype='step')
# plt.hist(err_layer2.flatten(), bins=20, range=(0,1), histtype='step')
# plt.subplot(212)
# plt.title('output layer')
# plt.hist(err_output.flatten(), bins=20, histtype='step')
# plt.tight_layout()

        
# plt.figure()
# for envi in range(10):
#     plt.plot(r2[envi,layer2,:], tdnet_lastlayer_wgt, marker='o', linestyle='')
# plt.xlim([-2,1.2])
# plt.tight_layout()

# plt.figure()
# plt.hist(tdnet_lastlayer_wgt, bins=20, histtype='step')
# plt.tight_layout()




# layer0=0
# layer1=1
# layer2=2
# plt.figure(figsize=(1.5,3.5))
# plt.subplot(311)
# plt.plot(coef[layer0,:,0], coef[layer0,:,2], marker='o', linestyle='')
# plt.plot(linreg_dopa.coef_[0], linreg_dopa.coef_[2], marker='x', ms=5, linestyle='')
# plt.ylim([-0.7,0.6])
# plt.subplot(312)
# plt.plot(coef[layer1,:,0], coef[layer1,:,2], marker='o', linestyle='')
# plt.plot(linreg_dopa.coef_[0], linreg_dopa.coef_[2], marker='x', ms=5, linestyle='')
# plt.ylim([-0.7,0.6])
# plt.subplot(313)
# plt.plot(coef[layer2,:,0], coef[layer2,:,2], marker='o', linestyle='')
# plt.plot(linreg_dopa.coef_[0], linreg_dopa.coef_[2], marker='x', ms=5, linestyle='')
# plt.ylim([-0.7,0.6])
# plt.tight_layout()




layer2 = 2
plt.figure(figsize=(1.5,3.5))
plt.subplot(311)
plt.plot(r2[layer2,:], coef[layer2,:,0] - linreg_dopa.coef_[0], marker='o', linestyle='')
plt.subplot(312)
plt.plot(r2[layer2,:], coef[layer2,:,1] - linreg_dopa.coef_[1], marker='o', linestyle='')
plt.subplot(313)
plt.plot(r2[layer2,:], coef[layer2,:,2] - linreg_dopa.coef_[2], marker='o', linestyle='')
plt.tight_layout()





#------ plots --------#


plt.figure(figsize=(4,3))
for epis in range(6):
    plt.subplot(3,2,epis+1)
    plt.plot(all_reward_dopa[epis], marker='.', linestyle='')
plt.tight_layout()
# plt.savefig('reward_evalpolicy.png', dpi=100)



envi = 2
plt.figure(figsize=(4,3))
done_idx = np.where(sim_dones[:,envi])[0]
for ix, d in enumerate(done_idx[:6]):
    plt.subplot(3,2,ix+1)
    if ix == 0:
        plt.plot(sim_raw_rewards[:d+1,envi], marker='.', linestyle='')
    else:
        plt.plot(sim_raw_rewards[dpre+1:d+1,envi], marker='.', linestyle='')
    dpre = d
    # plt.axvline(i, color='gray')
plt.tight_layout()



envi = 10
plt.figure()
idx_done = th.where(sim_dones[:,envi])[0]
for ix in idx_done:
    plt.axvline(ix, color='gray', linestyle='--')
plt.plot(sim_raw_rewards[:,envi])
plt.plot(sim_values[:,envi])
plt.tight_layout()





reward_avg = th.zeros(nenv)
for envi in range(nenv):
    dones_envi = th.where(sim_dones[:,envi] == 1)[0]
    reward_envi = th.zeros_like(dones_envi)
    for ix, d in enumerate(dones_envi):
        if ix == 0:
            reward_envi[ix] = th.sum(sim_raw_rewards[:d+1,envi])
            # reward_envi[ix] = th.sum(sim_rewards[:d+1,envi])
        else:
            reward_envi[ix] = th.sum(sim_raw_rewards[dpre+1:d+1,envi])
            # reward_envi[ix] = th.sum(sim_rewards[dpre+1:d+1,envi])
        dpre = d.clone()
    reward_avg[envi] = reward_envi.float().mean()
    



def compute_corr(var, net, corr):
    for layer in range(nlayer):
        for unit in range(nunit):
            for env in range(nenv):
                tensor_stacked = th.stack([var[env,:].flatten(), net[env,unit,layer,:].flatten()])
                corr[env,unit,layer] = th.corrcoef(tensor_stacked)[0,1]
    return corr    


dop = sim_dopa[1:,:].T
vpv = (model_dopa.gamma * sim_next_values - sim_values)[1:,:].T
rew = sim_raw_rewards[1:,:].T
val = sim_values[1:,:].T

corr_dop = th.zeros(nenv,nunit,nlayer)
corr_vpv = th.zeros(nenv,nunit,nlayer)
corr_rew = th.zeros(nenv,nunit,nlayer)
corr_val = th.zeros(nenv,nunit,nlayer)

corr_dop = compute_corr(dop, act, corr_dop).detach().numpy()
corr_vpv = compute_corr(vpv, act, corr_vpv).detach().numpy()
corr_rew = compute_corr(rew, act, corr_rew).detach().numpy()
corr_val = compute_corr(val, act, corr_val).detach().numpy()

avgcorr_out = np.mean(corr_dop,axis=0)
avgcorr_vpv = np.mean(corr_vpv,axis=0)
avgcorr_rew = np.mean(corr_rew,axis=0)
avgcorr_val = np.mean(corr_val,axis=0)


avgcorr = np.stack([avgcorr_rew, avgcorr_val, avgcorr_vpv, avgcorr_out], axis=2)







plt.figure(figsize=(1.5,3))
plt.subplot(3,1,1)
plt.title('layer 1')
layer0 = 0
plt.imshow(avgcorr[:,layer0,:], cmap='jet', vmin=-1, vmax=1, aspect='auto')
plt.colorbar()
plt.xticks([0,1,2,3], ['rew', 'val', 'vp-v', 'd'])

plt.subplot(3,1,2)
plt.title('layer 2')
layer1 = 1
plt.imshow(avgcorr[:,layer1,:], cmap='jet', vmin=-1, vmax=1, aspect='auto')
plt.colorbar()
plt.xticks([0,1,2,3], ['rew', 'val', 'vp-v', 'd'])

plt.subplot(3,1,3)
plt.title('layer 3')
layer2 = 2
plt.imshow(avgcorr[:,layer2,:], cmap='jet', vmin=-1, vmax=1, aspect='auto')
plt.colorbar()
plt.xticks([0,1,2,3], ['rew', 'val', 'vp-v', 'd'])
plt.tight_layout()




plt.figure(figsize=(1.5,3))
plt.subplot(3,1,1)
plt.title('tdnet out')
plt.hist(avgcorr_out[:,0], bins=20, range=(-1,1), histtype='step')
plt.subplot(3,1,2)
plt.hist(avgcorr_out[:,1], bins=20, range=(-1,1), histtype='step')
plt.subplot(3,1,3)
plt.hist(avgcorr_out[:,2], bins=20, range=(-1,1), histtype='step')
plt.tight_layout()


plt.figure(figsize=(1.5,3))
plt.subplot(3,1,1)
plt.title('vp - v')
plt.hist(avgcorr_vpv[:,0], bins=20, range=(-1,1), histtype='step')
plt.subplot(3,1,2)
plt.hist(avgcorr_vpv[:,1], bins=20, range=(-1,1), histtype='step')
plt.subplot(3,1,3)
plt.hist(avgcorr_vpv[:,2], bins=20, range=(-1,1), histtype='step')
plt.tight_layout()



plt.figure(figsize=(1.5,3))
plt.subplot(3,1,1)
plt.title('reward')
plt.hist(avgcorr_rew[:,0], bins=20, range=(-1,1), histtype='step')
plt.subplot(3,1,2)
plt.hist(avgcorr_rew[:,1], bins=20, range=(-1,1), histtype='step')
plt.subplot(3,1,3)
plt.hist(avgcorr_rew[:,2], bins=20, range=(-1,1), histtype='step')
plt.tight_layout()



plt.figure(figsize=(1.5,3))
plt.subplot(3,1,1)
plt.title('value')
plt.hist(avgcorr_val[:,0], bins=20, range=(-1,1), histtype='step')
plt.subplot(3,1,2)
plt.hist(avgcorr_val[:,1], bins=20, range=(-1,1), histtype='step')
plt.subplot(3,1,3)
plt.hist(avgcorr_val[:,2], bins=20, range=(-1,1), histtype='step')
plt.tight_layout()


#-------------------
# activation across layers
#-------------------
envi = 1
layeri = 2
act_env_layer = act[envi,:,layeri,:].detach().numpy()
act_env_layer_corr = np.corrcoef(act_env_layer)
    
plt.figure(figsize=(2,1.5))
plt.imshow(act_env_layer_corr, cmap='jet', vmin = -1, vmax = 1, aspect='auto')    
plt.colorbar()
plt.tight_layout()



plt.figure()
plt.imshow(act_env_layer, cmap='jet', vmin = 0, vmax = 10, aspect='auto')    
plt.colorbar()
plt.tight_layout()



plt.figure(figsize=(5,4))
idx_done = th.where(sim_dones[:,envi])[0]
ax1 = plt.subplot(311)
for donei in idx_done:
    plt.axvline(donei, color='gray', linestyle='--')
for i in range(nunit):
    plt.plot(act_env_layer[i,:])

plt.subplot(312, sharex=ax1)
for donei in idx_done:
    plt.axvline(donei, color='gray', linestyle='--')
plt.plot(sim_values[:,envi])

plt.subplot(313, sharex=ax1)
for donei in idx_done:
    plt.axvline(donei, color='gray', linestyle='--')
plt.plot(sim_rewards[:,envi])
plt.tight_layout()



