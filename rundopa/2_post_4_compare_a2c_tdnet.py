import numpy as np
import results_plotter
import matplotlib.pyplot as plt
import os
from utils import A2CReward
import Plots.compare_a2c_tdnet, Plots.tdnet_interpolation
import importlib

plt.style.use(os.path.join(os.path.dirname(__file__), 'pyplot_setting.mplstyle'))

# env_id  = 'CartPole-v1'
# env_id = 'LunarLander-v3'
env_id = 'BipedalWalker-v3'

pathfig = '/Users/kimchm/Documents/GitHub/stable-baselines3/rundopa/figure/a2c_vs_tdnet/'
pathsave = '/Users/kimchm/Documents/RL/biowulf/' + env_id + '/saved/' 
pathclass_a2c = 'A2CRewardClass-compact.npy'
pathclass_tdnet = 'TDnetRewardClass-compact.npy'
pathclass_tdnet_interpolate = 'TDinterpolateClass.npy'

a2cRew = np.load(pathsave + pathclass_a2c, allow_pickle=True)
tdnetRew = np.load(pathsave + pathclass_tdnet, allow_pickle=True)
tdnetInterRew = np.load(pathsave + pathclass_tdnet_interpolate, allow_pickle=True)

# a2c:   nenv x nsims
# tdnet: nenv x nsims x nunit
a2c, a2c_avg, a2c_sem = a2cRew.get_averages()
tdnet, \
tdnet_avg,     tdnet_sem, \
tdnet_nn_avg,  tdnet_nn_sem, \
tdnet_env_avg, tdnet_env_sem, \
tdnet_best_nn, tdnet_best_avg, tdnet_best_sem = tdnetRew.get_averages()
tdinter, tdinter_avg, tdinter_sem = tdnetInterRew.get_averages()

#-------------------------------------#
# compare a2c vs. tdnet
#-------------------------------------#
# a2c:           nenv x nsims
# a2c_avg:       nenv
#-------------------------------------#
# tdnet:         nenv x nsims x nunit
# tdnet_avg:     nenv x nunit
# tdnet_nn_avg:  nunit
# tdnet_env_avg: nenv
#-------------------------------------#
plot_compare_a2c_tdnet = Plots.compare_a2c_tdnet.gen(
                        pathfig,
                        env_id, 
                        a2cRew, tdnetRew,
                        a2c, a2c_avg, a2c_sem,
                        tdnet,
                        tdnet_avg, tdnet_sem,
                        tdnet_nn_avg, tdnet_nn_sem,
                        tdnet_env_avg, tdnet_env_sem,
                        tdnet_best_nn, tdnet_best_avg, tdnet_best_sem)
plot_compare_a2c_tdnet.compare_all()
plot_compare_a2c_tdnet.compare_env()
plot_compare_a2c_tdnet.compare_hist()

#----------------------------------------#
# compare a2c vs. tdnet vs. tdnet interpolation
#----------------------------------------#
# a2c:           nenv x nsims
# a2c_avg:       nenv                    <--- used for plotting (nenv=60)
#-------------------------------------#
# tdnet:         nenv x nsims x nunit
# tdnet_avg:     nenv x nunit            <--- used for plotting (nenv=60, nunit=128)
#-------------------------------------#
# tdinter:       ninterpolation x nsims
# tdineter_avg:  ninterpolation          <--- used for plotting
#-------------------------------------#
plot_tdnet_interpolation = Plots.tdnet_interpolation.gen(
                        pathfig,
                        env_id, 
                        a2cRew, tdnetRew, tdnetInterRew,
                        a2c, a2c_avg, a2c_sem,
                        tdnet, tdnet_avg, tdnet_sem,
                        tdinter, tdinter_avg, tdinter_sem)
plot_tdnet_interpolation.compare_tdnet_interpolation()

x=1


plt.figure(figsize=(3,3))
plt.subplot(2,2,1)
plt.hist(plot_tdnet_interpolation.a2c[-1,:], bins=10, color='r', label='A2C', histtype='step', alpha=1, range=(-200,300), density=True)
plt.axvline(np.mean(plot_tdnet_interpolation.a2c[-1,:]), color='r', linestyle='--')
plt.ylim([0,0.008])
plt.title('A2C')
plt.subplot(2,2,2)
plt.hist(plot_tdnet_interpolation.tdnet_e60_u128, bins=10, color='C2', label='TDnet', histtype='step', alpha=1, range=(-200,300), density=True)
plt.axvline(np.mean(plot_tdnet_interpolation.tdnet_e60_u128), color='C2', linestyle='--')
plt.ylim([0,0.008])
plt.title('TDnet')
plt.subplot(2,2,3)
plt.hist(plot_tdnet_interpolation.tdinter[0,:], bins=10, color='k', label='TDintp0', histtype='step', alpha=1, range=(-200,300), density=True)
plt.axvline(np.mean(plot_tdnet_interpolation.tdinter[0,:]), color='k', linestyle='--')
plt.ylim([0,0.008])
plt.title('TDintp0')
plt.subplot(2,2,4)
plt.hist(plot_tdnet_interpolation.tdinter[1,:], bins=10, color='b', label='TDintp1', histtype='step', alpha=1, range=(-200,300), density=True)
plt.axvline(np.mean(plot_tdnet_interpolation.tdinter[1,:]), color='b', linestyle='--')
plt.ylim([0,0.008])
plt.title('TDintp1')
# plt.legend(bbox_to_anchor=[1,1])
plt.tight_layout()
plt.savefig(pathfig + 'tdnet_interpolate.pdf')


_a2c_avg = np.mean(plot_tdnet_interpolation.a2c[-1,:])
_tdnet_avg = np.mean(plot_tdnet_interpolation.tdnet_e60_u128)
_tdintp0_avg = np.mean(plot_tdnet_interpolation.tdinter[0,:])
_tdintp1_avg = np.mean(plot_tdnet_interpolation.tdinter[1,:])
_avg = [_a2c_avg, _tdnet_avg, _tdintp0_avg, _tdintp1_avg]

plt.figure()
plt.plot(_avg, marker='o')
plt.xticks([0,1,2,3],['a2c','tdnet','tdinp0', 'tdinp1'])
plt.tight_layout()



plt.figure()
# plt.hist(plot_tdnet_interpolation.tdnet_e60_u128, bins=20, color='C1', label='Interp', histtype='step')
plt.hist(plot_tdnet_interpolation.a2c[-1,:], bins=20, color='r', label='A2C', histtype='bar', alpha=0.5, range=(-200,300))
plt.hist(plot_tdnet_interpolation.tdinter[0,:], bins=20, color='k', label='Interp', histtype='step', range=(-200,300))
plt.axvline(np.mean(plot_tdnet_interpolation.a2c[-1,:]), color='r', linestyle='--')
plt.axvline(np.mean(plot_tdnet_interpolation.tdinter[0,:]), color='k', linestyle='--')
plt.legend()
plt.tight_layout()




plt.figure()
plt.hist(a2c[-1,:], bins=10, color='r', label='A2C', histtype='bar', alpha=0.5, range=(-200,300))
plt.hist(plot_tdnet_interpolation.tdnet_e60_u128, bins=10, color='C2', label='TDnet', histtype='step', alpha=1, range=(-200,300))
# plt.hist(tdinter[0,:], bins=10, color='r', label='A2C', histtype='bar', alpha=0.5, range=(-200,300))
# plt.hist(tdinter[0,:], bins=10, color='k', label='TDnet', histtype='step', alpha=0.5, range=(-200,300))
plt.legend()
plt.tight_layout()


plt.figure(figsize=(2.3,1.5))
plt.hist(tdinter[0,:], bins=10, color='k', label='TDintp0', histtype='bar', alpha=0.5, range=(-200,300), density=True)
plt.hist(tdinter[1,:], bins=10, color='b', label='TDintp1', histtype='step', alpha=1, range=(-200,300), density=True)
plt.hist(plot_tdnet_interpolation.tdnet_e60_u128, bins=10, color='C2', label='TDnet', histtype='step', alpha=1, range=(-200,300), density=True)
plt.legend(bbox_to_anchor=[1,1])
plt.tight_layout()



# plt.figure(figsize=(6,1.3))
# for envi in range(a2cRew.nenv):
#     plt.subplot(1,a2cRew.nenv,envi+1)
#     plt.fill_between(np.arange(tdnetRew.nunit), tdnet_avg[envi] - tdnet_sem[envi], tdnet_avg[envi] + tdnet_sem[envi], fc='C0', ec='None', alpha=0.3) 
#     plt.plot(np.arange(tdnetRew.nunit), tdnet_avg[envi], marker='o', c='C0', label='TDnet')
#     plt.fill_between(np.arange(tdnetRew.nunit), a2c_avg[envi] - a2c_sem[envi], a2c_avg[envi] + a2c_sem[envi], fc='r', ec='None', alpha=0.3) 
#     plt.axhline(a2c_avg[envi], color='r', linestyle='--', label='A2C')
#     # plt.axhline(0, color='r', linestyle='--')
#     plt.xticks(np.arange(tdnetRew.nunit), tdnetRew.unit_list, rotation=45, fontsize=5)
#     plt.yticks(fontsize=5)
#     plt.title('nenv ' + str(a2cRew.env_list[envi]), fontsize=5)
#     # plt.ylim([-100,150])
#     plt.ylim([80,200])
#     if envi == 0:
#         plt.ylabel('reward', fontsize=5)
#         plt.legend(frameon=False, fontsize=5)
#     plt.xlabel('num unit', fontsize=5)
# plt.tight_layout()
# plt.savefig(pathfig + env_id + '_compare_all.pdf')


# plt.figure(figsize=(2.3,1.5))
# # plt.fill_between(a2cRew.env_list, tdnet_best_avg - tdnet_best_sem, tdnet_best_avg + tdnet_best_sem, fc='C1', ec='None', alpha=0.3) 
# # plt.plot(a2cRew.env_list, tdnet_best_avg, c='C1', label='TDnet Best', marker='o')
# plt.fill_between(tdnetRew.env_list, tdnet_env_avg - tdnet_env_sem, tdnet_env_avg + tdnet_env_sem, fc='C0', ec='None', alpha=0.3) 
# plt.plot(tdnetRew.env_list, tdnet_env_avg, c='C0', label='TDnet', marker='o')
# plt.fill_between(a2cRew.env_list, a2c_avg - a2c_sem, a2c_avg + a2c_sem, fc='r', ec='None', alpha=0.3) 
# plt.plot(a2cRew.env_list, a2c_avg, c='r', label='A2C', marker='o')
# plt.legend(frameon=False, bbox_to_anchor=[1,1])
# plt.xticks(a2cRew.env_list)
# plt.xlabel('num env')
# plt.ylabel('reward')
# plt.tight_layout()
# plt.savefig(pathfig + env_id + '_compare_env.pdf')



# plt.figure(figsize=(1.8,1.5))
# plt.hist(tdnet.flatten(), bins=20, range=(-150,300), histtype='step', density=True, color='C0', label='TDnet')
# plt.hist(a2c.flatten(), bins=20, range=(-150,300), histtype='bar', density=True, color='r', label='A2C', alpha=0.5)
# plt.xlabel('reward')
# plt.ylabel('agent density')
# plt.legend(frameon=False)
# plt.tight_layout()
# plt.savefig(pathfig + env_id + '_compare_hist.pdf')





# #-----------additional plots ------------#
# plt.figure(figsize=(5,1.5))
# for envi in range(a2cRew.nenv):
#     plt.subplot(1,a2cRew.nenv,envi+1)
#     uniti = np.argmax(tdnet_avg[envi,:])
#     nsim = a2cRew.nsims
#     plt.boxplot([a2cRew_all[envi], tdnet_best_nn[envi]], medianprops=dict(visible=False), meanprops=dict(color='r'), meanline=True, showmeans=True, showfliers=False, widths=0.5)
#     plt.plot(1 + 0.01*np.random.randn(nsim), a2cRew_all[envi], c='gray', marker='x', ms=2.5, mew=0.5, mfc='none', linestyle='')
#     plt.plot(2 + 0.01*np.random.randn(nsim), tdnet_best_nn[envi], c='gray', marker='x', ms=2.5, mew=0.5, mfc='none', linestyle='')
#     plt.xticks([1, 2], ['A2C', 'TDnet'], rotation=45)
#     plt.title('nenv ' + str(a2cRew.env_list[envi]))
#     if envi == 0:
#         plt.ylabel('reward')
#     # plt.ylim([-10,250])
#     plt.ylim([-150,300])
# plt.tight_layout()
# plt.savefig(pathfig + env_id + '_compare_bestunit.pdf')



# plt.figure(figsize=(5,1.5))
# for envi in range(a2cRew.nenv):
#     plt.subplot(1,a2cRew.nenv,envi+1)
#     plt.hist(tdnet[envi].flatten(), bins=20, range=(-150,300), histtype='step', density=True, color='C0')
#     plt.hist(a2c[envi], bins=20, range=(-150,300), histtype='bar', density=True, color='r', alpha=0.5)
#     plt.title('nenv ' + str(a2cRew.env_list[envi]))
#     if envi == 0:
#         plt.ylabel('agent density')
#     plt.yticks([])
#     # plt.ylim([-10,250])
#     # plt.ylim([0,15])
#     plt.xlabel('reward')
# plt.tight_layout()
# plt.savefig(pathfig + env_id + '_compare_hist_all.pdf')

