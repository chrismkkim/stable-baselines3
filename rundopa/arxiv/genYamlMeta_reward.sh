#!/bin/bash

# Lists of values to use
n_envs_list=(10 20 30)
lr_list=(0.0002 0.0004 0.0008)
entcoef_list=(0.5 0.05 0.005)
n_rlnet_reset=10

for n_envs in "${n_envs_list[@]}"; do
    n_timesteps=$(( $n_envs * $n_rlnet_reset * 10000 ))
    for lr in "${lr_list[@]}"; do
        for entcoef in "${entcoef_list[@]}"; do
        fname="/home/kimchm/RL/rl-baselines3-zoo/hyperparams/lunar/reward/swarm_meta_envs${n_envs}_lr${lr}_ent${entcoef}.yml"
        cat << EOF > "$fname"
LunarLander-v3:
    n_envs: $n_envs
    n_rlnet_reset: $n_rlnet_reset
    n_timesteps: !!float $n_timesteps
    policy: 'MlpPolicy'
    n_steps: 1
    n_meta_steps: 1
    learning_rate: !!float $lr
    learning_rate_dopa: !!float 1e-4
    ent_coef: $entcoef
    gae_lambda: 0.0
    tracker_window_size: 1000
    net_arch: dict(pi=[64, 64], vf=[64, 64], re=[64,64], td=[128,128,128,1])
    normalize_values: False
    traintype_meta: True
EOF
    echo "Generated $fname"
        done
    done
done

