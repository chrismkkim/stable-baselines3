#!/bin/bash

# Lists of values to use
n_envs_list=(1 4 7 10 20 30 40)
n_units_list=(16 32 64 128 256 384 512 640 768)
n_rlnet_reset=10

for n_envs in "${n_envs_list[@]}"; do
    n_timesteps=$(( $n_envs * 10000 ))
    for n_units in "${n_units_list[@]}"; do
    fname="/home/kimchm/RL/rl-baselines3-zoo/hyperparams/lunar/swarm_rl_envs${n_envs}_units${n_units}.yml"
    cat << EOF > "$fname"
LunarLander-v3:
    n_envs: $n_envs
    n_rlnet_reset: $n_rlnet_reset
    n_timesteps: !!float $n_timesteps
    policy: 'MlpPolicy'
    n_steps: 1
    n_meta_steps: 1
    learning_rate: !!float 0.00083
    learning_rate_dopa: !!float 1e-4
    ent_coef: 0.00001
    gae_lambda: 0.0
    tracker_window_size: 1000
    net_arch: dict(pi=[64, 64], vf=[64, 64], re=[64,64], td=[$n_units,$n_units,$n_units,1])
    normalize_values: False
    traintype_meta: False
EOF
    echo "Generated $fname"
    done
done