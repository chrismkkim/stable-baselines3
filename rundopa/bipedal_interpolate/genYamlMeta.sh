#!/bin/bash

# Lists of values to use
n_interpolate_list=(0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
n_envs_list=(60)
n_units_list=(128)
n_rlnet_reset=10
n_episode=10
episode_max_time=1600

for n_inter in "${n_interpolate_list[@]}"; do
    for n_envs in "${n_envs_list[@]}"; do
        n_timesteps=$(( $n_rlnet_reset * $n_envs * $n_episode * $episode_max_time ))
        for n_units in "${n_units_list[@]}"; do
        fname="/home/kimchm/RL/rl-baselines3-zoo/hyperparams/BipedalWalker-v3/tdnet_interpolate/swarm_meta_envs${n_envs}_units${n_units}_inter${n_inter}.yml"
        cat << EOF > "$fname"
BipedalWalker-v3:
    n_envs: $n_envs
    n_rlnet_reset: $n_rlnet_reset
    n_timesteps: !!float $n_timesteps
    policy: 'MlpPolicy'
    n_steps: 1
    n_meta_steps: 1
    learning_rate: !!float 6.0e-4
    learning_rate_dopa: !!float 1e-4
    ent_coef: !!float 15.0e-5
    gamma: 0.99
    gae_lambda: 0.0
    tracker_window_size: 1000
    net_arch: dict(pi=[64, 64], vf=[64, 64], re=[64,64], td=[$n_units,$n_units,$n_units,1])
    normalize_values: False
    traintype_meta: True    
    interpolation_constant: 1.0
EOF
        echo "Generated $fname"
        done
    done
done

