#!/bin/bash

# Lists of values to use
n_envs_list=(10 20 30 40 50)
lr_list=(0.0008 0.001 0.0012 0.0014 0.0016 0.0018 0.002)
entcoef_list=(0.000005 0.00005 0.0005 0.005 0.05 0.5 5.0)
n_rlnet_reset=10
episode_max_time=1000
n_episode=10

for n_envs in "${n_envs_list[@]}"; do
    n_timesteps=$(( $n_envs * $n_episode * $episode_max_time ))
    for lr in "${lr_list[@]}"; do
        for entcoef in "${entcoef_list[@]}"; do
            fname="/home/kimchm/RL/rl-baselines3-zoo/hyperparams/LunarLander-v3/a2c/a2c_env${n_envs}_lr${lr}_ent${entcoef}.yml"
            cat << EOF > "$fname"
LunarLander-v3:
    n_envs: $n_envs
    n_timesteps: !!float $n_timesteps
    policy: 'MlpPolicy'
    n_steps: 1
    learning_rate: $lr
    ent_coef: $entcoef
    gae_lambda: 0.0
    max_grad_norm: !!float 1e10
EOF
            echo "Generated $fname"
        done
    done
done


