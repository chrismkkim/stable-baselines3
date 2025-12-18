#!/bin/bash

# Lists of values to use
n_envs_list=(10 20 30 40 50 60)
lr_list=(0.00002 0.00004 0.00006 0.00008 0.0001 0.0002 0.0004 0.0006 0.0008 0.001)
entcoef_list=(0.000005 0.000015 0.00005 0.00015 0.0005 0.0015 0.005)
episode_max_time=1600
n_episode=10

for n_envs in "${n_envs_list[@]}"; do
    n_timesteps=$(( $n_envs * $n_episode * $episode_max_time ))
    for lr in "${lr_list[@]}"; do
        for entcoef in "${entcoef_list[@]}"; do
            fname="/home/kimchm/RL/rl-baselines3-zoo/hyperparams/BipedalWalker-v3/a2c/a2c_env${n_envs}_lr${lr}_ent${entcoef}.yml"
            cat << EOF > "$fname"
BipedalWalker-v3:
    n_envs: $n_envs
    n_timesteps: !!float $n_timesteps
    policy: 'MlpPolicy'
    n_steps: 1
    learning_rate: $lr
    ent_coef: $entcoef
    gamma: 0.99
    gae_lambda: 0.0
    max_grad_norm: !!float 1e10
EOF
            echo "Generated $fname"
        done
    done
done


