#!/bin/bash
dirname="/home/kimchm/RL/stable-baselines3/rundopa/"
fname="script_rundopa_Lunar_reward.sh"
touch "${dirname}${fname}"

# Lists of values to use
n_envs_list=(10 20 30)
lr_list=(0.0002 0.0004 0.0008)
entcoef_list=(0.5 0.05 0.005)

for n_envs in "${n_envs_list[@]}"; do
    for lr in "${lr_list[@]}"; do
        for entcoef in "${entcoef_list[@]}"; do
            for sim in {0..9}; do
            echo "source /data/kimchm/conda/bin/activate pyrl && python3 run_dopa_reward.py LunarLander-v3 LunarLander-v3 ${n_envs} ${lr} ${entcoef} ${sim}" >> "${dirname}${fname}"
            done
        done
    done
done
