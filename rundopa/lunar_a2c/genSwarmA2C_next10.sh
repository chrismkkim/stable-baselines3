#!/bin/bash
dirname="/home/kimchm/RL/stable-baselines3/rundopa/lunar_a2c/"
fname="script_runa2c_Lunar_next10.sh"
touch "${dirname}${fname}"

# Lists of values to use
n_envs_list=(10 20 30 40 50)
lr_list=(0.0008 0.001 0.0012 0.0014 0.0016 0.0018 0.002)
entcoef_list=(0.000005 0.00005 0.0005 0.005 0.05 0.5 5.0)
nsim=20

for n_env in "${n_envs_list[@]}"; do
    for lr in "${lr_list[@]}"; do
        for entcoef in "${entcoef_list[@]}"; do
            for sim in $(seq 10 $((nsim - 1))); do
                echo "source /data/kimchm/conda/bin/activate pyrl && python3 run_a2c_reward.py LunarLander-v3 ${n_env} ${lr} ${entcoef} ${sim}" >> "${dirname}${fname}"
            done
        done
    done
done