#!/bin/bash
dirname="/home/kimchm/RL/stable-baselines3/rundopa/bipedal_a2c/"
fname="script_runa2c_bipedal.sh"
touch "${dirname}${fname}"

# Lists of values to use
n_envs_list=(10 20 30 40 50 60)
lr_list=(0.00002 0.00004 0.00006 0.00008 0.0001 0.0002 0.0004 0.0006 0.0008 0.001)
entcoef_list=(0.000005 0.000015 0.00005 0.00015 0.0005 0.0015 0.005)
nsim=100

for n_env in "${n_envs_list[@]}"; do
    for lr in "${lr_list[@]}"; do
        for entcoef in "${entcoef_list[@]}"; do
            for sim in $(seq 0 $((nsim - 1))); do
                echo "source /data/kimchm/conda/bin/activate pyrl && python3 run_a2c.py BipedalWalker-v3 ${n_env} ${lr} ${entcoef} ${sim}" >> "${dirname}${fname}"
            done
        done
    done
done

