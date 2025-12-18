#!/bin/bash
dirname="/home/kimchm/RL/stable-baselines3/rundopa/bipedal_tdnet/"
fname="script_rundopa_bipedal.sh"
touch "${dirname}${fname}"

# Lists of values to use
n_envs_list=(10 20 30 40 50 60)
n_units_list=(16 32 64 128)
nsim=100

for n_env in "${n_envs_list[@]}"; do
    for n_units in "${n_units_list[@]}"; do
        for sim in $(seq 0 $((nsim - 1))); do
            echo "source /data/kimchm/conda/bin/activate pyrl && python3 run_dopa.py BipedalWalker-v3 BipedalWalker-v3 ${n_env} ${n_units} ${sim}" >> "${dirname}${fname}"
        done
    done
done


