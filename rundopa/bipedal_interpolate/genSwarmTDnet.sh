#!/bin/bash
dirname="/home/kimchm/RL/stable-baselines3/rundopa/bipedal_interpolate/"
fname="script_rundopa_bipedal.sh"
touch "${dirname}${fname}"

# Lists of values to use
n_interpolate_list=(0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
n_envs_list=(60)
n_units_list=(128)
nsim=20

for n_inter in "${n_interpolate_list[@]}"; do
    for n_env in "${n_envs_list[@]}"; do
        for n_units in "${n_units_list[@]}"; do
            for sim in $(seq 0 $((nsim - 1))); do
                echo "source /data/kimchm/conda/bin/activate pyrl && python3 run_dopa.py BipedalWalker-v3 BipedalWalker-v3 ${n_env} ${n_units} ${n_inter} ${sim}" >> "${dirname}${fname}"
            done
        done
    done
done

