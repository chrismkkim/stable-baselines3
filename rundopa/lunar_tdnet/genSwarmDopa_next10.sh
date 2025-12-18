#!/bin/bash
dirname="/home/kimchm/RL/stable-baselines3/rundopa/lunar_tdnet/"
fname="script_rundopa_Lunar_next10.sh"
touch "${dirname}${fname}"

# Lists of values to use
n_envs_list=(10 20 30 40 50 60)
n_units_list=(16 32 64 128 256 384 512)

for n_envs in "${n_envs_list[@]}"; do
    for n_units in "${n_units_list[@]}"; do
        for sim in {10..19}; do
            echo "source /data/kimchm/conda/bin/activate pyrl && python3 run_dopa.py LunarLander-v3 LunarLander-v3 ${n_envs} ${n_units} ${sim}" >> "${dirname}${fname}"
        done
    done
done
