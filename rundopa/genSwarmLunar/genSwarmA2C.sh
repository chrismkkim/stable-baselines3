#!/bin/bash
dirname="/home/kimchm/RL/stable-baselines3/rundopa/"
fname="script_runa2c_Lunar.sh"
touch "${dirname}${fname}"

# Lists of values to use
n_envs_list=(10 20 30 40)

for n_envs in "${n_envs_list[@]}"; do
    for sim in {0..9}; do
        echo "source /data/kimchm/conda/bin/activate pyrl && python3 run_a2c.py LunarLander-v3 ${n_envs} ${sim}" >> "${dirname}${fname}"
    done
done