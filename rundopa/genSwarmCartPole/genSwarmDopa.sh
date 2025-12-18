#!/bin/bash
dirname="/home/kimchm/RL/stable-baselines3/rundopa/"
fname="script_rundopa_CartPole_errors.sh"
touch "${dirname}${fname}"

# Lists of values to use
n_envs_list=(1 4 7 10 20 30 40)
n_units_list=(16 32 64 128 256 384 512 640 768)

for n_envs in "${n_envs_list[@]}"; do
    for n_units in "${n_units_list[@]}"; do
        for sim in {0..9}; do
            echo "source /data/kimchm/conda/bin/activate pyrl && python3 run_dopa.py CartPole-v1 CartPole-v1 ${n_envs} ${n_units} ${sim}" >> "${dirname}${fname}"
        done
    done
done