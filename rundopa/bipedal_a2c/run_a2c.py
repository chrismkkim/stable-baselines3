#!/usr/bin/env python3
"""
run_two_algos.py

Runs RL-Baselines3-Zoo’s train.py twice:
  1. --algo dopa --env <env_id>
  2. --algo a2c  --env <env_id>

Each invocation uses whatever hyperparameters are defined in:
  hyperparameters/dopa.yml   (under the <env_id> block)
  hyperparameters/a2c.yml    (under the <env_id> block)

Usage:
    python run_two_algos.py <env_id>

Example:
    python run_two_algos.py LunarLander-v3
    python run_two_algos.py CartPole-v1
"""

import argparse
import subprocess
import sys
import os
from stable_baselines3.common import results_plotter


#    BipedalWalker-v3 1 0.0006 0.00015 0

def main():
    parser = argparse.ArgumentParser(
        description="Run both Dopa and A2C (with their YAML defaults) on the same env."
    )
    parser.add_argument(
        "env_id",
        type=str,
        help="Gym environment ID (e.g. CartPole-v1, LunarLander-v3)."
    )
    parser.add_argument(
        "nenv",
        type=str,
        help="nenv"
    )
    parser.add_argument(
        "lr",
        type=str,
        help="learning rate"
    )
    parser.add_argument(
        "entcoef",
        type=str,
        help="entropy coef"
    )    
    parser.add_argument(
        "sim_id",
        type=str,
        help="Sim id."
    )
    
    args = parser.parse_args()
    env_id      = args.env_id
    nenv        = args.nenv
    lr          = args.lr
    entcoef     = args.entcoef    
    sim_id      = args.sim_id

    # We assume train.py lives in the same folder as this script.
    path = '/Users/kimchm/Documents/GitHub/rl-baselines3-zoo/'
    zoo_root = os.path.abspath(path)

    temporary_testing = True
    rndseed = str(123)
    
    # List of algorithms to run in sequence:
    algo    = 'a2c'
    if not temporary_testing:
        yaml    = [f"a2c_env{nenv}_lr{lr}_ent{entcoef}.yml"]
    else:
        yaml    = [f"a2c_env{nenv}_lr{lr}_ent{entcoef}_tmp.yml"]
    env_ids = [env_id]
    
    # path to log 
    path        = '/Users/kimchm/Documents/RL/trainedmodel/'
    path_to_log = path + env_id + '/' + 'env_' + nenv + '_lr_' + lr + '_ent_' + entcoef + '/' + sim_id
    path_to_par = 'hyperparams/' + env_id + '/a2c/'
    for i in range(len(yaml)):
        print("\n" + "=" * 60)
        print(f"Starting training with algo = {algo}, env = {env_ids[i]}")
        print("=" * 60 + "\n")

        if not temporary_testing:
            cmd = [
                sys.executable,         # ensures same Python interpreter
                "train.py",
                "--algo", algo,
                "--conf-file", path_to_par + yaml[i],
                "--env", env_ids[i],
                "--log-folder", path_to_log,
                "--verbose", "0",
                "--eval-episodes", "100",
                "--eval-num", "20",
                "--seed", rndseed
            ]
        else:
            cmd = [
                sys.executable,         # ensures same Python interpreter
                "train.py",
                "--algo", algo,
                "--conf-file", path_to_par + yaml[i],
                "--env", env_ids[i],
                "--log-folder", path_to_log,
                "--verbose", "0",
                "--eval-episodes", "1",
                "--eval-num", "2",
                "--seed", rndseed
            ]            
        # Run train.py in the zoo root
        try:
            subprocess.run(cmd, cwd=zoo_root, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\nERROR: `train.py --algo {algo} --env {env_ids[i]}` exited with code {e.returncode}\n")
            sys.exit(e.returncode)

    # print("\nAll done: both Dopa and A2C have finished training on:", env_ids[i])  

if __name__ == "__main__":
    main()