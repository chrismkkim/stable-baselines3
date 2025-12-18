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


#    CartPole-v1 CartPole-v1 1

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
    path = '/home/kimchm/RL/rl-baselines3-zoo/'
    zoo_root = os.path.abspath(path)

    # List of algorithms to run in sequence:
    algo    = 'a2c'
    yaml    = [f"a2c_env{nenv}_lr{lr}_ent{entcoef}.yml"]
    env_ids = [env_id]
    
    # path to log 
    path_to_log = '/data/kimchm/data/RL/a2c/' + env_id + '/' + 'env_' + nenv + '_lr_' + lr + '_ent_' + entcoef + '/' + sim_id
    path_to_par = 'hyperparams/' + env_id + '/a2c/'
    for i in range(len(yaml)):
        print("\n" + "=" * 60)
        print(f"Starting training with algo = {algo}, env = {env_ids[i]}")
        print("=" * 60 + "\n")

        cmd = [
            sys.executable,         # ensures same Python interpreter
            "train.py",
            "--algo", algo,
            "--conf-file", path_to_par + yaml[i],
            "--env", env_ids[i],
            "--log-folder", path_to_log,
            "--verbose", "0",
            "--eval-episodes", "100",
            "--eval-num", "20"
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