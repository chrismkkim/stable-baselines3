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
        "sim_id",
        type=str,
        help="Sim id."
    )
    
    args = parser.parse_args()
    env_id = args.env_id
    # env_id_rl   = args.env_id_rl
    sim_id      = args.sim_id

    # We assume train.py lives in the same folder as this script.
    path = '/Users/kimchm/Documents/GitHub/rl-baselines3-zoo/'
    zoo_root = os.path.abspath(path)

    # List of algorithms to run in sequence:
    algos   = ["a2c"]
    yaml    = ["a2c.yml"]
    env_ids = [env_id]
    
    # path to log 
    path_to_log = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/trainedmodel/' + sim_id
    # path_to_log = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/trainedmodel'
    path_to_par = 'hyperparams/'
    for i, algo in enumerate(algos):
        print("\n" + "=" * 60)
        print(f"Starting training with algo = {algos[i]}, env = {env_ids[i]}")
        print("=" * 60 + "\n")

        cmd = [
            sys.executable,         # ensures same Python interpreter
            "train.py",
            "--algo", algos[i],
            "--env", env_ids[i],
            "--log-folder", path_to_log,
            "--verbose", "0"
        ]

        # cmd = [
        #     sys.executable,         # ensures same Python interpreter
        #     "train.py",
        #     "--algo", algos[i],
        #     "--env", env_ids[i],
        #     "--gym-env", "CartPole-v1",
        #     "--log-folder", path_to_log
        # ]

        # Run train.py in the zoo root
        try:
            subprocess.run(cmd, cwd=zoo_root, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\nERROR: `train.py --algo {algo} --env {env_ids[i]}` exited with code {e.returncode}\n")
            sys.exit(e.returncode)

    print("\nAll done: both Dopa and A2C have finished training on:", env_ids[i])
    

if __name__ == "__main__":
    main()