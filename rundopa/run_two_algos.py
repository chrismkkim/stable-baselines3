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
import runDA_plotloss


def main():
    parser = argparse.ArgumentParser(
        description="Run both Dopa and A2C (with their YAML defaults) on the same env."
    )
    parser.add_argument(
        "env_id",
        type=str,
        help="Gym environment ID (e.g. CartPole-v1, LunarLander-v3)."
    )
    args = parser.parse_args()
    env_id = args.env_id

    # We assume train.py lives in the same folder as this script.
    path = '/Users/kimchm/Documents/GitHub/rl-baselines3-zoo'
    zoo_root = os.path.abspath(path)
    # zoo_root = os.path.abspath(os.path.dirname(__file__))

    # List of algorithms to run in sequence:
    algos = ["dopa", "a2c"]

    # path to log 
    path_to_log = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/logs'
    for algo in algos:
        print("\n" + "=" * 60)
        print(f"Starting training with algo = {algo}, env = {env_id}")
        print("=" * 60 + "\n")

        cmd = [
            sys.executable,         # ensures same Python interpreter
            "train.py",
            "--algo", algo,
            "--env", env_id,
            "--log-folder", path_to_log
        ]

        # Run train.py in the zoo root
        try:
            subprocess.run(cmd, cwd=zoo_root, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\nERROR: `train.py --algo {algo} --env {env_id}` exited with code {e.returncode}\n")
            sys.exit(e.returncode)

    print("\nAll done: both Dopa and A2C have finished training on:", env_id)
    

if __name__ == "__main__":
    main()