import argparse
import gc
import sys

import torch
from devtools import pprint

from seahorse.config.experiment_config import RunConfig
from seahorse.experiments import experiment_registry
from seahorse.experiments.experiment_utils import print_gpu_memory_usage
from seahorse.train.train import run_training


def print_usage():
    print("Usage: python run_experiment.py <experiment_name>")
    print("\nAvailable experiments:")
    for experiment in dir(experiment_registry):
        if not experiment.startswith("__"):
            print(f"  - {experiment}")


if len(sys.argv) < 2:
    print("Error: Missing experiment name.")
    print_usage()
    sys.exit(1)


def collect_garbage():
    # Clear deleted GPU items
    for _ in range(3):
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser(description="Run a Seahorse experiment")
    parser.add_argument("name", type=str, help="Name of the experiment to run")
    args = parser.parse_args()

    # Get the experiment function from the experiment_registry module
    if not hasattr(experiment_registry, args.name):
        raise ValueError(f"Experiment '{args.name}' not found in the registry")

    print_gpu_memory_usage()

    print(f"Running experiment '{args.name}'")
    experiment_func = getattr(experiment_registry, args.name)

    for i, run_config in enumerate(experiment_func()):
        if isinstance(run_config, tuple):
            run_config, optuna_trial = run_config
        else:
            optuna_trial = None

        print(f"--------- Running experiment {args.name} ({i + 1}) ----------")
        if not isinstance(run_config, RunConfig):
            raise TypeError(f"Experiment '{args.name}' did not return a RunConfig")
        pprint(run_config)

        try:
            run_training(run_config, optuna_trial=optuna_trial)
        except Exception as e:
            print(f"Error running experiment {args.name}: {e}")
            print("Continuing to the next experiment")

        print("Running garbage collection")
        collect_garbage()
        print_gpu_memory_usage()


if __name__ == "__main__":
    main()
