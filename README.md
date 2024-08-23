# Seahorse: "A small VLLM for research"
This repo contains all of the code used in the entire development process for Seahorse, a VLLM based on Phi3 and CLIP.

## A quick tour
```
seahorse/
├── seahorse/         # Main Python package for the project
│   ├── config/       # HF-style configuration files for SeahorseModel
│   ├── data/         # Data preprocessing and loading (e.g. datasets, collators, etc.)
│   ├── eval/         # Evaluation code and utilities (e.g. benchmark scoring, etc.)
│   ├── experiments/  # Experiment configuration and launching
│   ├── models/       # Model architectures and construction
│   └── train/        # Training script and custom HF Trainer class
├── tests/            # Sanity-preserving unit tests for the project
└── Makefile          # Simple task management (`run-experiment`, `test`, etc.)
```

## Running an experiment
This project uses a [`Makefile`](Makefile) to manage tasks. Tasks in the Makefile rely on [uv](https://github.com/astral-sh/uv) for dependency management.

To run an experiment, use the `run-experiment` task. For example:
```bash
make run-experiment pretrain
```
> **Tip:** This is equivalent to running:
> ```bash
> uv run python seahorse/experiments/run_experiment.py pretrain
> ```
> If you prefer not to use `uv`, you can manually install the project dependencies and then run:
> ```bash
> python seahorse/experiments/run_experiment.py pretrain
> ```

This will look for the function `pretrain()` in the [experiment registry](seahorse/experiments/experiment_registry.py) and execute it to create a (set of) experiment configuration(s). Then for each of those configurations, a training run will be launched.


## Running tests
To run the unit tests for the project, use the `test` task:
```bash
make test
```