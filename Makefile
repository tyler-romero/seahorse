.PHONY: help test run-experiment
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
# Script to create useful "make help" messages
import re
import sys
target_list = []
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target_list.append(match.groups())

target_list.sort()
for target, help in target_list:
	print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help: ## Print a description of all targets
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

test: ## Run tests (may require a GPU)
	CUDA_LAUNCH_BLOCKING=1 uv run pytest --tb=native --show-capture=stdout \
		--log-cli-level DEBUG --durations=10 $(filter-out $@,$(MAKECMDGOALS))

run-experiment: ## Run an experiment defined in experiments/experiment_registry.py (e.g. `make run-experiment pretrain`)
	uv run python seahorse/experiments/run_experiment.py $(filter-out $@,$(MAKECMDGOALS))