import os
from typing import Literal

from datasets.arrow_dataset import Dataset as HFDataset
from optuna import Trial
from pydantic import BaseModel

import wandb
from seahorse.data.collator import SeahorseDataCollator
from seahorse.data.dataset_construction import DataConfig, construct_dataset
from seahorse.experiments.experiment_utils import (
    enable_transformers_logging,
    print_gpu_memory_usage,
)
from seahorse.experiments.optuna_callback import OptunaCallback
from seahorse.models.construction import ModelingConfig, construct_seahorse
from seahorse.train.seahorse_trainer import SeahorseTrainer, SeahorseTrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WANDB_PROJECT = "seahorse"


class RunConfig(BaseModel):
    modeling_config: ModelingConfig
    training_arguments: SeahorseTrainingArguments
    data_config: DataConfig
    job_type: Literal["training", "pretrain", "instr-tune"] = "training"


def run_training(run_config: RunConfig, optuna_trial: Trial | None = None) -> None:
    with wandb.init(  # type: ignore
        project=WANDB_PROJECT,
        config=run_config.model_dump(),
        job_type=run_config.job_type,
        group=None,
        name=run_config.training_arguments.run_name,
    ) as wandb_run:
        enable_transformers_logging()

        model = construct_seahorse(run_config.modeling_config)
        print_gpu_memory_usage()

        ds: HFDataset = construct_dataset(run_config.data_config, model)
        ds_splits = ds.train_test_split(test_size=5000, seed=42)
        train_ds = ds_splits["train"]
        eval_ds = ds_splits["test"]

        collator = SeahorseDataCollator(model=model)

        args: SeahorseTrainingArguments = run_config.training_arguments

        callbacks = []
        if optuna_trial is not None:
            callbacks.append(OptunaCallback(trial=optuna_trial))

        trainer = SeahorseTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            callbacks=callbacks,
        )

        try:
            trainer.train()
        finally:
            model.to("cpu")  # needed in order to actually free the GPU memory for follow up jobs
            del train_ds, model, collator, trainer
