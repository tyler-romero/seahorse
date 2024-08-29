import os
from enum import StrEnum

from datasets.arrow_dataset import Dataset as HFDataset
from optuna import Trial
from pydantic import BaseModel
from transformers.integrations import WandbCallback

import wandb
from seahorse.data.collator import SeahorseDataCollator
from seahorse.data.dataset_construction import DataConfig, construct_dataset
from seahorse.eval.eval_callback import SeahorseEvalCallback
from seahorse.experiments.experiment_utils import (
    enable_transformers_logging,
    print_gpu_memory_usage,
)
from seahorse.experiments.optuna_callback import OptunaCallback
from seahorse.models.construction import ModelingConfig, construct_seahorse
from seahorse.train.efficiency_callback import EfficiencyCallback
from seahorse.train.seahorse_trainer import SeahorseTrainer, SeahorseTrainingArguments
from seahorse.utils.profiling import ProfCallback, maybe_profile

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WANDB_PROJECT = "seahorse"


class JobType(StrEnum):
    PRETRAIN = "pretrain"
    INSTR_TUNE = "instr_tune"
    PROFILE = "profile"


class RunConfig(BaseModel):
    modeling_config: ModelingConfig
    training_arguments: SeahorseTrainingArguments
    data_config: DataConfig
    job_type: JobType  # TODO: refactor this to training_arguments.job_type


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

        ift_mask = run_config.job_type == JobType.INSTR_TUNE
        user_token_id: int = model.tokenizer.convert_tokens_to_ids("<|user|>")  # type: ignore
        assistant_token_id: int = model.tokenizer.convert_tokens_to_ids("<|assistant|>")  # type: ignore
        collator = SeahorseDataCollator(
            model=model,
            ift_mask=ift_mask,
            user_token=user_token_id,
            assistant_token=assistant_token_id,
        )

        args: SeahorseTrainingArguments = run_config.training_arguments

        callbacks = [
            EfficiencyCallback(),
            WandbCallback(),
            SeahorseEvalCallback(model=model, wandb_run=wandb_run),
        ]

        trainer = SeahorseTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            callbacks=callbacks,
        )

        if optuna_trial is not None:
            trainer.add_callback(OptunaCallback(trial=optuna_trial))

        do_profile = run_config.job_type == JobType.PROFILE
        with maybe_profile(do_profile) as prof:
            if do_profile:
                trainer.add_callback(ProfCallback(prof))

            try:
                trainer.train()
            finally:
                # needed in order to actually free the GPU memory for follow up jobs
                model.to("cpu")
                del train_ds, model, collator, trainer
