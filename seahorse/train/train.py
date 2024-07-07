import os

from datasets import load_dataset

import wandb
from seahorse.config.experiment_config import RunConfig
from seahorse.data.collator import SeahorseDataCollator
from seahorse.data.dataset import SeahorseDataset
from seahorse.experiments.utils import (
    enable_transformers_logging,
    print_gpu_memory_usage,
)
from seahorse.models.construction import construct_seahorse
from seahorse.train.seahorse_trainer import SeahorseTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WANDB_PROJECT = "seahorse"


def run_training(run_config: RunConfig) -> None:
    with wandb.init(  # type: ignore
        project=WANDB_PROJECT,
        config=run_config.model_dump(),
        job_type="training",
        group=None,
        name=run_config.training_arguments.run_name,
    ) as run:
        enable_transformers_logging()
        ds = load_dataset(  # https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
            "HuggingFaceM4/the_cauldron", "vqav2", split="train"
        )
        train_ds = SeahorseDataset(dataset=ds)

        model = construct_seahorse(run_config.modeling_config)
        print_gpu_memory_usage()

        collator = SeahorseDataCollator(model=model)

        trainer = SeahorseTrainer(
            model=model,
            args=run_config.training_arguments,
            train_dataset=train_ds,
            data_collator=collator,
            # optimizers=(schedule_free_adamw, None)
        )
        trainer.train()

        model.to("cpu")  # needed in order to actually free the GPU memory for follow up jobs
        del train_ds, model, collator, trainer
