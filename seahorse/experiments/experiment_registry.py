from copy import deepcopy

import optuna
from optuna.trial import Trial
from transformers.trainer_utils import IntervalStrategy

from seahorse.data.dataset_construction import DataConfig, DatasetSpec
from seahorse.experiments.experiment_utils import randstr
from seahorse.models.construction import ModelingConfig
from seahorse.train.seahorse_trainer import SeahorseTrainingArguments
from seahorse.train.train import RunConfig


def get_default_job_config() -> RunConfig:
    return RunConfig(
        modeling_config=ModelingConfig(),
        data_config=DataConfig(dataset_specs=[DatasetSpec(name="llava_v1_5_mix665k_ift")]),
        training_arguments=SeahorseTrainingArguments(
            run_name="baseline",
            output_dir="./results/baseline",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,  # if enabled, slows training by ~20%
            torch_compile=False,  # doesnt make a big difference either way
            bf16=True,
            optim="adamw_torch_fused",
            learning_rate=1e-4,
            embedding_learning_rate=1e-5,
            weight_decay=0.00,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            dataloader_num_workers=8,
            # Eval args
            eval_strategy=IntervalStrategy.STEPS,
            eval_steps=2500,
            eval_on_start=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),  # type: ignore
    )


def pretrain() -> list[RunConfig]:
    base_job = get_default_job_config()

    # Model is frozen except for projection layer
    base_job.modeling_config.peft_config = None
    base_job.modeling_config.seahorse_config.freeze_llm_input = True
    base_job.modeling_config.seahorse_config.freeze_llm_output = True

    # Llava Pretrain Dataset
    base_job.data_config = DataConfig(dataset_specs=[DatasetSpec(name="llava_pretrain_cc3m")])

    base_job.training_arguments.run_name = f"pretrain-{randstr()}"
    base_job.training_arguments.embedding_learning_rate = None
    base_job.job_type = "pretrain"
    return [base_job]


def pretrain_sweep() -> list[tuple[RunConfig, Trial]]:
    pretrain_base: RunConfig = pretrain()[0]

    sampler = optuna.samplers.QMCSampler(qmc_type="sobol")  # Quasi-Monte Carlo sampler
    pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=4, interval_steps=1)
    study = optuna.create_study(
        sampler=sampler, pruner=pruner, direction="minimize", study_name="pretrain_sweep"
    )
    # study.enqueue_trial(
    #     params={
    #         "with_image_patch_positional_embeddings": None,
    #         "learning_rate": 1e-3,
    #         "freeze_llm_input": True,
    #     }
    # )

    n_trials = 100
    for i in range(n_trials):
        new_job = deepcopy(pretrain_base)
        trial = study.ask()
        new_job.training_arguments.run_name = f"pretrain-{i}-{randstr()}"
        new_job.training_arguments.output_dir = f"./results/{new_job.training_arguments.run_name}"

        new_job.modeling_config.seahorse_config.freeze_llm_input = trial.suggest_categorical(
            "freeze_llm_input", [True, False]
        )

        new_job.modeling_config.seahorse_config.with_image_patch_positional_embeddings = (  # type: ignore
            trial.suggest_categorical(
                "with_image_patch_positional_embeddings",
                [None, "xy", "each"],
            )
        )
        new_job.training_arguments.learning_rate = trial.suggest_float(
            "learning_rate", 8e-4, 1e-2, log=False
        )

        yield new_job, trial


def instr_tune() -> list[RunConfig]:
    base_job = get_default_job_config()

    # Peft model is used

    # Llava Pretrain Dataset
    base_job.data_config = DataConfig(dataset_specs=[DatasetSpec(name="llava_v1_5_mix665k_ift")])

    base_job.training_arguments.run_name = f"ift-{randstr()}"
    base_job.job_type = "instr-tune"
    return [base_job]


def find_good_learning_rate() -> list[RunConfig]:
    """lr of 1e-4 is good, with emb_lr at either 1/5th or 1/10th of that"""
    jobs = []
    for lr in [1e-3, 1e-4, 1e-5, 5e-4, 5e-5]:
        for emb_lr in [None, 1 / 10, 1 / 5]:
            emb_lr = lr * emb_lr if emb_lr is not None else None

            job = get_default_job_config()
            job.training_arguments.learning_rate = lr
            job.training_arguments.run_name = f"baseline-lr{lr}-emblr{emb_lr}-{randstr()}"
            jobs.append(job)
    return jobs


def schedulefree_sweep() -> list[RunConfig]:
    base_job = get_default_job_config()
    base_job.training_arguments.optim = "adamw_schedulefree"
    base_job.training_arguments.lr_scheduler_type = "constant"
    base_job.training_arguments.warmup_ratio = 0.0
    base_job.training_arguments.warmup_steps = 0
    # base_job.training_arguments.max_steps = 10000  # shorter experiments since no schedule

    # The schedulefree authors suggest a learning rate 1-10x higher than AdamW
    for lr in [5e-4]:  # 1e-4,
        for warmup_steps in [500]:  # 0, 100, 1000
            for beta1 in [0.9]:  # 0.95
                job = deepcopy(base_job)
                job.training_arguments.learning_rate = lr
                job.training_arguments.embedding_learning_rate = lr * 1 / 10
                job.training_arguments.warmup_steps = warmup_steps
                job.training_arguments.adam_beta1 = beta1
                job.training_arguments.run_name = (
                    f"schedulefree-lr{lr}-warmup{warmup_steps}-b1{beta1}-{randstr()}"
                )
                yield job
