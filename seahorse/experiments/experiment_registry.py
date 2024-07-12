from copy import deepcopy

import optuna
from optuna.trial import Trial
from peft.tuners.lora import LoraConfig
from transformers.trainer_utils import IntervalStrategy

from seahorse.config.configuration_seahorse import SeahorseConfig
from seahorse.config.experiment_config import ModelingConfig, RunConfig
from seahorse.data.dataset_construction import DataConfig, DatasetSpec
from seahorse.experiments.experiment_utils import randstr
from seahorse.train.seahorse_trainer import SeahorseTrainingArguments


def get_default_job_config() -> RunConfig:
    return RunConfig(
        modeling_config=ModelingConfig(
            seahorse_config=SeahorseConfig(
                language_model="microsoft/Phi-3-mini-4k-instruct",
                lm_revision="c1358f8a35e6d2af81890deffbbfa575b978c62f",  # phi3.5
                vision_encoder="vit_base_patch16_clip_224.openai",
                with_image_patch_positional_embeddings=None,
                freeze_lm_input_output=False,
            ),
            peft_config=LoraConfig(
                r=8,  # 128 caused divergence
                lora_alpha=8,
                # A regex that matches the names of linear layers within Phi3
                target_modules=r"^language_model.*(?:down_proj|gate_up_proj|o_proj|qkv_proj)$",
                modules_to_save=[
                    "vision_projector",
                    "img_pos_embed",
                    "language_model.lm_head",
                    "language_model.model.embed_tokens",
                ],
                lora_dropout=0.05,
                bias="none",
                init_lora_weights="pissa_niter_15",  # type: ignore
                use_dora=False,  # using dora led to no improvement but was 2x as slow
                task_type="CAUSAL_LM",
            ),
        ),
        data_config=DataConfig(dataset_specs=[DatasetSpec(name="llava_pretrain_cc3m")]),
        training_arguments=SeahorseTrainingArguments(
            run_name="baseline",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            gradient_checkpointing=False,  # if enabled, slows training by ~20%
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
            metric_for_best_model="eval/loss",
            greater_is_better=False,
        ),  # type: ignore
    )


def pretrain() -> list[RunConfig]:
    base_job = get_default_job_config()

    # Model is frozen except for projection layer
    base_job.modeling_config.peft_config = None
    base_job.modeling_config.seahorse_config.freeze_lm_input_output = True

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

    n_trials = 100
    new_job = deepcopy(pretrain_base)
    for i in range(n_trials):
        trial = study.ask()
        new_job.training_arguments.run_name = f"pretrain-{i}-{randstr()}"

        new_job.modeling_config.seahorse_config.with_image_patch_positional_embeddings = (  # type: ignore
            trial.suggest_categorical(
                "with_image_patch_positional_embeddings",
                [None, "xy", "each"],
            )
        )
        new_job.training_arguments.learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 5e-3, log=True
        )

        new_job.training_arguments.optim = trial.suggest_categorical(
            "optim", ["adamw_torch_fused", "adamw_schedulefree"]
        )
        if new_job.training_arguments.optim == "adamw_schedulefree":
            new_job.training_arguments.lr_scheduler_type = "constant"
            new_job.training_arguments.warmup_ratio = 0.0
            new_job.training_arguments.warmup_steps = 2200

        yield new_job, trial


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
