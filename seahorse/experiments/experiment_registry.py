from copy import deepcopy

from peft.tuners.lora import LoraConfig

from seahorse.config.configuration_seahorse import SeahorseConfig
from seahorse.config.experiment_config import ModelingConfig, RunConfig
from seahorse.experiments.utils import randstr
from seahorse.train.seahorse_trainer import SeahorseTrainingArguments


def get_default_job_config() -> RunConfig:
    return RunConfig(
        modeling_config=ModelingConfig(
            seahorse_config=SeahorseConfig(
                language_model="microsoft/Phi-3-mini-4k-instruct",
                lm_revision="ff07dc01615f8113924aed013115ab2abd32115b",
                vision_encoder="vit_base_patch16_clip_224.openai",
                with_image_patch_positional_embeddings=None,
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
            dataloader_num_workers=4,
        ),
    )


def find_good_learning_rate() -> list[RunConfig]:
    jobs = []
    for lr in [1e-3, 1e-4, 1e-5, 5e-4, 5e-5]:
        for emb_lr in [1 / 5]:  # None, 1 / 10,
            emb_lr = lr * emb_lr if emb_lr is not None else None

            job = get_default_job_config()
            job.training_arguments.learning_rate = lr
            job.training_arguments.embedding_learning_rate = emb_lr
            job.training_arguments.run_name = f"baseline-lr{lr}-emblr{emb_lr}-{randstr()}"
            jobs.append(job)
    return jobs


def schedulefree_experiment() -> list[RunConfig]:
    base_job = get_default_job_config()
    base_job.training_arguments.optim = "adamw_schedulefree"
    base_job.training_arguments.lr_scheduler_type = "constant"
    base_job.training_arguments.warmup_ratio = 0.0
    base_job.training_arguments.warmup_steps = 0
    base_job.training_arguments.max_steps = 4000  # we can run shorter experiments since no schedule

    # The schedulefree authors suggest a learning rate 1-10x higher than AdamW
    for lr in [1e-3, 1e-4]:
        for emb_lr in [1 / 10]:
            for warmup_steps in [0, 100, 500, 1000]:
                emb_lr = lr * emb_lr if emb_lr is not None else None
                job = deepcopy(base_job)
                job.training_arguments.learning_rate = lr
                job.training_arguments.embedding_learning_rate = emb_lr
                job.training_arguments.warmup_steps = warmup_steps
                job.training_arguments.run_name = (
                    f"schedulefree-lr{lr}-emblr{emb_lr}-warmup{warmup_steps}-{randstr()}"
                )
                yield job


def baseline() -> list[RunConfig]:
    baseline = get_default_job_config()
    return [baseline]
