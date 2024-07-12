from dataclasses import dataclass

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import Trainer, has_length
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments


@dataclass
class SeahorseTrainingArguments(TrainingArguments):
    #
    # Dataset-related args
    #
    label_names: list[str] | None = None  # just looks for strings with "label" in them
    remove_unused_columns: bool = False

    #
    # Training-related args
    #
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False  # if enabled, slows training by ~20%
    torch_compile: bool = False  # doesnt make a big difference either way
    bf16: bool = True
    optim: str = "adamw_torch_fused"
    learning_rate: float = (
        1e-3  # https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/pretrain.sh
    )
    embedding_learning_rate: float | None = None  # Custom. Different lr for embeddings and lm_head.
    weight_decay: float = 0.00
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    dataloader_num_workers: int = 8

    #
    # Eval-related arguments
    #
    eval_strategy: IntervalStrategy = IntervalStrategy.NO
    eval_steps: float | None = None
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    per_device_eval_batch_size: int = 8
    eval_accumulation_steps: int | None = 1
    eval_on_start: bool = False
    prediction_loss_only: bool = True  # b/c we use a custom callback for evaluation

    #
    # Tracking-related arguments
    #
    run_name: str = "default-change-me"
    output_dir: str = "./results/default-change-me"
    save_strategy: str = IntervalStrategy.STEPS
    save_steps: int = 10000
    save_total_limit: int | None = 1
    report_to: str = "wandb"
    logging_dir: str = "./logs"
    logging_steps: int = 100
    include_num_input_tokens_seen: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.output_dir = f"./results/{self.run_name}"


class SeahorseTrainer(Trainer):
    def _get_train_sampler(self) -> torch.utils.data.Sampler | None:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            # get lengths from a property of the dataset instead of from
            # the column of a huggingface dataset. This is a workaround
            # taken from LLaVA's implementation of `Trainer`
            # https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L140
            lengths = self.train_dataset.lengths  # type: ignore
            print("Using LengthGroupedSampler")
            return LengthGroupedSampler(
                batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
            )
        else:
            return super()._get_train_sampler()

    def _get_parameter_groups(self) -> dict:
        # Inspired by unsloth's implementation of `Trainer`
        # Separate the parameters of the model into two groups: one for the embeddings
        # and one for the rest of the model.
        param_groups = {"non_embeddings": {}, "embeddings": {}}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "embed_tokens" in name or "lm_head" in name:
                param_groups["embeddings"][name] = param
                print(f"Adding {name} to embeddings parameter group")
            else:
                param_groups["non_embeddings"][name] = param
        return param_groups

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: SeahorseTrainingArguments, model: PreTrainedModel | None = None
    ):
        if args.optim == "adamw_schedulefree":
            from schedulefree import AdamWScheduleFree

            # Get the adamw args, but replace the optimizer class with AdamWScheduleFree
            args.optim = "adamw_torch"
            _, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
            args.optim = "adamw_schedulefree"  # Reset the optim to its original value
            optimizer_cls = AdamWScheduleFree
            optimizer_kwargs["warmup_steps"] = args.warmup_steps
            return optimizer_cls, optimizer_kwargs
        return Trainer.get_optimizer_cls_and_kwargs(args, model)

    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            parameter_groups = self._get_parameter_groups()

            optimizer_cls, optimizer_kwargs = SeahorseTrainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            lr = optimizer_kwargs["lr"]
            optimizer_grouped_parameters = [
                # Embeddings
                {
                    "params": [
                        p
                        for n, p in parameter_groups["embeddings"].items()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": embedding_learning_rate,
                },
                {
                    "params": [
                        p
                        for n, p in parameter_groups["embeddings"].items()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": embedding_learning_rate,
                },
                # Non-embeddings
                {
                    "params": [
                        p
                        for n, p in parameter_groups["non_embeddings"].items()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [
                        p
                        for n, p in parameter_groups["non_embeddings"].items()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            try:
                self.optimizer.train()  # optimizers like schedulefree need to be put in train mode
            except AttributeError:
                pass
        return self.optimizer

    def evaluate(self, *args, **kwargs) -> dict[str, float]:
        """
        Note that this also wraps the on_evalute method of the SeahorseEvalCallback
        """
        try:
            self.optimizer.eval()  # schedulefree optimizer needs to be put in eval mode
        except AttributeError:
            pass

        output = super().evaluate(*args, **kwargs)

        try:
            self.optimizer.train()  # return to train mode
        except AttributeError:
            pass
        return output

    def predict(self, *args, **kwargs) -> dict[str, float]:
        try:
            self.optimizer.eval()  # schedulefree optimizer needs to be put in eval mode
        except AttributeError:
            pass

        output = super().evaluate(*args, **kwargs)

        try:
            self.optimizer.train()  # return to train mode
        except AttributeError:
            pass
        return output
