from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from seahorse.eval.eval import run_midtraining_evaluation
from seahorse.models.seahorse import SeahorseModel
from seahorse.train.seahorse_trainer import SeahorseTrainingArguments
from wandb import sdk as wandb_sdk


class SeahorseEvalCallback(TrainerCallback):
    def __init__(self, model: SeahorseModel, wandb_run: wandb_sdk.wandb_run.Run | None = None):
        self.model = model
        self.wandb_run = wandb_run

    def on_evaluate(
        self,
        args: SeahorseTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after an evaluation phase.
        """
        print("Running midtraining evaluation via SeahorseEvalCallback")
        run_midtraining_evaluation(self.model, self.wandb_run)
