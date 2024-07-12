import optuna
from optuna import Trial
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments


class OptunaCallback(TrainerCallback):
    def __init__(self, trial: Trial):
        print("Using OptunaCallback")
        self.trial = trial
        self.direction = self.trial.study.direction
        self.eval_step = 0
        self.is_pruned = False
        self.best_metric = None

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        assert (
            args.metric_for_best_model is not None
        ), "OptunaCallback requires metric_for_best_model is defined"
        if args.greater_is_better and self.direction == optuna.study.StudyDirection.MINIMIZE:
            raise ValueError(
                "greater_is_better=True but optuna study direction is minimize, please set greater_is_better=False"
            )
        elif not args.greater_is_better and self.direction == optuna.study.StudyDirection.MAXIMIZE:
            raise ValueError(
                "greater_is_better=False but optuna study direction is maximize, please set greater_is_better=True"
            )
        assert (
            args.eval_strategy != IntervalStrategy.NO
        ), "OptunaCallback requires evaluation IntervalStrategy of steps or epoch"

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float],
        **kwargs,
    ):
        print("Optuna step {self.eval_step}")

        metric_to_check: str = args.metric_for_best_model
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            print(
                f"optuna required metric_for_best_model, but did not find {metric_to_check} so optuna reporting is disabled"
            )
            return

        print(f"Reporting metric {metric_to_check} to Optuna: {metric_value}")
        self.trial.report(metric_value, self.eval_step)
        self.eval_step += 1

        if self.best_metric is None:
            self.best_metric = metric_value
        elif self.direction == optuna.study.StudyDirection.MAXIMIZE:
            if metric_value > self.best_metric:
                self.best_metric = metric_value
        else:
            if metric_value < self.best_metric:
                self.best_metric = metric_value

        if self.trial.should_prune():
            print("Determined trial should be prined, stopping training early.")
            self.is_pruned = True
            control.should_training_stop = True  # stop training early

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if self.is_pruned:
            self.trial.study.tell(self.trial, state=optuna.trial.TrialState.PRUNED)
            print("Trial was pruned.")
        else:
            self.trial.study.tell(self.trial, self.best_metric)
            print(f"Trial completed, best_metric={self.best_metric}.")
