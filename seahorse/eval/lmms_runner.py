import tempfile
from argparse import Namespace

import lmms_eval
import torch
from lmms_eval.evaluator import evaluate
from lmms_eval.utils import run_task_tests

from seahorse.eval.lmms_eval_wrapper import SeahorseLmms
from seahorse.utils.rng import isolate_rng

DEFAULT_EVAL_GEN_KWARGS = {
    "max_new_tokens": 16,
}


@torch.inference_mode()
def run_lmms_eval(model, task: str, limit: int | None = None, gen_kwargs: dict | None = None):
    with isolate_rng():  # So eval frequency doesnt affect training rng
        lmms_eval.tasks.initialize_tasks()  # quick, has internal mechanism to prevent re-initialization

        seahorse_lmms = SeahorseLmms(model=model)
        results = run_evaluation(
            lm=seahorse_lmms,
            tasks=[task],
            limit=limit,
            gen_kwargs=gen_kwargs if gen_kwargs is not None else DEFAULT_EVAL_GEN_KWARGS,
        )
        results = format_eval_results(results["results"])
        return results


@torch.inference_mode()
def run_evaluation(
    lm,
    *,
    model_name: str = "no-name",
    tasks=[],
    num_fewshot=None,
    limit=None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    show_task_to_terminal: bool = False,
    log_samples: bool = True,
    gen_kwargs: dict | None = None,
    predict_only: bool = False,
) -> dict:
    """Evaluate the model on a list of tasks"""
    assert tasks != [], "No tasks specified, or no tasks found. Please verify the task names."

    task_dict = lmms_eval.tasks.get_task_dict(tasks, model_name=model_name)
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if type(task_obj) == tuple:
            group, task_obj = task_obj
            if task_obj is None:
                continue
        lm.task_dict[task_name] = task_obj.dataset

        config = task_obj._config
        if config["output_type"] == "generate_until" and gen_kwargs:
            config["generation_kwargs"].update(gen_kwargs)

        if predict_only:
            log_samples = True
            print(f"Processing {task_name} in output-only mode. Metrics will not be calculated!")
            # we have to change the class properties post-hoc. This is pretty hacky.
            task_obj.override_metric(metric_name="bypass")

        if num_fewshot is not None:
            if config["num_fewshot"] == 0:
                print(
                    f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                )
            else:
                default_num_fewshot = config["num_fewshot"]
                print(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )

                task_obj._config["num_fewshot"] = num_fewshot

    if check_integrity:
        run_task_tests(task_list=tasks)

    results: dict = evaluate(  # type: ignore
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        show_task_to_terminal=show_task_to_terminal,
        log_samples=log_samples,
        cli_args=Namespace(output_path=tempfile.gettempdir()),
    )

    if lm.rank == 0:
        # add info about the model and few shot config
        results["model_configs"] = {
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "gen_kwargs": gen_kwargs,
        }
        return results
    else:
        return None


def format_eval_results(results) -> dict:
    formatted_results = {}
    for task_name, task_results in results.items():
        for metric_name, metric_value in task_results.items():
            if isinstance(metric_value, int | float):
                formatted_results[f"{task_name}/{metric_name}"] = metric_value
    return formatted_results


class CliArgs:
    """Hate this, just a quick hack to pass the output path so lmms_eval doesnt throw an error."""

    def __init__(self, output_path: str | None = None):
        self.output_path = output_path or tempfile.mkdtemp()
