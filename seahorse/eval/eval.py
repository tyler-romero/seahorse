import torch

from seahorse.eval.lmms_runner import run_lmms_eval
from seahorse.models.seahorse import SeahorseModel
from wandb import sdk as wandb_sdk


@torch.inference_mode()
def run_midtraining_evaluation(
    model: SeahorseModel, wandb_run: wandb_sdk.wandb_run.Run | None = None
) -> dict[str, float]:
    """
    An evaluation function that runs evaluation on the model and logs the results to W&B.
    This is expected to be called during training to monitor the model's progress. To that end,
    only *validation* sets are to be used here.
    """
    is_training = model.training
    model.eval()

    # Do evaluations on VALIDATION sets only
    results = run_lmms_eval(model, task="vqav2_val_lite")
    results |= run_lmms_eval(model, task="ok_vqa_val2014_lite")

    results = {f"eval/{k}": v for k, v in results.items()}

    print("All Evaluation Results:", results)
    if wandb_run is not None:
        wandb_run.log(results)

    if is_training:
        print("Returning model to training mode.")
        model.train()
    else:
        print("Leaving model in evaluation mode.")
    return results
