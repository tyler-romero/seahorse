from typing import Literal

import pandas as pd
import torch
from datasets.arrow_dataset import Dataset as HFDataset
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from seahorse.data.data_prep.eval.pope import make_pope
from seahorse.eval.eval_utils import next_token_probs, print_df
from seahorse.models.seahorse import SeahorseModel


def _compute_pope_metrics(df: pd.DataFrame) -> dict[str, float]:
    return {
        "accuracy": df["is_correct"].sum() / len(df),
        "roc_auc": roc_auc_score(df["gt_yes_labels"], df["model_yes_probs"]),  # type: ignore
        "average_precision": average_precision_score(df["gt_yes_labels"], df["model_yes_probs"]),  # type: ignore
    }


@torch.inference_mode()
def eval_pope(
    seahorse: SeahorseModel,
    *,
    split: Literal["popular", "adversarial", "random"],
    eval_batch_size: int = 24,
) -> dict[str, float]:
    predictions, is_correct, model_yes_probs, gt_yes_labels = [], [], [], []

    eval_dataset: HFDataset = make_pope(
        seahorse, split=split, ablate_images=False, load_from_cache_file=False
    )
    total_examples = len(eval_dataset)

    for example_batch in tqdm(
        eval_dataset.iter(batch_size=eval_batch_size),
        desc="Running Pope evaluation",
        total=total_examples // eval_batch_size,
    ):
        example_batch: dict[str, list]
        has_images = example_batch["image"] is not None
        yes_no_probs_batch = next_token_probs(
            seahorse=seahorse,
            prompts=example_batch["text"],
            images=example_batch["image"] if has_images else None,
            query_strings=["yes", "no"],
        )

        for yes_no_probs, answer in zip(yes_no_probs_batch, example_batch["answer"]):
            predicted_yes_no = "yes" if yes_no_probs["yes"] > yes_no_probs["no"] else "no"
            is_correct.append(predicted_yes_no == answer)
            model_yes_probs.append(yes_no_probs["yes"])
            gt_yes_labels.append(1 if answer == "yes" else 0)
            predictions.append(predicted_yes_no)

    df = pd.DataFrame(
        {
            "category": eval_dataset["category"],
            "text": eval_dataset["text"],
            "predicted_yes_no": predictions,
            "is_correct": is_correct,
            "model_yes_probs": model_yes_probs,
            "gt_yes_labels": gt_yes_labels,
        }
    )

    results = _compute_pope_metrics(df)
    prefixed_results = {f"pope-{split}/{k}": v for k, v in results.items()}
    print(f"Pope-{split} Results:", prefixed_results)
    print(f"Pope-{split} Samples:")
    print_df(df.sample(n=15))

    return prefixed_results
