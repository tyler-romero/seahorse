import re

import pandas as pd
import torch
from datasets.arrow_dataset import Dataset as HFDataset
from tqdm import tqdm

from seahorse.data.data_prep.eval.cv_bench import make_cv_bench
from seahorse.eval.eval_utils import make_batch_predictions, print_df
from seahorse.models.seahorse import SeahorseModel


def _determine_match(
    predicted_text: str, answer_letter: str, answer_value: str, choices: list[str]
) -> int:
    """
    Returns 1 if the predicted text matches the answer, 0 otherwise.

    answer_letter: e.g. "(A)"
    answer_value: e.g. "door"
    choices: e.g. ["door", "window", "wall", "floor", "ceiling"]
    """
    assert answer_letter.startswith("(") and answer_letter.endswith(")") and len(answer_letter) == 3

    def fuzzy_match_target(text: str, target: str, incorrect_choices: list | str) -> bool:
        target_pattern = rf"\b{re.escape(target)}\b"
        incorrect_choices_patterns = [rf"\b{re.escape(choice)}\b" for choice in incorrect_choices]
        is_target_matched = re.search(target_pattern, text, re.IGNORECASE) is not None
        is_incorrect_choice_matched = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in incorrect_choices_patterns
        )
        return is_target_matched and not is_incorrect_choice_matched

    incorrect_choices = [choice for choice in choices if choice != answer_value]
    incorrect_letters = [ltr for ltr in "ABCDEF" if ltr != answer_letter[1]]

    result = 0
    if (
        fuzzy_match_target(
            predicted_text,
            target=answer_value,  # e.g. "door"
            incorrect_choices=incorrect_choices,  # but not "window", "wall", "floor", "ceiling"
        )
        or fuzzy_match_target(
            predicted_text,
            target=answer_letter,  # e.g. (A)
            incorrect_choices=[f"({ltr})" for ltr in incorrect_letters],  # but not (B), (C), ...
        )
        or fuzzy_match_target(
            predicted_text,
            target=answer_letter[1],  # e.g. "A"
            incorrect_choices=incorrect_letters,  # but not "B", "C", ...
        )
    ):
        result = 1
    return result


def _calculate_accuracy_for_source(df: pd.DataFrame, source: str):
    source_df = df[df["source"] == source]
    accuracy = source_df["result"].mean()  # Assuming 'result' is 1 for correct and 0 for incorrect
    return accuracy


def _compute_cv_bench_acc(df) -> dict[str, float]:
    # Calculate CV-Bench accuracy as specified
    # https://huggingface.co/datasets/nyu-visionx/CV-Bench
    accuracy_2d_ade = _calculate_accuracy_for_source(df, "ADE20K")
    accuracy_2d_coco = _calculate_accuracy_for_source(df, "COCO")
    accuracy_3d_omni = _calculate_accuracy_for_source(df, "Omni3D")
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni
    combined_accuracy = (accuracy_2d + accuracy_3d) / 2
    return {
        "accuracy_2d_ade": accuracy_2d_ade,
        "accuracy_2d_coco": accuracy_2d_coco,
        "accuracy_3d_omni": accuracy_3d_omni,
        "accuracy_2d": accuracy_2d,
        "accuracy_3d": accuracy_3d,
        "combined_accuracy": combined_accuracy,
    }


@torch.inference_mode()
def eval_cv_bench(
    seahorse: SeahorseModel,
    *,
    split: str,  # NOTE: test is the only split... there is no validation split
    eval_batch_size: int = 24,
) -> dict[str, float]:
    predictions = []
    results = []

    eval_dataset: HFDataset = make_cv_bench(
        seahorse,
        split=split,
        ablate_images=False,  # "no_img"
        load_from_cache_file=False,
    )
    total_examples = len(eval_dataset)

    for example_batch in tqdm(
        eval_dataset.iter(batch_size=eval_batch_size),
        desc="Running CV-Bench evaluation",
        total=total_examples // eval_batch_size,
    ):
        example_batch: dict[str, list]
        has_images = example_batch["image"] is not None
        predicted_texts = make_batch_predictions(
            seahorse=seahorse,
            prompts=example_batch["text"],
            images=example_batch["image"] if has_images else None,
            choices=example_batch["choices"],
        )

        for predicted_text, answer, answer_value, choices in zip(
            predicted_texts.predictions,
            example_batch["answer"],  # e.g. "(A)"
            example_batch["answer_value"],  # e.g. "door"
            example_batch["choices"],
        ):
            result = _determine_match(
                predicted_text, answer_letter=answer, answer_value=answer_value, choices=choices
            )
            predictions.append(predicted_text)
            results.append(result)

    df = pd.DataFrame(
        {
            "source": eval_dataset["source"],
            "text": eval_dataset["text"],
            "prediction": predictions,
            "gt_answer": eval_dataset["answer"],
            "gt_answer_value": eval_dataset["answer_value"],
            "result": results,
        }
    )

    results = _compute_cv_bench_acc(df)
    prefixed_results = {f"cv_bench/{k}": v for k, v in results.items()}
    print("CVBench Eval Results:", prefixed_results)
    print("CVBench Samples:")

    print_df(df.sample(n=15))

    return prefixed_results
