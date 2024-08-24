import random
from unittest.mock import patch

import pytest

from seahorse.eval.benchmarks import cv_bench, pope
from seahorse.eval.eval import run_evaluation
from seahorse.eval.eval_utils import PredictionsWithMetadata
from seahorse.eval.lmms_runner import run_lmms_eval
from seahorse.models.seahorse import SeahorseModel


def test_run_evaluation(seahorse: SeahorseModel, capsys):
    """Run the entire evaluation pipeline."""
    with capsys.disabled():
        run_evaluation(seahorse)


@pytest.mark.parametrize("answer_format", ["choices", "letters", "letters_with_parens"])
def test_eval_cv_bench_with_random_predictions(seahorse: SeahorseModel, capsys, answer_format: str):
    bs = 16

    def choice_fn(choices: list[list[str]], format: str):
        letters = "abcdef"
        if format == "choices":
            return [random.choice(choice) for choice in choices]
        elif format == "letters":
            return [random.choice(letters[: len(choice)]) for choice in choices]
        elif format == "letters_with_parens":
            return [f"({random.choice(letters[: len(choice)])})" for choice in choices]
        else:
            raise ValueError(f"Unknown format: {format}")

    # Mock the make_batch_predictions function to return random predictions
    with capsys.disabled():
        with patch.object(cv_bench, "make_batch_predictions") as mock_predictions:
            mock_predictions.side_effect = lambda choices, **kwargs: PredictionsWithMetadata(
                predictions=choice_fn(choices, format=answer_format),
                predictions_with_special_tokens=["foo"] * len(choices),
            )
            results = cv_bench.eval_cv_bench(seahorse, split="test", eval_batch_size=bs)

    combined_accuracy = results["cv_bench/combined_accuracy"]
    assert (
        0.3 < combined_accuracy < 0.5
    ), f"Random predictions get about 0.42 combined_accuracy on cv-bench... got {combined_accuracy=}"


def test_eval_pope_with_random_predictions(seahorse: SeahorseModel, capsys):
    bs = 16

    def rand_yn_prediction(prompts, **kwargs) -> list[dict[str, float]]:
        preds = []
        for _ in prompts:
            p = random.random()
            preds.append({"yes": p, "no": 1 - p})
        return preds

    # Mock the make_batch_predictions function to return random predictions
    with capsys.disabled():
        with patch.object(pope, "next_token_probs") as mock_predictions:
            mock_predictions.side_effect = rand_yn_prediction
            results = pope.eval_pope(seahorse, split="random", eval_batch_size=bs)

    acc = results["pope-random/accuracy"]
    roc_auc = results["pope-random/roc_auc"]
    assert 0.45 < acc < 0.55, f"Random predictions get about 0.42 acc on pope-random... got {acc=}"
    assert (
        0.45 < roc_auc < 0.55
    ), f"Random predictions get about 0.42 roc_auc on pope-random... got {roc_auc=}"


def test_eval_pope_with_model(seahorse: SeahorseModel, capsys):
    # Mock the make_batch_predictions function to return random predictions
    with capsys.disabled():
        results = pope.eval_pope(seahorse, split="random")

    assert 0.45 < results["pope-random/accuracy"] < 0.55
    assert 0.45 < results["pope-random/roc_auc"] < 0.55


@pytest.mark.parametrize("task", ["vqav2_val_lite", "ok_vqa_val2014_lite"])
def test_run_lmms_eval(seahorse: SeahorseModel, capsys, task: str):
    with capsys.disabled():
        print("Running LMMS evaluation on", task)
        results = run_lmms_eval(seahorse, task=task, limit=10)
    assert isinstance(results, dict)
    if "vqav2" in task:
        assert "vqav2_val_lite/exact_match,none" in results
    elif "ok_vqa" in task:
        assert "ok_vqa_val2014_lite/exact_match,none" in results
