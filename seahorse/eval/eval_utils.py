from typing import NamedTuple

import numpy as np
import pandas as pd
from PIL.Image import Image as PILImage
from scipy.special import logsumexp  # type: ignore
from transformers.modeling_outputs import CausalLMOutputWithPast

from seahorse.models.seahorse import SeahorseModel


class PredictionsWithMetadata(NamedTuple):
    predictions: list[str]
    predictions_with_special_tokens: list[str]


def make_batch_predictions(
    seahorse: SeahorseModel,
    prompts: list[str],
    images: list[PILImage] | None,
    max_new_tokens: int = 20,
    return_only_new: bool = True,
    **kwargs,  # other metadata to help with mocks
) -> PredictionsWithMetadata:
    tokens = seahorse.tokenize_text(prompts)
    if images is not None and images[0] is not None:
        preprocessed_imgs = seahorse.preprocess_image(images).to(seahorse.device)
    else:
        preprocessed_imgs = None

    outputs = seahorse.generate(
        input_ids=tokens.input_ids.to(seahorse.device),
        pixel_values=preprocessed_imgs,
        attention_mask=tokens.attention_mask.to(seahorse.device),
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    if return_only_new:
        outputs = outputs[:, tokens.input_ids.shape[1] :]
    predicted_texts = seahorse.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    predictions_with_special_tokens = seahorse.tokenizer.batch_decode(
        outputs, skip_special_tokens=False
    )
    return PredictionsWithMetadata(
        predictions=predicted_texts,
        predictions_with_special_tokens=predictions_with_special_tokens,
    )


def next_token_probs(
    seahorse: SeahorseModel,
    prompts: list[str],
    images: list[PILImage] | None,
    query_strings: list[str],
) -> list[dict[str, float]]:
    tokens = seahorse.tokenize_text(prompts)
    if images is not None and images[0] is not None:
        preprocessed_imgs = seahorse.preprocess_image(images).to(seahorse.device)
    else:
        preprocessed_imgs = None

    out: CausalLMOutputWithPast = seahorse(
        input_ids=tokens.input_ids.to(seahorse.device),
        pixel_values=preprocessed_imgs,
        attention_mask=tokens.attention_mask.to(seahorse.device),
    )
    logprobs = out.logits[:, -1, :].log_softmax(dim=-1)  # Get logprobs for the last token

    results = []
    for ex_logprobs in logprobs:
        top_k_logprobs, top_k_indices = ex_logprobs.topk(k=20, dim=-1)
        top_k_tokens = seahorse.tokenizer.batch_decode(top_k_indices)
        top_k_results = dict(zip(top_k_tokens, top_k_logprobs.cpu().tolist()))

        query_string_probs = {s: get_single_tok_prob(top_k_results, s) for s in query_strings}
        results.append(query_string_probs)
    return results


def get_single_tok_prob(token_logprobs: dict[str, float], token: str) -> float:
    """
    Aggregate over all logprobs for a token-string, regardless of case or leading spaces.
    """
    normalized_token = token.lower().strip().strip("▁_")  # some tokenizers use "▁" for space
    token_logprobs = {k.lower().strip().strip("▁_"): v for k, v in token_logprobs.items()}
    matching_token_logprobs = [p for t, p in token_logprobs.items() if t == normalized_token]
    if len(matching_token_logprobs) > 0:
        token_prob = np.exp(logsumexp(matching_token_logprobs))
    else:
        token_prob = 0.0
    return token_prob


def print_df(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        print("-" * 80)
        for column, value in row.items():
            print(f" - {column}: {value}")
        print("-" * 80)
