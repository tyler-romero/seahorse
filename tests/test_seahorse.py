import time

import pytest
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from seahorse.data.utils import random_pil
from seahorse.models.seahorse import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, SeahorseModel


@pytest.fixture(scope="module")
def seahorse():
    """
    Returns a SeahorseModel instance, with the model loaded on the appropriate
    device and dtype. This model is cached and reused across all tests in this module.
    """
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = SeahorseModel(config=None).to(device=device, dtype=dtype)
    print(f"SeahorseModel loaded in {time.time() - start:.2f}s on {device} with {dtype}")
    return model


def test_seahorse_batched_inference(seahorse: SeahorseModel):
    prompt = DEFAULT_IMAGE_TOKEN + "\nIs there a seahorse?"
    tokens = seahorse.tokenize_text([prompt, prompt])
    preprocessed_imgs = seahorse.preprocess_image([random_pil(), random_pil()])
    assert tokens.input_ids.shape[0] == 2, "Tokenized prompts should have batch size of 2"
    assert preprocessed_imgs.shape[0] == 2, "Preprocessed images should have batch size of 2"

    out: CausalLMOutputWithPast = seahorse(
        input_ids=tokens.input_ids.to(seahorse.device),
        pixel_values=preprocessed_imgs.to(seahorse.device),
        attention_mask=tokens.attention_mask.to(seahorse.device),
    )
    assert isinstance(out, CausalLMOutputWithPast)
    assert out.logits.shape[0] == 2, "Batch size should be 2"


def test_seahorse_batched_inference_no_image(seahorse: SeahorseModel):
    prompt = "What is the answer?"
    tokens = seahorse.tokenize_text([prompt, prompt])
    assert tokens.input_ids.shape[0] == 2, "Tokenized prompts should have batch size of 2"

    out: CausalLMOutputWithPast = seahorse(
        input_ids=tokens.input_ids.to(seahorse.device),
        attention_mask=tokens.attention_mask.to(seahorse.device),
    )
    assert isinstance(out, CausalLMOutputWithPast)
    assert out.logits.shape[0] == 2, "Batch size should be 2"


def test_seahorse_inference(seahorse: SeahorseModel):
    prompt = DEFAULT_IMAGE_TOKEN + "\nIs there a seahorse?"
    tokens = seahorse.tokenize_text(prompt)
    preprocessed_img = seahorse.preprocess_image([random_pil()])
    out: CausalLMOutputWithPast = seahorse(
        input_ids=tokens.input_ids.to(seahorse.device),
        pixel_values=preprocessed_img.to(seahorse.device),
        attention_mask=tokens.attention_mask.to(seahorse.device),
    )
    assert isinstance(out, CausalLMOutputWithPast)
    assert out.logits.shape[0] == 1, "Batch size should be 1"


def test_seahorse_merge_text_and_image_tokens(seahorse: SeahorseModel):
    tokens = seahorse.tokenize_text("<|user|>\n" + DEFAULT_IMAGE_TOKEN + "\nIs there a seahorse?")
    projected_patch_embeddings = seahorse.encode_and_project_image(random_pil())
    merged_token_embeddings, merged_attention_mask, merged_labels = (
        seahorse.merge_text_and_image_tokens(
            tokens.input_ids.to(seahorse.device),
            image_patch_embeds=projected_patch_embeddings,
            attention_mask=tokens.attention_mask.to(seahorse.device),
            text_labels=None,
        )
    )
    assert merged_token_embeddings.ndim == 3  # (B, T, EMBED_DIM)
    assert merged_token_embeddings.shape[0] == 1
    assert merged_attention_mask.ndim == 2
    assert merged_attention_mask.shape[0] == 1
    assert merged_attention_mask.shape[1] == merged_token_embeddings.shape[1]
    assert merged_labels is None, "Labels should be None for inference (no text_labels provided)"

    batch_queries = [
        "<|user|>\n" + DEFAULT_IMAGE_TOKEN + "\nIs there a seahorse?",
        "<|user|>\n" + DEFAULT_IMAGE_TOKEN + "\nWhere is the seahorse in the image?",
    ]
    tokens = seahorse.tokenize_text(batch_queries)
    projected_patch_embeddings = seahorse.encode_and_project_image([random_pil(), random_pil()])
    merged_token_embeddings, merged_attention_mask, merged_labels = (
        seahorse.merge_text_and_image_tokens(
            tokens.input_ids.to(seahorse.device),
            projected_patch_embeddings,
            attention_mask=tokens.attention_mask.to(seahorse.device),
            text_labels=tokens.input_ids.clone().to(seahorse.device),
        )
    )
    assert merged_token_embeddings.ndim == 3  # (B, T, EMBED_DIM)
    assert merged_token_embeddings.shape[0] == 2
    assert merged_attention_mask.ndim == 2
    assert merged_attention_mask.shape[0] == 2
    assert merged_attention_mask.shape[1] == merged_token_embeddings.shape[1]
    assert merged_labels.shape[0] == 2
    assert torch.all(torch.any(merged_labels == IGNORE_INDEX, dim=1))


def test_seahorse_encode_and_project_image(seahorse: SeahorseModel):
    projected_patch_embeddings = seahorse.encode_and_project_image(random_pil())
    assert projected_patch_embeddings.ndim == 4
    assert projected_patch_embeddings.shape[0] == 1
    assert projected_patch_embeddings.shape[1] == projected_patch_embeddings.shape[2]


def test_seahorse_tokenizer(seahorse: SeahorseModel):
    tokens = seahorse.tokenize_text(DEFAULT_IMAGE_TOKEN + "\nIs there a seahorse?")
    input_ids = tokens.input_ids
    assert input_ids.ndim == 2
    assert input_ids.shape[0] == 1
    len_a = input_ids.shape[1]

    batch_queries = [
        DEFAULT_IMAGE_TOKEN + "\nIs there a seahorse?",
        DEFAULT_IMAGE_TOKEN + "\nWhere is the seahorse in the image?",  # more tokens
    ]
    tokens = seahorse.tokenize_text(batch_queries)
    input_ids = tokens.input_ids
    assert input_ids.ndim == 2
    assert input_ids.shape[0] == 2
    assert input_ids.shape[1] > len_a, "Second query has more tokens than the first one"
    assert (
        input_ids[0, 0].item() == seahorse.tokenizer.pad_token_id
    )  # First sequence should be left padded
    assert tokens.attention_mask[0][0] == 0  # First sequence should have padding mask
    assert input_ids[1, 0].item() != seahorse.tokenizer.pad_token_id  # Second sequence not padded
