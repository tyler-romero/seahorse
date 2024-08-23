import pytest
import torch

from seahorse.data.collator import SeahorseDataCollator, span_mask
from seahorse.data.data_utils import random_pil
from seahorse.models.seahorse import DEFAULT_IMAGE_TOKEN, SeahorseModel


@pytest.fixture(scope="module")
def seahorse_data_collator(seahorse: SeahorseModel):
    return SeahorseDataCollator(seahorse)


def test_seahorse_data_collator(seahorse: SeahorseModel):
    # Setup
    seahorse_data_collator = SeahorseDataCollator(seahorse)

    message_1 = [
        {"role": "user", "content": f"{DEFAULT_IMAGE_TOKEN}\nIs there a seahorse?"},
        {"role": "assistant", "content": "No, there is no seahorse."},
    ]
    message_2 = [
        {"role": "user", "content": f"{DEFAULT_IMAGE_TOKEN}\nIs there a seahorse?"},
        {"role": "assistant", "content": "Yes! There is a seahorse in the ocean."},
    ]
    text = seahorse.tokenizer.apply_chat_template(
        [message_1, message_2], tokenize=False, add_generation_prompt=False
    )

    features = [
        {"image": random_pil(), "text": text[0]},
        {"image": random_pil(), "text": text[1]},
    ]

    # Call the collator
    batch = seahorse_data_collator(features)

    # Check if all expected keys are in the batch
    assert set(batch.keys()) == {"pixel_values", "input_ids", "attention_mask", "labels"}

    # Check pixel_values
    assert batch["pixel_values"].shape[0] == 2
    assert batch["pixel_values"].shape[1] == 3  # RGB channels

    # Check input_ids, attention_mask, and labels
    assert len(batch["input_ids"]) == 2
    assert len(batch["attention_mask"]) == 2
    assert len(batch["labels"]) == 2

    # Check that padding is applied correctly
    max_length = max(len(ids) for ids in batch["input_ids"])
    for i in range(2):
        assert len(batch["input_ids"][i]) == max_length
        assert len(batch["attention_mask"][i]) == max_length
        assert len(batch["labels"][i]) == max_length

        # Check left-padding
        non_pad_length = sum(batch["attention_mask"][i])
        assert all(
            id == seahorse.tokenizer.pad_token_id
            for id in batch["input_ids"][i][: max_length - non_pad_length]
        )
        assert all(mask == 0 for mask in batch["attention_mask"][i][: max_length - non_pad_length])
        assert all(label == -100 for label in batch["labels"][i][: max_length - non_pad_length])

        # Check non-padded content to the right
        assert all(
            id != seahorse.tokenizer.pad_token_id
            for id in batch["input_ids"][i][max_length - non_pad_length :]
        )
        assert all(mask == 1 for mask in batch["attention_mask"][i][max_length - non_pad_length :])
        assert all(label != -100 for label in batch["labels"][i][max_length - non_pad_length :])

    # Check that labels are set correctly
    pad_token_id = seahorse.tokenizer.pad_token_id
    for i in range(2):
        assert all(
            label == -100 if input_id == pad_token_id else label == input_id
            for label, input_id in zip(batch["labels"][i], batch["input_ids"][i])
        )


def test_seahorse_data_collator_no_images(seahorse: SeahorseModel):
    # Setup
    seahorse_data_collator = SeahorseDataCollator(seahorse)

    message_1 = [
        {"role": "user", "content": f"{DEFAULT_IMAGE_TOKEN}\nIs there a seahorse?"},
        {"role": "assistant", "content": "No, there is no seahorse."},
    ]
    message_2 = [
        {"role": "user", "content": f"{DEFAULT_IMAGE_TOKEN}\nIs there a seahorse?"},
        {"role": "assistant", "content": "Yes! There is a seahorse in the ocean."},
    ]
    text = seahorse.tokenizer.apply_chat_template(
        [message_1, message_2], tokenize=False, add_generation_prompt=False
    )

    features = [{"image": None, "text": text[0]}, {"image": None, "text": text[1]}]

    # Call the collator
    batch = seahorse_data_collator(features)

    # Check if all expected keys are in the batch
    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}

    # Check input_ids, attention_mask, and labels
    assert len(batch["input_ids"]) == 2
    assert len(batch["attention_mask"]) == 2
    assert len(batch["labels"]) == 2

    # Check that padding is applied correctly
    max_length = max(len(ids) for ids in batch["input_ids"])
    for i in range(2):
        assert len(batch["input_ids"][i]) == max_length
        assert len(batch["attention_mask"][i]) == max_length
        assert len(batch["labels"][i]) == max_length

    # Check that labels are set correctly
    pad_token_id = seahorse.tokenizer.pad_token_id
    for i in range(2):
        assert all(
            label == -100 if input_id == pad_token_id else label == input_id
            for label, input_id in zip(batch["labels"][i], batch["input_ids"][i])
        )


def test_seahorse_data_collator_ift_masking(seahorse: SeahorseModel):
    # Setup
    user_token: int = seahorse.tokenizer.convert_tokens_to_ids("<|user|>")
    assistant_token: int = seahorse.tokenizer.convert_tokens_to_ids("<|assistant|>")
    assert user_token == 32010  # This test assumes Phi3's tokenizer

    seahorse_data_collator = SeahorseDataCollator(
        seahorse, ift_mask=True, user_token=user_token, assistant_token=assistant_token
    )

    message_1 = [
        {"role": "user", "content": f"{DEFAULT_IMAGE_TOKEN}\nIs there a seahorse?"},
        {"role": "assistant", "content": "No, there is no seahorse."},
    ]
    message_2 = [
        {"role": "user", "content": f"{DEFAULT_IMAGE_TOKEN}\nIs there a seahorse?"},
        {"role": "assistant", "content": "Yes! There is a seahorse in the ocean."},
        {"role": "user", "content": "What color is it?"},
    ]

    text = seahorse.tokenizer.apply_chat_template(
        [message_1, message_2], tokenize=False, add_generation_prompt=False
    )

    features = [
        {"image": random_pil(), "text": text[0]},
        {"image": random_pil(), "text": text[1]},
    ]

    # Call the collator
    batch = seahorse_data_collator(features)

    # Check if all expected keys are in the batch
    assert set(batch.keys()) == {"pixel_values", "input_ids", "attention_mask", "labels"}

    # Check that padding is applied correctly
    max_length = max(len(ids) for ids in batch["input_ids"])
    for i in range(2):
        assert len(batch["input_ids"][i]) == max_length
        assert len(batch["attention_mask"][i]) == max_length
        assert len(batch["labels"][i]) == max_length

        # Check left-padding
        non_pad_length = sum(batch["attention_mask"][i])
        assert all(
            id == seahorse.tokenizer.pad_token_id
            for id in batch["input_ids"][i][: max_length - non_pad_length]
        )
        assert all(mask == 0 for mask in batch["attention_mask"][i][: max_length - non_pad_length])
        assert all(label == -100 for label in batch["labels"][i][: max_length - non_pad_length])

        # Check non-padded content to the right
        assert all(
            id != seahorse.tokenizer.pad_token_id
            for id in batch["input_ids"][i][max_length - non_pad_length :]
        )
        assert all(mask == 1 for mask in batch["attention_mask"][i][max_length - non_pad_length :])

        # Assert that there are a mix of masked and unmasked tokens outside of the pad length
        # TODO: strengthen this test
        assert any(label != -100 for label in batch["labels"][i][max_length - non_pad_length :])
        assert any(label == -100 for label in batch["labels"][i][max_length - non_pad_length :])

    # Assert that the last token in the second sequence is not padded but is label-masked
    # because it is part of a user prompt
    assert batch["input_ids"][1][-1] != seahorse.tokenizer.pad_token_id
    assert batch["labels"][1][-1] == -100


def test_span_mask():
    tokens = torch.Tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [2, 6, 4, 5, 2, 3, 6, 9],
        ]
    )

    mask = span_mask(tokens, start_token=2, end_token=6)

    expected_mask = torch.Tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 1, 0],
        ]
    )

    assert torch.all(mask.bool() == expected_mask.bool())
