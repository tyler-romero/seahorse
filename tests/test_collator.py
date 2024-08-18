import pytest

from seahorse.data.collator import SeahorseDataCollator
from seahorse.data.data_utils import random_pil
from seahorse.models.seahorse import DEFAULT_IMAGE_TOKEN, SeahorseModel


@pytest.fixture(scope="module")
def seahorse_data_collator(seahorse: SeahorseModel):
    return SeahorseDataCollator(seahorse)


def test_seahorse_data_collator(
    seahorse_data_collator: SeahorseDataCollator, seahorse: SeahorseModel
):
    # Setup
    message_1 = {
        "user": f"{DEFAULT_IMAGE_TOKEN}\nIs there a seahorse?",
        "assistant": "No, there is no seahorse.",
    }
    message_2 = {
        "user": f"{DEFAULT_IMAGE_TOKEN}\nIs there a seahorse?",
        "assistant": "Yes! There is a seahorse in the ocean.",
    }
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


def test_seahorse_data_collator_no_images(
    seahorse_data_collator: SeahorseDataCollator, seahorse: SeahorseModel
):
    # Setup
    message_1 = {
        "user": "Is there a seahorse?",
        "assistant": "No, there is no seahorse.",
    }
    message_2 = {
        "user": "Is there a seahorse?",
        "assistant": "Yes! There is a seahorse in the ocean.",
    }
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
