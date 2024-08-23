import logging
from typing import Any

import torch

from seahorse.models.seahorse import SeahorseModel

logger = logging.getLogger(__name__)


def span_mask(tokens: torch.Tensor, start_token: int, end_token: int) -> torch.Tensor:
    """Returns a mask over tokens indicating the span between start_token and end_token in each example"""
    start_mask = (tokens == start_token).cumsum(dim=1)
    end_positions = tokens == end_token
    end_mask = end_positions.cumsum(dim=1)
    span_mask = (start_mask > end_mask) | end_positions
    return span_mask.bool()


class SeahorseDataCollator:
    def __init__(
        self,
        model: SeahorseModel,
        ift_mask: bool = False,
        user_token: int = -1,
        assistant_token: int = -1,
    ):
        self.model = model
        self.ift_mask = ift_mask
        self.user_token, self.assistant_token = user_token, assistant_token
        if self.ift_mask and (self.user_token == -1 or self.assistant_token == -1):
            raise ValueError("User and assistant tokens must be provided if ift_mask is True")
        if self.ift_mask and self.user_token == self.assistant_token:
            raise ValueError("User and assistant tokens must be different")
        if self.ift_mask:
            logger.warning(
                f"Using IFT user-prompt-masking with user token {self.user_token} and assistant token {self.assistant_token}"
            )

    def __call__(self, features: list[dict[str, Any]]) -> dict:
        """
        Create a batch of inputs for the model from a list of features.
        Will:
          1) Preprocess images
          2) Tokenize text and pad to the same length
          3) Set labels for language modeling, padding tokens are masked out
        """
        batch = {}
        images = [f["image"] for f in features]
        if images[0] is None:
            if not all(img is None for img in images):
                raise ValueError("All images in batch should be None if one image is None")
        else:
            batch["pixel_values"] = self.model.preprocess_image(images)
        tokens = self.model.tokenize_text([f["text"] for f in features])
        batch["input_ids"] = tokens.input_ids
        batch["attention_mask"] = tokens.attention_mask

        # Set labels for language modeling
        #   * padding tokens are masked out
        #   * ift_mask: user prompts are masked out
        loss_mask = tokens.input_ids == self.model.tokenizer.pad_token_id
        if self.ift_mask:
            user_mask = span_mask(tokens.input_ids, self.user_token, self.assistant_token)
            loss_mask |= user_mask

        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][loss_mask] = -100
        return batch
