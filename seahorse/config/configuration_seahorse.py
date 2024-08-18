from typing import Literal

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SeahorseConfig(PretrainedConfig):
    with_image_patch_positional_embeddings: Literal["xy", "each"] | None = None
    freeze_llm_input: bool = False
    freeze_llm_output: bool = False
