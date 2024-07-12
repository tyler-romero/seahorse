from typing import Literal

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SeahorseConfig(PretrainedConfig):
    language_model: str = "microsoft/Phi-3-mini-4k-instruct"
    lm_revision: str = "c1358f8a35e6d2af81890deffbbfa575b978c62f"
    vision_encoder: str = "vit_base_patch16_clip_224.openai"
    with_image_patch_positional_embeddings: Literal["xy", "each"] | None = None
    freeze_lm_input_output: bool = False


class Seahorse128kConfig(SeahorseConfig):
    language_model: str = "microsoft/Phi-3-mini-128k-instruct"
    lm_revision: str = "d548c233192db00165d842bf8edff054bb3212f8"
