from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SeahorseConfig(PretrainedConfig):
    language_model: str = "microsoft/Phi-3-mini-4k-instruct"
    lm_revision: str = "ff07dc01615f8113924aed013115ab2abd32115b"
    vision_encoder: str = "vit_base_patch16_clip_224.openai"
    image_positional_embedding: bool = False


class Seahorse128kConfig(SeahorseConfig):
    language_model: str = "microsoft/Phi-3-mini-128k-instruct"
    lm_revision: str = "bb5bf1e4001277a606e11debca0ef80323e5f824"
