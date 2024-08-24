import random
import warnings

import torch
from transformers.utils import logging as transformers_logging


def two_word_name() -> str:
    # fmt: off
    adjectives = [
        'happy', 'sunny', 'clever', 'brave', 'gentle', 'kind', 'swift', 'bright',
        'wise', 'calm', 'eager', 'fierce', 'jolly', 'lively', 'merry', 'proud',
        'quiet', 'witty', 'zealous', 'daring', 'elegant', 'friendly', 'graceful',
        'vibrant', 'serene', 'bold', 'charming', 'dazzling', 'enchanting', 'fantastic',
        'glorious', 'harmonious', 'innovative', 'joyful', 'keen', 'luminous', 'majestic',
        'noble', 'optimistic', 'peaceful', 'radiant', 'splendid', 'tranquil', 'upbeat',
        'vivacious', 'whimsical', 'exuberant', 'youthful', 'zestful', 'adventurous',
        'blissful', 'cosmic', 'dynamic', 'ethereal', 'flourishing', 'gallant'
    ]
    nouns = [
        'panda', 'river', 'mountain', 'forest', 'ocean', 'star', 'cloud', 'tiger',
        'falcon', 'meadow', 'canyon', 'island', 'breeze', 'phoenix', 'dolphin',
        'eagle', 'garden', 'horizon', 'lagoon', 'nebula', 'oasis', 'quasar',
        'aurora', 'blossom', 'cascade', 'dune', 'ember', 'fjord', 'galaxy',
        'harbor', 'iceberg', 'jungle', 'kaleidoscope', 'lighthouse', 'mirage',
        'nova', 'orchid', 'plateau', 'reef', 'savanna', 'tempest', 'universe',
        'volcano', 'waterfall', 'zenith', 'archipelago', 'beacon', 'citadel',
        'delta', 'eclipse', 'fountain', 'geyser', 'haven', 'isthmus'
    ]
    # fmt: on
    return f"{random.choice(adjectives)}-{random.choice(nouns)}"


def randstr(length=8):
    return "".join(random.choices("0123456789", k=length))


def get_gpu_memory_usage_gb():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    else:
        return 0


def print_gpu_memory_usage():
    print(f"GPU memory usage: {get_gpu_memory_usage_gb():.2f}GB")


def enable_transformers_logging():
    transformers_logging.enable_default_handler()
    transformers_logging.set_verbosity_info()


def mute_warnings():
    warnings.filterwarnings(
        "ignore",
        ".*torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*",
    )
