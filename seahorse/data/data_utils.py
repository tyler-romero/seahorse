from pathlib import Path

import numpy as np
from PIL import Image

DATAOCEAN_PATH = Path("/home/tromero/workspace/dataocean")


def random_pil(height: int = 480, width: int = 640, random_seed=None, mode="RGB") -> Image.Image:
    """
    Generate a random pil image
    """
    if random_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(random_seed)
    np_image = rng.uniform(0, 255, (height, width, 3))
    pil_image = Image.fromarray(np.asarray(np_image, dtype=np.uint8))
    if mode == "L":
        pil_image = pil_image.convert("L")
    return pil_image
