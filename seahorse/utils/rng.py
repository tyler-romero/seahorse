import os
import random
from contextlib import contextmanager
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Dict, Generator

import numpy as np
import torch

# This module contains utilities borrowed from PyTorch Lightning:
# https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/utilities/seed.py
# We only want these simple utilities and not the entire PyTorch Lightning library, so we copied them here.


def _collect_rng_states(include_cuda: bool = True) -> Dict[str, Any]:
    r"""Collect the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python."""
    states = {
        "torch": torch.get_rng_state(),
        "python": python_get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if include_cuda:
        states["torch.cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return states


def _set_rng_states(rng_state_dict: Dict[str, Any]) -> None:
    r"""Set the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python in the current
    process."""
    torch.set_rng_state(rng_state_dict["torch"])
    if "torch.cuda" in rng_state_dict:
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])
    np.random.set_state(rng_state_dict["numpy"])
    version, state, gauss = rng_state_dict["python"]
    python_set_rng_state((version, tuple(state), gauss))


@contextmanager
def isolate_rng(include_cuda: bool = True) -> Generator[None, None, None]:
    """A context manager that resets the global random state on exit to what it was before entering.

    It supports isolating the states for PyTorch, Numpy, and Python built-in random number generators.

    Example:
        >>> torch.manual_seed(1)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> with isolate_rng():
        ...     [torch.rand(1) for _ in range(3)]
        [tensor([0.7576]), tensor([0.2793]), tensor([0.4031])]
        >>> torch.rand(1)
        tensor([0.7576])
    """
    states = _collect_rng_states()
    yield
    _set_rng_states(states)
