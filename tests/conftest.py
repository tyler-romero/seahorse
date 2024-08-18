import pytest

from seahorse.models.construction import construct_seahorse


@pytest.fixture(scope="session")
def seahorse():
    """
    Returns a SeahorseModel instance, with the model loaded on the appropriate
    device and dtype. This model is cached and reused across all tests.
    """
    return construct_seahorse()
