import random

import pytest
from datasets.arrow_dataset import Dataset as HFDataset
from PIL.Image import Image as PILImage

from seahorse.data.data_prep.cv_bench import make_cv_bench
from seahorse.data.data_prep.llava_pretrain_cc3m import make_llava_pretrain_cc3m
from seahorse.data.data_prep.the_cauldron import make_the_cauldron
from seahorse.models.seahorse import SeahorseModel


@pytest.fixture(scope="module")
def seahorse():
    return SeahorseModel(config=None)


def test_make_the_cauldron(seahorse: SeahorseModel, capsys):
    random.seed(42)
    with capsys.disabled():
        ds: HFDataset = make_the_cauldron(seahorse, subset="vqav2")
    assert len(ds) == 443757
    assert set(ds.column_names) == set(["text", "image", "length"])
    assert (
        ds[0]["text"]
        == """<|system|>
Answer the question using a single word or phrase.<|end|>
<|user|>
<image>
What is this photo taken looking through?
Concise answer only.<|end|>
<|assistant|>
Net.<|end|>
<|endoftext|>"""  # NOTE: add_generation_prompt=False here, so <|endoftext|> is added
    ), ds[0]["text"]


def test_make_cv_bench(seahorse: SeahorseModel, capsys):
    with capsys.disabled():
        ds: HFDataset = make_cv_bench(seahorse, split="test")
    assert len(ds) == 2638
    assert set(ds.column_names) == set(
        ["image", "answer", "answer_value", "answer_idx", "source", "length", "text", "choices"]
    )
    assert (
        ds[0]["text"]
        == """<|system|>
Answer with the option’s letter from the given choices directly.<|end|>
<|user|>
<image>
How many organs are in the image? Select from the following choices.
(A) 3
(B) 2
(C) 1
(D) 0<|end|>
<|assistant|>
"""  # NOTE: add_generation_prompt=True here, because cv-bench is used for generation
    ), ds[0]["text"]
    assert ds[0]["answer_value"] == ["3", "2", "1", "0"][ds[0]["answer_idx"]]

    for ex in ds.select(range(100)):
        assert ex["answer_idx"] in set(range(6))
        assert ex["answer"] in set(["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"])


def test_make_llava_pretrain_cc3m(seahorse: SeahorseModel, capsys):
    with capsys.disabled():
        ds: HFDataset = make_llava_pretrain_cc3m(seahorse)
    assert len(ds) == 595375
    assert set(ds.column_names) == set(["text", "image", "length"])
    assert (
        ds[0]["text"]
        == """<|user|>
Provide a brief description of the given image.
<image><|end|>
<|assistant|>
olive oil is a healthy ingredient used liberally .<|end|>
<|endoftext|>"""  # NOTE: add_generation_prompt=False here, so <|endoftext|> is added
    ), ds[0]["text"]
    assert isinstance(ds[0]["image"], PILImage)
