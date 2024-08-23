from typing import Literal

from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HFDataset

from seahorse.data.data_utils import random_pil
from seahorse.models.seahorse import DEFAULT_IMAGE_TOKEN, SeahorseModel

# As per llava: https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#evaluate-on-custom-datasets
PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."


def _ablate_image(example: dict, how: str = "random"):
    if how == "random":
        return {"image": random_pil()}
    elif how == "no_img":
        return {"image": None}
    else:
        raise ValueError(f"Unknown image ablation method: {how}")


def _make_messages(prompt: str, image: bool) -> list[dict[str, str]]:
    if image:
        # TODO: evaluate choice to put image at front of message
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    return [
        # {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt + PROMPT_SUFFIX},
    ]


def _format_for_convo(example: dict, tokenizer) -> dict[str, str | int]:
    messages = _make_messages(prompt=example["question"], image=example["image"])
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"text": text, "length": len(text)}


def make_pope(
    model: SeahorseModel,
    split: Literal["popular", "adversarial", "random"],
    ablate_images: str | bool = False,
    num_proc: int = 16,
    load_from_cache_file: bool = False,
) -> HFDataset:
    """
    POPE: Polling-based Object Probing Evaluation for Object Hallucination
    https://huggingface.co/datasets/lmms-lab/POPE

    Returns a dataset formatted for the CV-Bench evaluation task. The following columns are retained:
    - image: PIL.Image - the image
    - text: str - the formatted text for the conversation, ready for generation
    - length: int - the length of the text
    - answer: str - the correct answer (e.g. "yes" or "no")
    - category: str - the split of the data (e.g. "popular", "adversarial", "random")
    """
    ds: HFDataset = load_dataset("lmms-lab/POPE", name="Full", split=split, num_proc=num_proc)

    prefix = "Pope: "
    if ablate_images:
        ds = ds.map(
            _ablate_image,
            fn_kwargs={"how": ablate_images},
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            desc=prefix + "Ablating images",
        )

    ds = ds.map(
        _format_for_convo,
        fn_kwargs={"tokenizer": model.tokenizer},
        remove_columns=["question"],
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc=prefix + "Formatting for convo",
    )
    return ds
