import random

from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HFDataset

from seahorse.data.data_utils import random_pil
from seahorse.models.seahorse import DEFAULT_IMAGE_TOKEN, SeahorseModel

SUBSET_TO_SYSTEM_PROMPTS = {
    "vqav2": [
        "Answer the question using a single word or phrase.",  # cambrian-1
        "Give a short answer directly.",  # cambrian-1
        # "Use the provided image to answer the question.",  # custom
    ]
}


def _validate_and_extract(examples) -> dict[str, list]:
    batch_texts = []
    batch_images = []
    for images, texts in zip(examples["images"], examples["texts"]):
        if not images:
            raise ValueError("No image found in example from the_cauldron dataset")
        if len(images) > 1:
            raise ValueError("Only one image per example is supported")
        image = images[0]
        batch_texts += texts
        batch_images += [image] * len(texts)
    return {"text": batch_texts, "image": batch_images, "has_image": [True] * len(batch_texts)}


def _ablate_image(example, how: str = "random"):
    if how == "random":
        return {"image": random_pil(), "has_image": True}
    elif how == "no_img":
        return {"image": None, "has_image": False}
    else:
        raise ValueError(f"Unknown image ablation method: {how}")


def _reformat_messages(messages: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": messages["system"]},
        {"role": "user", "content": messages["user"]},
        {"role": "assistant", "content": messages["assistant"]},
    ]


def _format_for_convo(example, tokenizer, system_prompts: list[str]):
    # cauldron data is already in message format {"user": ..., "assistant": ...}
    messages = example["text"]
    if example["has_image"]:
        # TODO: evaluate choice to put image at front of message
        messages["user"] = f"{DEFAULT_IMAGE_TOKEN}\n{messages['user']}"
    messages["system"] = random.choice(system_prompts)  # potential randomness between runs
    messages = _reformat_messages(messages)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "length": len(text)}


def make_the_cauldron(
    model: SeahorseModel,
    subset: str,
    split: str = "train",
    ablate_images: str | bool = False,
    num_proc: int = 16,
    load_from_cache_file: bool = False,
) -> HFDataset:
    """
    The Cauldron is a massive collection of 50 vision-language datasets (training sets only)
    that were used for the fine-tuning of the vision-language model Idefics2.
    https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
    """
    system_prompts: list[str] = SUBSET_TO_SYSTEM_PROMPTS[subset]
    ds: HFDataset = load_dataset("HuggingFaceM4/the_cauldron", subset, split=split)  # type: ignore

    prefix = f"TheCauldron/{subset}/{split}: "
    ds = ds.map(
        _validate_and_extract,
        remove_columns=["texts", "images"],
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc=prefix + "Extracting text and images",
    )
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
        fn_kwargs={"tokenizer": model.tokenizer, "system_prompts": system_prompts},
        num_proc=num_proc - 1,
        load_from_cache_file=load_from_cache_file,
        desc=prefix + "Formatting for convo",
    )
    ds = ds.remove_columns(["has_image"])

    return ds
