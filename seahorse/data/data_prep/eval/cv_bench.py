from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HFDataset

from seahorse.data.data_utils import random_pil
from seahorse.models.seahorse import DEFAULT_IMAGE_TOKEN, SeahorseModel

# Taken from table 22 of the Cambrian-1 paper
SYSTEM_PROMPT = "Answer with the option’s letter from the given choices directly."
PROMPT_SUFFIX = "\nAnswer with the option’s letter from the given choices directly."
ANSWER_TO_IDX = {a: i for i, a in enumerate(["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"])}


def _ablate_image(example, how: str = "random"):
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


def _format_for_convo(example, tokenizer):
    messages = _make_messages(prompt=example["prompt"], image=example["image"])
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"text": text, "length": len(text)}


def _prep_answer_metadata(example):
    answer_idx = ANSWER_TO_IDX[example["answer"]]
    answer_value = example["choices"][answer_idx]
    return {"answer_idx": answer_idx, "answer_value": answer_value}


def make_cv_bench(
    model: SeahorseModel,
    split: str,  # NOTE: test is the only split... there is no validation split
    ablate_images: str | bool = False,
    num_proc: int = 16,
    load_from_cache_file: bool = False,
) -> HFDataset:
    """
    https://huggingface.co/datasets/nyu-visionx/CV-Bench

    Returns a dataset formatted for the CV-Bench evaluation task. The following columns are retained:
    - image: PIL.Image - the image
    - text: str - the formatted text for the conversation, ready for generation
    - length: int - the length of the text
    - source: str - the source of the image (e.g. "ADE20K")
    - answer: str - the correct answer (e.g. "(A)")
    - answer_idx: int - the index of the correct answer in the choices  (e.g. 0)
    - answer_value: str - the correct answer (e.g. "Door")
    """
    ds: HFDataset = load_dataset("nyu-visionx/CV-Bench", split=split)  # type: ignore
    ds = ds.select_columns(["image", "prompt", "answer", "source", "choices"])
    prefix = f"CV-Bench/{split}: "
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
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        remove_columns=["prompt"],
        desc=prefix + "Formatting for convo",
    )
    ds = ds.map(
        _prep_answer_metadata,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc=prefix + "Preparing answer metadata",
    )
    return ds
