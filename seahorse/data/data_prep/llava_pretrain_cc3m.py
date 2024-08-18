from datasets import Image as ImageFeature
from datasets.arrow_dataset import Dataset as HFDataset

from seahorse.data.data_utils import DATAOCEAN_PATH, random_pil
from seahorse.models.seahorse import SeahorseModel

# Taken from table 22 of the Cambrian-1 paper
SYSTEM_PROMPT = "Answer with the option’s letter from the given choices directly."
LLAVA_PRETRAIN_PATH = DATAOCEAN_PATH / "llava/LLaVA-CC3M-Pretrain-595K/"


def _set_image_paths(example):
    example["image"] = (LLAVA_PRETRAIN_PATH / "images" / example["image"]).as_posix()
    return example


def _ablate_image(example, how: str = "random"):
    if how == "random":
        return {"image": random_pil()}
    elif how == "no_img":
        return {"image": None}
    else:
        raise ValueError(f"Unknown image ablation method: {how}")


def _make_messages(convo: list[dict]) -> list[dict[str, str]]:
    messages = {c["from"]: c["value"] for c in convo}
    turns = []
    # turns.append({"role": "system", "content": SYSTEM_PROMPT})
    for turn, content in messages.items():
        role = "user" if turn == "human" else "assistant"
        turns.append({"role": role, "content": content})
    return turns


# NOTE: llava pretrain prompts already have <image> inserted
def _format_for_convo(example, tokenizer):
    messages = _make_messages(convo=example["conversations"])
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "length": len(text)}


def make_llava_pretrain_cc3m(
    model: SeahorseModel,
    ablate_images: str | bool = False,
    num_proc: int = 16,
    load_from_cache_file: bool = False,
) -> HFDataset:
    """
    https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K

    Expected dir format:
    llava/LLaVA-CC3M-Pretrain-595K/
    ├── chat.json
    ├── images/
    """
    ds: HFDataset = HFDataset.from_json((LLAVA_PRETRAIN_PATH / "chat.json").as_posix())

    prefix = "LLaVA-CC3M-Pretrain-595K: "
    ds = ds.map(
        _set_image_paths,
        remove_columns=["id"],
        num_proc=8,
        desc=prefix + "Loading images",
    ).cast_column("image", ImageFeature())

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
        remove_columns=["conversations"],
        desc=prefix + "Formatting for convo",
    )
    return ds
