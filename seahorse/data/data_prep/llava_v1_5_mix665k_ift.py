import pandas as pd
from datasets import Image as ImageFeature
from datasets.arrow_dataset import Dataset as HFDataset
from PIL.Image import Image as PILImage

from seahorse.data.data_utils import DATAOCEAN_PATH, random_pil
from seahorse.models.seahorse import DEFAULT_IMAGE_TOKEN, SeahorseModel

# Taken from table 22 of the Cambrian-1 paper
SYSTEM_PROMPT = "Answer with the option’s letter from the given choices directly."

# See instructions for downloading llava-v1.5-instruct data here:
# https://github.com/TRI-ML/prismatic-vlms/tree/main?tab=readme-ov-file#pretraining-datasets
LLAVA_INSTRUCT_PATH = DATAOCEAN_PATH / "prismatic/download/llava-v1.5-instruct/"


def _set_image_paths(image) -> dict[str, str]:
    if image:  # image can be None for some examples
        image = (LLAVA_INSTRUCT_PATH / image).as_posix()
    return {"image": image}


def _ablate_image(image, how: str = "random") -> dict[str, PILImage | None]:
    if how == "random":
        return {"image": random_pil()}
    elif how == "no_img":
        return {"image": None}
    else:
        raise ValueError(f"Unknown image ablation method: {how}")


def _make_messages(convo: list[dict], has_image: bool) -> list[dict[str, str]]:
    messages = {c["from"]: c["value"] for c in convo}
    turns = []
    # turns.append({"role": "system", "content": SYSTEM_PROMPT})
    img_token_count = 0
    for i, (turn, content) in enumerate(messages.items()):
        if has_image and i == 0 and DEFAULT_IMAGE_TOKEN not in content:
            # Manually insert image token at the beginning of the first message if
            # it's not already there
            content = f"{DEFAULT_IMAGE_TOKEN}\n{content}"
        role = "user" if turn == "human" else "assistant"
        turns.append({"role": role, "content": content})
        img_token_count += content.count(DEFAULT_IMAGE_TOKEN)
    if img_token_count != 1:
        raise ValueError(f"Expected 1 image token, found {img_token_count}. {turns=}")
    return turns


def _format_for_convo(has_image, conversations, tokenizer) -> dict[str, str | int]:
    messages = _make_messages(convo=conversations, has_image=has_image)
    # TODO: avoid taking loss for user messages!
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "length": len(text)}


def make_llava_v1_5_mix665k_ift(
    model: SeahorseModel,
    ablate_images: str | bool = False,  # TODO
    num_proc: int = 16,
    load_from_cache_file: bool = True,
    load_multimodal_only: bool = True,
) -> HFDataset:
    prefix = "LLaVA-1.5k-Mix665k-IFT: "

    # Load and process with pandas, b/c HFDataset.from_json cant handle
    # something about this file's json encoding. Pandas is more robust
    # to the issues with this file's formatting. Once loaded, dropping
    # "id" and "model" columns allows HFDataset.from_pandas to handle
    # the dataframe.
    df = pd.read_json(
        (LLAVA_INSTRUCT_PATH / "llava_v1_5_mix665k.json").as_posix(),
    ).drop(columns=["id", "model"])
    ds = HFDataset.from_pandas(df, preserve_index=False)

    df["has_image"] = df["image"].notnull()
    if load_multimodal_only:
        df = df.dropna(subset=["image"])
    else:
        raise NotImplementedError(
            "Text-only examples are not yet supported for llava-v1.5-instruct"
        )

    ds = HFDataset.from_pandas(df, preserve_index=False)

    ds = ds.map(
        _set_image_paths,
        input_columns=["image"],
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc=prefix + "Loading images",
    ).cast_column("image", ImageFeature())

    if ablate_images:
        ds = ds.map(
            _ablate_image,
            fn_kwargs={"how": ablate_images},
            input_columns=["image"],
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            desc=prefix + "Ablating images",
        )

    ds = ds.map(
        _format_for_convo,
        fn_kwargs={"tokenizer": model.tokenizer},
        # Avoid loading the "image" column, since that loads the
        # actual image data, which is very slow.
        input_columns=["has_image", "conversations"],
        remove_columns=["has_image", "conversations"],
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc=prefix + "Formatting for convo",
    )

    return ds
