from datasets.arrow_dataset import Dataset as HFDataset
from torch.utils.data import Dataset

from seahorse.data.utils import random_pil


def make_instruction_text(text: dict, image) -> str:
    # Simple instruction text based off of Phi3's format
    image_token = "<image> " if image is not None else ""
    return f"<|user|>\n{image_token}{text['user']}\n<|assistant|>\n{text['assistant']}"


def add_lengths(batch: dict[str, list]) -> dict:
    """
    Since currently all examples have images, we can ignore the image for the length calculation.
    It would be better to calcualte length based on tokens, but that makes collation slightly harder.
    """
    lengths = [len(make_instruction_text(texts[0], image=True)) for texts in batch["texts"]]
    batch["length"] = lengths
    return batch


class SeahorseDataset(Dataset):
    def __init__(self, dataset: HFDataset, ablate_images: bool | str = False):
        self.dataset = dataset.map(add_lengths, batched=True, num_proc=8)
        self.ablate_images = ablate_images

    def __len__(self):
        return len(self.dataset)

    @property
    def lengths(self) -> list[int]:
        return self.dataset["length"]

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError("Index must be an integer")
        item = self.dataset[idx]
        if not item["images"]:
            print(f"Skipping example {idx} as it does not contain any images")
            return None
        if len(item["images"]) > 1:
            raise ValueError("Only one image per example is supported")

        text = item["texts"][0]  # TODO: Handle multiple texts
        if self.ablate_images == "random":
            image = random_pil()
        elif self.ablate_images == "no_img":
            image = None
        else:
            image = item["images"][0]

        output = {"image": image, "text": make_instruction_text(text, image=image)}
        return output
