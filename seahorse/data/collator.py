from seahorse.models.seahorse import SeahorseModel


class SeahorseDataCollator:
    def __init__(self, model: SeahorseModel):
        self.model = model

    def __call__(self, features: dict) -> dict:
        """
        Create a batch of inputs for the model from a list of features.
        Will:
          1) Preprocess images
          2) Tokenize text and pad to the same length
          3) Set labels for language modeling, padding tokens are masked out
        """
        batch = {}
        images = [f["image"] for f in features]
        if images[0] is None:
            if not all(img is None for img in images):
                raise ValueError("All images in batch should be None if one image is None")
        else:
            batch["pixel_values"] = self.model.preprocess_image(images)
        tokens = self.model.tokenize_text([f["text"] for f in features])
        batch["input_ids"] = tokens.input_ids
        batch["attention_mask"] = tokens.attention_mask

        # Set labels for language modeling, padding tokens are masked out
        padding_mask = tokens.input_ids == self.model.tokenizer.pad_token_id
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][padding_mask] = -100
        return batch
