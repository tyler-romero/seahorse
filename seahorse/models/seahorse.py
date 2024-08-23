import logging

import torch
from PIL.Image import Image as PILImage
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from seahorse.config.configuration_seahorse import SeahorseConfig
from seahorse.models.vision_encoder import LlavaVisionProjector, TimmEncoder

logger = logging.getLogger(__name__)

# NOTE: Figure these out a bit better, see existing placeholder tokens
LABEL_IGNORE_INDEX = -100
DEFAULT_PADDING_TOKEN = "<pad>"
DEFAULT_IMAGE_TOKEN = "<image>"
# https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/added_tokens.json


class ImagePositionalEmbeddingForEach(nn.Module):
    """Every image patch has its own positional embedding."""

    def __init__(self, num_patches_x, num_patches_y, embedding_dim) -> None:
        super().__init__()
        self.image_patch_pos = nn.Parameter(
            torch.zeros(1, num_patches_x, num_patches_y, embedding_dim)
        )
        nn.init.normal_(self.image_patch_pos, std=0.02)

    def forward(self, image_patch_embeddings: torch.Tensor) -> torch.Tensor:
        return image_patch_embeddings + self.image_patch_pos


class ImagePositionalEmbeddingXY(nn.Module):
    """Positional embeddings for image patches in X and Y directions."""

    def __init__(self, num_patches_x, num_patches_y, embedding_dim) -> None:
        super().__init__()
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.image_patch_pos_x = nn.Parameter(torch.zeros(1, num_patches_x, 1, embedding_dim))
        self.image_patch_pos_y = nn.Parameter(torch.zeros(1, 1, num_patches_y, embedding_dim))
        nn.init.normal_(self.image_patch_pos_x, std=0.02)
        nn.init.normal_(self.image_patch_pos_y, std=0.02)

    def forward(self, image_patch_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = image_patch_embeddings.shape[0]
        pos_x = self.image_patch_pos_x.expand(batch_size, -1, self.num_patches_y, -1)
        pos_y = self.image_patch_pos_y.expand(batch_size, self.num_patches_x, -1, -1)
        return image_patch_embeddings + pos_x + pos_y


class SeahorseModel(PreTrainedModel):
    config_class = SeahorseConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Phi3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def __init__(
        self,
        vision_encoder: TimmEncoder,
        language_model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        config: SeahorseConfig | None = None,
    ):
        self.config = config or SeahorseConfig()
        super().__init__(self.config)

        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.tokenizer: PreTrainedTokenizer = self.setup_tokenizer_for_vision(tokenizer)

        embedding_dim = self.language_model.get_input_embeddings().embedding_dim  # type: ignore
        self.vision_projector = LlavaVisionProjector(
            mm_hidden_size=self.vision_encoder.timm_model.num_features, hidden_size=embedding_dim
        )

        self.img_pos_embed = None
        if self.config.with_image_patch_positional_embeddings == "xy":
            self.img_pos_embed = ImagePositionalEmbeddingXY(
                num_patches_x=self.vision_encoder.num_patches_x,
                num_patches_y=self.vision_encoder.num_patches_y,
                embedding_dim=embedding_dim,
            )
        elif self.config.with_image_patch_positional_embeddings == "each":
            self.img_pos_embed = ImagePositionalEmbeddingForEach(
                num_patches_x=self.vision_encoder.num_patches_x,
                num_patches_y=self.vision_encoder.num_patches_y,
                embedding_dim=embedding_dim,
            )

        self.generation_config = GenerationConfig(
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=[  # # phi3 specific generation eos tokens
                self.tokenizer.convert_tokens_to_ids("<|assistant|>"),
                self.tokenizer.convert_tokens_to_ids("<|end|>"),
                self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            ],
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=64,
        )

        self._specify_trainable_params()

    def _specify_trainable_params(self):
        # Freeze the vision encoder
        logger.info("[Frozen 🥶] vision encoder")
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Freeze the language model...
        logger.info("[Frozen 🥶] language model")
        for name, param in self.language_model.named_parameters():
            param.requires_grad = False

        # ...except for the embedding table and the lm_head
        # Note: without lm_head requiring a grad, torch.compile doesn't work
        if not self.config.freeze_llm_input:
            for name, param in self.language_model.named_parameters():
                if "embed_tokens" in name:
                    param.requires_grad = True
            self.language_model.get_input_embeddings().requires_grad = True  # type: ignore

        if not self.config.freeze_llm_output:
            for name, param in self.language_model.named_parameters():
                if "lm_head" in name:
                    param.requires_grad = True
            self.language_model.get_output_embeddings().requires_grad = True  # type: ignore

        # Make sure the vision projector is trainable
        logger.info("[Trainable 🔥] projector")
        for param in self.vision_projector.parameters():
            param.requires_grad = True

        # And the image positional embeddings
        logger.info("[Trainable 🔥] image positional embeddings")
        if self.img_pos_embed is not None:
            for param in self.img_pos_embed.parameters():
                param.requires_grad = True

    def setup_tokenizer_for_vision(self, tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
        self.tokenizer = tokenizer
        # use unk rather than eos token to prevent endless generation
        # see https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/sample_finetune.py
        # overwrite the default pad token to prevent endless generation
        self.tokenizer.add_tokens([DEFAULT_PADDING_TOKEN], special_tokens=True)
        self.tokenizer.pad_token = DEFAULT_PADDING_TOKEN
        print(f"Pad token ID: {tokenizer.pad_token_id}")
        self.tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        print(f"Image token ID: {self.image_token_id}")
        self.eos_token_id = self.tokenizer.eos_token_id
        print(f"EOS token ID: {self.eos_token_id}")
        return self.tokenizer

    def tokenize_text(
        self, text: str | list[str], pad_to_multiple_of: int | None = None
    ) -> "BatchEncoding":
        tokens = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        images_per_ex = (tokens.input_ids == self.image_token_id).sum(dim=-1)
        if (images_per_ex > 1).any():
            raise ValueError("Multiple image tokens found in a single example")
        return tokens

    def apply_chat_template(
        self,
        messages: list[dict[str, str]] | list[list[dict[str, str]]],
        tokenize: bool = True,
        pad_to_multiple_of: int | None = None,
    ) -> "BatchEncoding":
        return self.tokenizer.apply_chat_template(  # type: ignore
            messages,
            tokenize=tokenize,
            add_generation_prompt=False,  # doesnt do anything for Phi3
            padding=True,  # pads to longest length
            truncation=True,
            max_length=self.tokenizer.model_max_length,  # TODO: truncation w/ image tokens
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.lm_head

    def preprocess_image(
        self, image: PILImage | list[PILImage] | torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.vision_encoder.preprocess(image)

    def encode_image(
        self, image: PILImage | list[PILImage] | torch.FloatTensor
    ) -> torch.FloatTensor:
        # taken from moondream
        with torch.no_grad():
            return self.vision_encoder(image)

    def encode_and_project_image(
        self, image: PILImage | list[PILImage] | torch.FloatTensor
    ) -> torch.FloatTensor:
        patch_embeddings = self.encode_image(image)  # (BS, T_x, T_y, EMBED_DIM)

        # Apply vision_projector patch-wise
        B, T_x, T_y, C = patch_embeddings.shape
        patch_embeddings = patch_embeddings.view(B, -1, C)  # Flatten
        projected_embeddings = self.vision_projector(patch_embeddings)
        projected_embeddings = projected_embeddings.view(B, T_x, T_y, -1)  # Restore
        return projected_embeddings

    def merge_text_and_image_tokens(
        self,
        input_ids: torch.LongTensor,
        text_embeds: torch.Tensor,
        image_patch_embeds: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        text_labels: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.LongTensor | None, torch.LongTensor | None, torch.LongTensor | None
    ]:
        """
        Merge text and image tokens into a single tensor for input to the language model.
        The image token in each sequence is replaced with the image patch embeddings.
        Only one image token is allowed per sequence (currently). Alternatively, batches
        with no image tokens are also supported. But mixed batches are not allowed.

        Args:
            input_ids (torch.LongTensor): Tokenized text input (BS, SQ_LEN)
            text_embeds (torch.Tensor): Text embeddings (BS, SQ_LEN, EMBED_DIM)
            text_labels (torch.LongTensor): Text token labels, if any (BS, SQ_LEN)
            image_patch_embeds (torch.Tensor | None): Image patch embeddings, if any (BS, T_x, T_y, EMBED_DIM)

        Returns:
            torch.Tensor: Merged embeddings for input to the language model (BS, SQ_LEN, EMBED_DIM)
            torch.LongTensor | None: Attention mask for the merged embeddings, if any (BS, SQ_LEN)
            torch.LongTensor | None: Merged labels for the language model, if any (BS, SQ_LEN)
            torch.LongTensor | None: Merged position_ids for the language model, if any (BS, SQ_LEN)
        """
        # Inspired by moondream
        bs, sl = input_ids.shape
        image_mask_orig_seq = input_ids == self.image_token_id

        if image_mask_orig_seq.sum() == 0:  # no image tokens in any sequence
            if image_patch_embeds is not None:
                logger.warning("Images were provided, but no image tokens were found.")
            # TODO: pad_to_multiple_of for the easy case
            return text_embeds, attention_mask, text_labels, position_ids
        elif (image_mask_orig_seq.sum(dim=-1) != 1).any():
            raise ValueError(
                f"Multiple image tokens found in a single example. {image_mask_orig_seq=}"
            )
        elif image_patch_embeds is None:
            raise ValueError("Image tokens found but no images provided")
        elif image_patch_embeds.shape[0] != bs:
            raise ValueError(
                f"Number of images provided does not match batch size: {bs=} != {image_patch_embeds.shape[0]}"
            )
        elif position_ids is not None:
            raise ValueError("position_ids is not (yet) supported by SeahorseModel")

        _, tile_x, tile_y, chan = image_patch_embeds.shape
        img_seq_len = tile_x * tile_y
        new_seq_len = sl + img_seq_len - 1  # remove the image token

        image_idx_orig_seq = image_mask_orig_seq.nonzero(as_tuple=True)[1]  # (BS,)
        image_patch_embeds = image_patch_embeds.view(bs, -1, chan)

        # Create indices for text embeddings, excluding the image token
        text_indices_orig_seq = torch.arange(sl, device=self.device).unsqueeze(0).expand(bs, -1)
        text_mask_orig_seq = input_ids != self.image_token_id  # TODO
        text_indices_orig_seq = text_indices_orig_seq[text_mask_orig_seq].view(bs, -1)

        text_embeds = text_embeds[text_mask_orig_seq]  # drop image token "embeddings"

        # Calculate new positions for text tokens
        text_indices_new_seq = text_indices_orig_seq + (  # (BS, SL)
            text_indices_orig_seq >= image_idx_orig_seq.unsqueeze(1)
        ) * (img_seq_len - 1)  # -1 to account for the removed image token

        # Create indices for image patch embeddings
        image_indices_new_seq = image_idx_orig_seq.unsqueeze(1) + torch.arange(  # (BS, ISL)
            img_seq_len, device=self.device
        ).unsqueeze(0)

        # Fill in the placeholder tensor with text and image embeddings
        output_embeds = torch.zeros(
            (bs, new_seq_len, chan), device=self.device, dtype=self.language_model.dtype
        )
        # NOTE: text_indices_new_seq and image_indices_new_seq should have no overlap in the same example
        #       and they should completely span the range [0, new_seq_len)
        output_embeds.scatter_(  # (BS, NSL, C)
            1,
            text_indices_new_seq.unsqueeze(-1).expand(-1, -1, chan),  # (BS, SL, C) 0 <= x < NSL
            text_embeds.view(bs, -1, chan),  # (BS, SL, C)
        )
        output_embeds.scatter_(
            1,
            image_indices_new_seq.unsqueeze(-1).expand(-1, -1, chan),  # (BS, ISL, C) 0 <= x < NSL
            image_patch_embeds,  # (BS, ISL, C)
        )

        attention_mask_new_seq = None
        if attention_mask is not None:
            # Extend attention_mask with 1s on the right
            is_padding_right = attention_mask[:, sl - 1].sum().item() != bs
            if is_padding_right:
                raise ValueError("Padding must be on the left side of the sequence")

            attention_mask_new_seq = torch.ones(  # type: ignore
                (bs, new_seq_len), device=self.device, dtype=attention_mask.dtype
            )
            attention_mask_new_seq[:, :sl] = attention_mask  # copy the original mask back

        # Mask out the image token locations in the labels
        text_labels_new_seq = None
        if text_labels is not None:
            text_labels_new_seq = torch.full(
                (bs, new_seq_len),
                fill_value=LABEL_IGNORE_INDEX,  # Ignore image positions for loss calculation
                device=self.device,
                dtype=text_labels.dtype,
            )  # (BS, SL + ISL - 1)
            text_labels_new_seq.scatter_(
                1, text_indices_new_seq, text_labels[text_mask_orig_seq].view(bs, -1)
            )

        # Add our own position_ids
        position_ids_new_seq = (attention_mask_new_seq.cumsum(-1) - 1).masked_fill_(
            (attention_mask_new_seq == 0), 1
        )
        return (
            output_embeds,
            attention_mask_new_seq,  # type: ignore
            text_labels_new_seq,
            position_ids_new_seq,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor | PILImage | None = None,
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,  # (BS, SQ_LEN)
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        if inputs_embeds is not None:
            raise NotImplementedError("input_embeds is not yet supported")

        if inputs_embeds is None:
            do_merge_text_and_image = pixel_values is not None and input_ids.shape[1] != 1
            do_generate_with_cache = (
                past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1
            )

            # 1. Extract text embeddings
            # Note that image_id has not been removed so need to add a dummy embed for it
            inputs_embeds: torch.Tensor = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if do_merge_text_and_image:
                # a. Extract image embeddings
                projected_patch_embeddings = self.encode_and_project_image(pixel_values)
                if self.img_pos_embed is not None:
                    projected_patch_embeddings = self.img_pos_embed(projected_patch_embeddings)

                # b. Do the merge
                inputs_embeds, attention_mask, labels, position_ids = (
                    self.merge_text_and_image_tokens(  # type: ignore
                        input_ids=input_ids,
                        text_embeds=inputs_embeds,
                        image_patch_embeds=projected_patch_embeddings,
                        attention_mask=attention_mask,
                        text_labels=labels,
                    )
                )

            elif do_generate_with_cache:
                # mask out hidden states that are not attended in the first layer
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]
                batch_index, non_attended_tokens = torch.where(
                    first_layer_past_key_value.float().sum(-2) == 0
                )

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat(
                    (extended_attention_mask, attention_mask[:, -target_length:]), dim=1
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        return self.language_model(
            input_ids=None,  # we already have the text embeddings
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # Taken directly from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py#L517
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        pixel_values=None,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = (
                cache_position[0]
                if cache_position is not None
                else past_key_values.get_seq_length()
            )
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = (
                past_length
                if max_cache_length is None
                else torch.min(max_cache_length, past_length)
            )

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.image_token_id in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_length, device=input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "pixel_values": pixel_values,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
