import torch
from PIL.Image import Image as PILImage
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.phi3 import Phi3ForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from seahorse.config.configuration_seahorse import SeahorseConfig
from seahorse.models.vision_encoder import LlavaVisionProjector, TimmEncoder

# NOTE: Figure these out a bit better, see existing placeholder tokens
IGNORE_INDEX = -100
DEFAULT_PADDING_TOKEN = "<pad>"
DEFAULT_IMAGE_TOKEN = "<image>"
# https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/added_tokens.json
DEFAULT_IM_START_TOKEN = "<|placeholder1|>"
DEFAULT_IM_END_TOKEN = "<|placeholder2|>"
IMAGE_NEWLINE_TOKEN = "<|placeholder3|>"


class SeahorseModel(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Phi3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False

    def __init__(
        self,
        config: SeahorseConfig | None = None,
    ):
        self.config = config or SeahorseConfig()
        super().__init__(self.config)

        self.vision_encoder = TimmEncoder(timm_model=self.config.vision_encoder)
        self.language_model: Phi3ForCausalLM = Phi3ForCausalLM.from_pretrained(  # type: ignore
            self.config.language_model,
            device_map="cuda",  # TODO: make this configurable
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # pip install flash_attn
            revision=self.config.lm_revision,
        )
        self.vision_projector = LlavaVisionProjector(
            mm_hidden_size=self.vision_encoder.timm_model.num_features,
            hidden_size=self.language_model.get_input_embeddings().embedding_dim,  # type: ignore
        )

        self.tokenizer = self.setup_tokenizer_for_vision(self.config.language_model)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        print(f"Image token ID: {self.image_token_id}")

        self._specify_trainable_params()

    def _specify_trainable_params(self):
        # Freeze the vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Freeze the language model except for the embedding table and the lm_head
        # Note: without lm_head requiring a grad, torch.compile doesn't work
        for name, param in self.language_model.named_parameters():
            if "embed_tokens" not in name and "lm_head" not in name:
                param.requires_grad = False
        self.language_model.get_input_embeddings().requires_grad = True  # type: ignore
        self.language_model.get_output_embeddings().requires_grad = True  # type: ignore

        # Make sure the vision projector is trainable
        for param in self.vision_projector.parameters():
            param.requires_grad = True

    def setup_tokenizer_for_vision(self, model_name: str) -> PreTrainedTokenizer:
        # Phi3 tokenizer is identicle to LlamaTokenizer, except for some added tokens
        # https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            pad_token=DEFAULT_PADDING_TOKEN,  # LlamaTokenizer doesn't have a pad token by default
        )

        # https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py#L326
        num_new_tokens = 0
        num_new_tokens += tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        # num_new_tokens += tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        # if num_new_tokens > 0:
        #     input_embeddings = self.get_input_embeddings().weight.data
        #     # Init new embeddings as the average of the existing embeddings
        #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
        return tokenizer

    def tokenize_text(self, text: str | list[str]) -> "BatchEncoding":
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        images_per_ex = (tokens.input_ids == self.image_token_id).sum(dim=-1)
        if (images_per_ex > 1).any():
            raise ValueError("Multiple image tokens found in a single example")
        return tokens

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
        text_token_ids: torch.LongTensor,
        image_patch_embeds: torch.Tensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        text_labels: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.LongTensor | None, torch.LongTensor | None]:
        """
        Merge text and image tokens into a single tensor for input to the language model.
        The image token in each sequence is replaced with the image patch embeddings.
        Only one image token is allowed per sequence (currently). Alternatively, batches
        with no image tokens are also supported. But mixed batches are not allowed.

        Args:
            text_token_ids (torch.LongTensor): Tokenized text input (BS, SQ_LEN)
            text_labels (torch.LongTensor): Text token labels, if any (BS, SQ_LEN)
            image_patch_embeds (torch.Tensor | None): Image patch embeddings, if any (BS, T_x, T_y, EMBED_DIM)

        Returns:
            torch.Tensor: Merged embeddings for input to the language model (BS, SQ_LEN, EMBED_DIM)
            torch.LongTensor | None: Attention mask for the merged embeddings, if any (BS, SQ_LEN)
            torch.LongTensor | None: Merged labels for the language model, if any (BS, SQ_LEN)
        """
        # Inspired by moondream
        text_emb = self.language_model.get_input_embeddings()
        bs, sl = text_token_ids.shape
        image_mask_orig_seq = text_token_ids == self.image_token_id

        if image_mask_orig_seq.sum() == 0:  # no image tokens in any sequence
            return text_emb(text_token_ids), attention_mask, text_labels
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

        _, tile_x, tile_y, chan = image_patch_embeds.shape
        img_seq_len = tile_x * tile_y
        new_seq_len = sl + img_seq_len - 1  # remove the image token

        image_idx_orig_seq = image_mask_orig_seq.nonzero(as_tuple=True)[1]  # (BS,)
        image_patch_embeds = image_patch_embeds.view(bs, -1, chan)

        # Create indices for text embeddings, excluding the image token
        text_indices_orig_seq = torch.arange(sl, device=self.device).unsqueeze(0).expand(bs, -1)
        text_mask_orig_seq = text_token_ids != self.image_token_id  # TODO
        text_indices_orig_seq = text_indices_orig_seq[text_mask_orig_seq].view(bs, -1)

        text_embeds = text_emb(text_token_ids[text_mask_orig_seq]).view(bs, -1, chan)

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
                fill_value=IGNORE_INDEX,  # Splice in -100s for the image positions in order to ignore them
                device=self.device,
                dtype=text_labels.dtype,
            )  # (BS, SL + ISL - 1)
            text_labels_new_seq.scatter_(
                1, text_indices_new_seq, text_labels[text_mask_orig_seq].view(bs, -1)
            )
        return output_embeds, attention_mask_new_seq, text_labels_new_seq

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor | PILImage | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,  # (BS, SQ_LEN)
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        if position_ids is not None:
            raise ValueError("position_ids is not (yet) supported by SeahorseModel")
        if inputs_embeds is not None:
            raise ValueError("inputs_embeds is not supported by SeahorseModel")
        if past_key_values is not None:
            raise ValueError("past_key_values is not supported by SeahorseModel")

        projected_patch_embeddings = None
        if pixel_values is not None:
            projected_patch_embeddings = self.encode_and_project_image(pixel_values)

        input_embeds, attention_mask, labels = self.merge_text_and_image_tokens(
            text_token_ids=input_ids,
            image_patch_embeds=projected_patch_embeddings,
            attention_mask=attention_mask,
            text_labels=labels,
        )
        return self.language_model(
            input_ids=None,  # we already have the text embeddings
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
