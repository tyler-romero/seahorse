import time

import torch
from liger_kernel.transformers import apply_liger_kernel_to_phi3
from peft.config import PeftConfig
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from pydantic import BaseModel, Field
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from seahorse.config.configuration_seahorse import SeahorseConfig
from seahorse.models.seahorse import SeahorseModel
from seahorse.models.vision_encoder import TimmEncoder


class LanguageModelConfig(BaseModel):
    language_model: str = "microsoft/Phi-3.5-mini-instruct"
    from_pretrained_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    use_liger_kernel: bool = True
    use_torch_compile: bool = False


class VisionEncoderConfig(BaseModel):
    timm_model: str = "vit_base_patch16_siglip_224.webli"
    output_layer: int = -1


DEFAULT_LORA_CONFIG = LoraConfig(
    r=8,  # 128 caused divergence
    lora_alpha=8,
    # A regex that matches the names of linear layers within the language model
    target_modules=r"^language_model.*(?:down_proj|gate_up_proj|o_proj|qkv_proj)$",
    modules_to_save=[
        "vision_projector",
        "img_pos_embed",
        "language_model.lm_head",
        "language_model.model.embed_tokens",
    ],
    lora_dropout=0.05,
    bias="none",
    init_lora_weights="pissa_niter_15",
    use_dora=False,  # using dora led to no improvement but was 2x as slow
    task_type="CAUSAL_LM",
)


class ModelingConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True  # for SeahorseConfig

    llm_config: LanguageModelConfig = Field(default_factory=LanguageModelConfig)
    vision_encoder_config: VisionEncoderConfig = Field(default_factory=VisionEncoderConfig)
    seahorse_config: SeahorseConfig = Field(default_factory=SeahorseConfig)
    peft_config: PeftConfig | None = Field(default=DEFAULT_LORA_CONFIG)
    checkpoint_path: str | None = (
        None  # as opposed to "resume_from_checkpoint", this doesnt load other state
    )


def construct_language_model(
    llm_config: LanguageModelConfig,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    language_model = AutoModelForCausalLM.from_pretrained(
        llm_config.language_model,
        **llm_config.from_pretrained_kwargs,
    )

    if llm_config.use_torch_compile:
        language_model = torch.compile(language_model)

    if llm_config.use_liger_kernel:
        # Monkey-patch HF Phi3 with LigerKernel. Must occur after torch.compile to avoid errors
        print("[Patch 🔧] Patching Phi3 with LigerKernel")
        apply_liger_kernel_to_phi3()

    tokenizer = AutoTokenizer.from_pretrained(
        llm_config.language_model, revision=llm_config.from_pretrained_kwargs.get("revision")
    )
    return language_model, tokenizer


def construct_vision_encoder(vision_encoder_config: VisionEncoderConfig) -> TimmEncoder:
    vision_encoder = TimmEncoder(
        timm_model=vision_encoder_config.timm_model, output_layer=vision_encoder_config.output_layer
    )
    return vision_encoder


def construct_seahorse(
    model_config: ModelingConfig | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> SeahorseModel:
    start = time.time()
    if model_config is None:
        model_config = ModelingConfig()

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    vision_encoder = construct_vision_encoder(model_config.vision_encoder_config)
    language_model, tokenizer = construct_language_model(model_config.llm_config)

    model = SeahorseModel(
        vision_encoder=vision_encoder,
        language_model=language_model,
        tokenizer=tokenizer,
        config=model_config.seahorse_config,
    ).to(device, dtype)  # type: ignore

    peft_config = model_config.peft_config
    if peft_config is not None:
        print("[Trainable 🔥] Applying PEFT to Seahorse")
        print(peft_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if model_config.checkpoint_path is not None:
        print(f"[Checkpoint 📦] Loading Seahorse from {model_config.checkpoint_path}")
        state_dict = torch.load(model_config.checkpoint_path, map_location=device)
        if peft_config is not None:
            model = set_peft_model_state_dict(model, state_dict)
        else:
            model.load_state_dict(state_dict)

    print(f"SeahorseModel constructed in {time.time() - start:.2f}s on {device} with {dtype}")
    return model  # type: ignore
