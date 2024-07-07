from peft.config import PeftConfig
from peft.tuners.lora import LoraConfig
from pydantic import BaseModel, Field

from seahorse.config.configuration_seahorse import SeahorseConfig
from seahorse.train.seahorse_trainer import SeahorseTrainingArguments


class ModelingConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True  # for SeahorseConfig

    seahorse_config: SeahorseConfig = Field(default_factory=SeahorseConfig)
    peft_config: PeftConfig | None = LoraConfig(
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


class RunConfig(BaseModel):
    modeling_config: ModelingConfig
    training_arguments: SeahorseTrainingArguments
    # TODO: data_arguments: SeahorseDataArguments
