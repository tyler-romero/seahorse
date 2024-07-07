import torch
from peft.mapping import get_peft_model

from seahorse.config.experiment_config import ModelingConfig
from seahorse.models.seahorse import SeahorseModel


def construct_seahorse(
    model_config: ModelingConfig,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> SeahorseModel:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    model = SeahorseModel(model_config.seahorse_config).to(device, dtype)

    peft_config = model_config.peft_config
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model
