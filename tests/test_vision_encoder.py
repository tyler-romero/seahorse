import torch

from seahorse.data.utils import random_pil
from seahorse.models.vision_encoder import TimmEncoder


def test_timm_vision_encoder_with_clip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    timm_model = "vit_base_patch16_clip_224.openai"
    timm_encoder = TimmEncoder(timm_model=timm_model).to(device, dtype)

    rand_img = random_pil()

    processed_img = timm_encoder.preprocess(rand_img)
    assert isinstance(processed_img, torch.Tensor)

    out0 = timm_encoder.forward(processed_img.to(device, dtype))  # type: ignore
    out1 = timm_encoder.forward(rand_img)
    out2 = timm_encoder.forward([rand_img, rand_img])

    assert out0.shape == out1.shape == torch.Size([1, 14, 14, 768])
    assert out2.shape == torch.Size([2, 14, 14, 768])
    assert timm_encoder.timm_model.num_features == 768


def test_timm_vision_encoder_with_siglip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    timm_model = "vit_base_patch16_siglip_gap_224.webli"
    timm_encoder = TimmEncoder(timm_model=timm_model).to(device, dtype)

    rand_img = random_pil()

    processed_img = timm_encoder.preprocess(rand_img)  # no batch dim
    assert isinstance(processed_img, torch.Tensor)

    out0 = timm_encoder.forward(processed_img)
    out1 = timm_encoder.forward(rand_img)
    out2 = timm_encoder.forward([rand_img, rand_img])

    assert out0.shape == out1.shape == torch.Size([1, 14, 14, 768])
    assert out2.shape == torch.Size([2, 14, 14, 768])
    assert timm_encoder.timm_model.num_features == 768


def test_timm_vision_encoder_black_and_white_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    timm_encoder = TimmEncoder().to(device, dtype)

    rand_img = random_pil(mode="L")  # black and white PIL image (1 channel)

    processed_img = timm_encoder.preprocess(rand_img)
    assert isinstance(processed_img, torch.Tensor)

    out0 = timm_encoder.forward(processed_img.to(device, dtype))  # type: ignore
    out1 = timm_encoder.forward(rand_img)
    out2 = timm_encoder.forward([rand_img, rand_img])

    assert out0.shape == out1.shape == torch.Size([1, 14, 14, 768])
    assert out2.shape == torch.Size([2, 14, 14, 768])
    assert timm_encoder.timm_model.num_features == 768
