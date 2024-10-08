import timm
import torch
from PIL.Image import Image as PILImage
from timm.models.vision_transformer import VisionTransformer
from torch import nn


class LlavaVisionProjector(nn.Sequential):
    def __init__(self, mm_hidden_size, hidden_size, depth=2):
        # Lifted from https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py#L42
        # mlp2x_gelu is the Llava 1.5 default
        repeat = [nn.GELU(), nn.Linear(hidden_size, hidden_size)] * depth
        layers = [nn.Linear(mm_hidden_size, hidden_size)] + repeat
        super().__init__(*layers)


class TimmEncoder(nn.Module):
    def __init__(
        self,
        timm_model="vit_base_patch16_clip_224.openai",
        output_layer=-1,
        do_compile: bool = True,
    ):
        super().__init__()
        self.timm_model: VisionTransformer = timm.create_model(timm_model, pretrained=True)
        self.do_compile = do_compile
        self.output_layer = output_layer

        pretrained_cfg = timm.models.get_pretrained_cfg(timm_model, allow_unregistered=False)
        if pretrained_cfg is None:
            raise ValueError(f"No pretrained configuration found for model {timm_model}")
        pretrained_cfg = pretrained_cfg.to_dict()

        # Assumes this is a ViT, or ViT-like (processes images in patches)
        image_size = pretrained_cfg["input_size"][1]  # Assumes square images
        patch_size = self.timm_model.patch_embed.patch_size[0]  # Assumes square patches
        if not image_size % patch_size == 0:
            raise ValueError(
                f"Image size {image_size} must be divisible by patch size {patch_size}"
            )
        self.num_patches_x = self.num_patches_y = image_size // patch_size

        data_config = timm.data.resolve_data_config(  # type: ignore
            pretrained_cfg=pretrained_cfg, use_test_size=True
        )
        self.timm_transform = timm.data.transforms_factory.create_transform(**data_config)  # type: ignore

    @property
    def dtype(self):
        return next(self.timm_model.parameters()).dtype

    @property
    def device(self):
        return next(self.timm_model.parameters()).device

    def preprocess(
        self, images: PILImage | list[PILImage] | torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        preprocess a single img, or a batch of images, and return a tensor with shape (B, C, H, W)
        """
        if isinstance(images, torch.Tensor):
            return images  # Assume images are already preprocessed

        if isinstance(images, PILImage):
            images = [images]

        if isinstance(images, list):
            rgb_images = [img.convert("RGB") for img in images]
            tensors = [self.timm_transform(img) for img in rgb_images]
            return torch.stack(tensors)  # type: ignore
        else:
            raise ValueError("Input must be a PIL image, list of PIL images, or a tensor")

    @torch.compile
    def compiled_encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode is a good candiate for compile because it always takes the same input shape.
        """
        return self.encode(pixel_values)

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.timm_model.forward_intermediates(
            pixel_values,
            indices=[self.output_layer],
            return_prefix_tokens=False,  # TODO: pass CLS through too; self.timm_model.has_class_token
            intermediates_only=True,
            output_fmt="NLC",
        )
        patch_embeddings = features[0]  # We only get one output as we specified one output_layer
        patch_embeddings = patch_embeddings.view(  # Reshape to (B, T_x, T_y, C)
            -1, self.num_patches_x, self.num_patches_y, patch_embeddings.shape[-1]
        )
        return patch_embeddings

    def forward(self, images: PILImage | list[PILImage] | torch.FloatTensor) -> torch.Tensor:
        """Forward pass through the model, return patch embeddings (B, T_x, T_y, C)"""
        pixel_values = self.preprocess(images)
        pixel_values = pixel_values.to(self.device, self.dtype)
        if self.do_compile:
            return self.compiled_encode(pixel_values)
        return self.encode(pixel_values)
