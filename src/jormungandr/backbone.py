"""
Wrapper around backbone models.
One should be able to swap out backbone to different models, e.g. ResNet, Swin, etc., without affecting the rest of the architecture.

"""

import torch
from transformers import DetrForObjectDetection
from torch import nn, Tensor


class Backbone(nn.Module):
    def __init__(self, model_name: str = "facebook/detr-resnet-50"):
        super(Backbone, self).__init__()
        self.backbone = DetrForObjectDetection.from_pretrained(
            model_name
        ).model.backbone
        self.input_projection = DetrForObjectDetection.from_pretrained(
            model_name
        ).model.input_projection

    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            pixel_values: Tensor of shape (batch_size, channels, height, width)
        Returns:
            features (batch_size, h_0, w_0, model_dimension):
                Features from the backbone model.
            mask (batch_size, h_0, w_0):
                Mask indicating valid features.
        """
        batch_size, channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)

        vision_features = self.backbone(pixel_values, pixel_mask)
        feature_maps, mask = vision_features[-1]

        return feature_maps, mask

    def project_feature_maps(self, feature_maps: Tensor) -> Tensor:
        projected_feature_maps = self.input_projection(feature_maps)
        return projected_feature_maps
