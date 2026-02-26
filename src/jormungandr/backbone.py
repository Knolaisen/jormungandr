"""
Wrapper around backbone models.
One should be able to swap out backbone to different models, e.g. ResNet, Swin, etc., without affecting the rest of the architecture.

"""
from transformers import DetrForObjectDetection
from torch import nn


class Backbone(nn.Module):
    def __init__(self, model_name: str = "facebook/detr-resnet-50"):
        super(Backbone, self).__init__()
        self.backbone = DetrForObjectDetection.from_pretrained(model_name).model.backbone
        self.input_projection 

                
    def get_output_shape() -> tuple[int]:
        # Return the output shape of the backbone
        pass

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, height, width)
        Returns:
            flattened_features (batch_size, sequence_length, model_dimension):
                Flattened features from the backbone model.
            flattened_mask (batch_size, sequence_length):
                Mask indicating valid features.
        """
        pass