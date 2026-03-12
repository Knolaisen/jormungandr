from typing import Protocol
from mamba_ssm import Mamba
from transformers import DetrForObjectDetection
from torch import nn, Tensor
import torch


class Encoder(Protocol):
    def forward(
        self,
        flattened_feature_maps: Tensor,
        position_embedding: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> torch.Tensor: ...


class MambaEncoder(nn.Module, Encoder):
    def __init__(
        self,
        model_dimension: int = 16,
        state_expansion_factor: int = 16,
        num_layers: int = 6,
    ):
        super(MambaEncoder, self).__init__()
        if num_layers < 0:
            raise ValueError("num_layers cant be negative")
        if model_dimension < 1:
            raise ValueError("model_dimension must be at least 1")
        if state_expansion_factor < 1:
            raise ValueError("state_expansion_factor must be at least 1")

        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=model_dimension,
                    d_state=state_expansion_factor,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = nn.RMSNorm(model_dimension)

    def forward(self, x: Tensor, position_embedding: Tensor | None = None) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, model_dimension)
        Returns:
            Tensor of shape (batch_size, model_dimension)
        """

        for layer in self.layers:
            x = x + position_embedding if position_embedding is not None else x
            x = layer(self.norm(x))
        return x


class DETREncoder(nn.Module, Encoder):
    def __init__(self, model_name: str = "facebook/detr-resnet-50"):
        super(DETREncoder, self).__init__()
        self.encoder = DetrForObjectDetection.from_pretrained(model_name).model.encoder

    def forward(self, x: Tensor, position_embedding: Tensor | None = None) -> Tensor:
        encoder_outputs = self.encoder.forward(
            x, spatial_position_embeddings=position_embedding
        )
        return encoder_outputs.last_hidden_state
