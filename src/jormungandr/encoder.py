from typing import Protocol
from mamba_ssm import Mamba
from torch import nn, Tensor
import torch

from jormungandr.utils.model_fetcher import fetch_detr_model


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
        model_dimension: int = 256,
        hidden_state_dim: int = 16,
        num_layers: int = 6,
    ):
        super(MambaEncoder, self).__init__()
        if num_layers < 0:
            raise ValueError("num_layers cant be negative")
        if model_dimension < 1:
            raise ValueError("model_dimension must be at least 1")
        if hidden_state_dim < 1:
            raise ValueError("hidden_state_dim must be at least 1")

        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=model_dimension,
                    d_state=hidden_state_dim,
                )
                for _ in range(self.num_layers)
            ]
        )
        # Per-layer norms (pre-norm architecture)
        self.norms = nn.ModuleList(
            [nn.RMSNorm(model_dimension) for _ in range(self.num_layers)]
        )
        self.final_norm = nn.RMSNorm(model_dimension)
 
    def forward(
        self,
        x: Tensor,
        position_embedding: Tensor | None = None,
        pixel_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, model_dimension)
        Returns:
            Tensor of shape (batch_size, model_dimension)
        """
        for layer, norm in zip(self.layers, self.norms):
            residual = x
 
            # Pre-norm
            normed = norm(x)
 
            # Inject position into the layer input — NOT into the residual.
            # This is the Mamba analog of DETR adding pos to Q and K:
            # position influences the layer's processing (selective scan gating)
            # but the residual stream (analogous to V) stays position-free.
            if position_embedding is not None:
                layer_input = normed + position_embedding
            else:
                layer_input = normed
 
            # Mamba selective scan
            layer_output = layer(layer_input)
 
            # Zero padded positions on the layer output (before residual add)
            if pixel_mask is not None:
                layer_output = layer_output * pixel_mask.unsqueeze(-1)
 
            # Residual connection — x never directly receives position_embedding
            x = residual + layer_output
 
        return self.final_norm(x)


class DETREncoder(nn.Module, Encoder):
    def __init__(self, model_name: str = "facebook/detr-resnet-50", use_pre_trained: bool = True, num_layers: int = 6):
        super(DETREncoder, self).__init__()
        self.encoder = fetch_detr_model(model_name, is_pre_trained=use_pre_trained, num_encoder_layers=num_layers).model.encoder

        for layer in self.encoder.layers:
            layer.training = True

    def forward(self, x: Tensor, position_embedding: Tensor | None = None, pixel_mask: Tensor | None = None) -> Tensor:
        encoder_outputs = self.encoder.forward(
            x, spatial_position_embeddings=position_embedding, attention_mask=pixel_mask
        )
        return encoder_outputs.last_hidden_state
