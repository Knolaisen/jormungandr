from typing import Literal, Protocol

import torch
from mamba_ssm import Mamba2
from torch import Tensor, nn

from jormungandr.utils.model_fetcher import fetch_detr_model


class Encoder(Protocol):
    def forward(
        self,
        flattened_feature_maps: Tensor,
        position_embedding: Tensor | None = None,
        pixel_mask: Tensor | None = None,
    ) -> torch.Tensor: ...


class MambaEncoder(nn.Module, Encoder):
    """
    Bidirectional Mamba encoder that mirrors the DETR encoder structure.

    Each of the `num_layers` stacked blocks follows:

        x → BiMambaLayer(SSM sub-layer → FFN sub-layer) → x'

    where each sub-layer uses a pre-norm residual connection.  The final
    output is passed through a shared RMSNorm, matching the original design.

    Args:
        model_dimension:  token embedding width (d_model).
        hidden_state_dim: SSM state size (d_state in Mamba notation).
        num_layers:       number of stacked BidirectionalMambaLayers.
        ffn_expand:       FFN inner-width multiplier (default 8).
    """

    def __init__(
        self,
        model_dimension: int = 256,
        hidden_state_dim: int = 16,
        num_layers: int = 6,
        ffn_expand: int = 8,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers cant be negative")
        if model_dimension < 1:
            raise ValueError("model_dimension must be at least 1")
        if hidden_state_dim < 1:
            raise ValueError("hidden_state_dim must be at least 1")

        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                BidirectionalMambaLayer(
                    d_model=model_dimension,
                    d_state=hidden_state_dim,
                    ffn_expand=ffn_expand,
                )
                for _ in range(num_layers)
            ]
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
            x:                  (batch, seq_len, model_dimension)
            position_embedding: (batch, seq_len, model_dimension) or None
            pixel_mask:         (batch, seq_len) bool/float; 0 = padding token
        Returns:
            (batch, seq_len, model_dimension)
        """
        mask = pixel_mask.unsqueeze(-1) if pixel_mask is not None else None
        for layer in self.layers:
            x = layer(x, position_embedding=position_embedding, pixel_mask=mask)

        return self.final_norm(x)


class BidirectionalMambaLayer(nn.Module):
    """
    One layer of bidirectional Mamba followed by a position-wise FFN.

    Args:
        d_model:    token embedding width.
        d_state:    SSM hidden state size (d_state in Mamba notation).
        ffn_expand: FFN inner width multiplier relative to d_model (default 8,
                    same as DETR / vanilla Transformer).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        ffn_expand: int = 8,
    ) -> None:
        super().__init__()

        # Bidirectional SSM
        self.forward_mamba = Mamba2(d_model=d_model, d_state=d_state)
        self.backward_mamba = Mamba2(d_model=d_model, d_state=d_state)

        # Pre-norm for the SSM sub-layer.
        self.norm_ssm = nn.RMSNorm(d_model)

        # --- FFN sub-layer ---
        dim_ffn = d_model * ffn_expand
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.GELU(),
            nn.Linear(dim_ffn, d_model),
        )

        # Pre-norm for the FFN sub-layer.
        self.norm_ffn = nn.RMSNorm(d_model)

    def forward(
        self,
        x: Tensor,
        position_embedding: Tensor | None = None,
        pixel_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:                  (batch, seq_len, d_model)
            position_embedding: (batch, seq_len, d_model) or None —
                                added to the SSM input only, not the FFN.
            pixel_mask:         (batch, seq_len) float/bool mask; 0 = padding —
                                applied to the SSM output before the residual add.
        Returns:
            (batch, seq_len, d_model)
        """
        # Bidirectional SSM with pre-norm and residual connection.
        residual = x
        h = self.norm_ssm(x)

        scan_input = h + position_embedding if position_embedding is not None else h

        # Zero-out padding positions on the scan input, ensuring the SSM doesn't attend to them.
        if pixel_mask is not None:
            scan_input = scan_input * pixel_mask

        fwd = self.forward_mamba(scan_input)
        bwd = self.backward_mamba(scan_input.flip(1)).flip(1)
        ssm_out = (fwd + bwd) / 2

        if pixel_mask is not None:
            ssm_out = ssm_out * pixel_mask

        x = residual + ssm_out

        # FFN with pre-norm with residual connection.
        ffn_out = self.ffn(self.norm_ffn(x))
        x = x + ffn_out

        return x


class DETREncoder(nn.Module, Encoder):
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        use_pre_trained: bool = True,
        num_layers: int = 6,
    ):
        super(DETREncoder, self).__init__()
        self.encoder = fetch_detr_model(
            model_name, is_pre_trained=use_pre_trained, num_encoder_layers=num_layers
        ).model.encoder

        for layer in self.encoder.layers:
            layer.training = True

    def forward(
        self,
        x: Tensor,
        position_embedding: Tensor | None = None,
        pixel_mask: Tensor | None = None,
    ) -> Tensor:
        encoder_outputs = self.encoder.forward(
            x, spatial_position_embeddings=position_embedding, attention_mask=pixel_mask
        )
        return encoder_outputs.last_hidden_state
