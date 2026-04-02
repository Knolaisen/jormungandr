from typing import Protocol

import torch
from torch import Tensor, nn
from timm.layers import DropPath
from mamba_ssm import Mamba2

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
        dim_feedforward:  FFN inner width (default 2048).
        dropout:          dropout probability for FFN (default 0.1).
        drop_path_rate:   max stochastic depth rate (default 0.1);
                          linearly increases from 0 at layer 0 to this
                          value at the last layer.

    """

    def __init__(
        self,
        model_dimension: int = 256,
        hidden_state_dim: int = 16,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers cant be negative")
        if model_dimension < 1:
            raise ValueError("model_dimension must be at least 1")
        if hidden_state_dim < 1:
            raise ValueError("hidden_state_dim must be at least 1")

        self.num_layers = num_layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList(
            [
                BidirectionalMambaLayer(
                    d_model=model_dimension,
                    d_state=hidden_state_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(num_layers)
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
        dim_feedforward: FFN inner width (default 2048).
        dropout:      dropout probability (default 0.1).
        drop_path:       stochastic depth rate for this layer (default 0.0).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        # Bidirectional SSM
        self.forward_mamba = Mamba2(d_model=d_model, d_state=d_state)
        self.backward_mamba = Mamba2(d_model=d_model, d_state=d_state)

        # Pre-norm for the SSM sub-layer.
        self.norm_ssm = nn.RMSNorm(d_model)

        # --- FFN sub-layer ---
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Pre-norm for the FFN sub-layer.
        self.norm_ffn = nn.RMSNorm(d_model)

        # Stochastic depth for both sub-layers
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

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

        x = residual + self.drop_path(ssm_out)

        # FFN with pre-norm with residual connection.
        ffn_output = self.ffn(self.norm_ffn(x))
        x = x + self.drop_path(ffn_output)

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
