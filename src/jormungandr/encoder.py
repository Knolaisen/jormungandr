from typing import Literal, Protocol

import torch
from mamba_ssm import Mamba
from torch import Tensor, nn

from jormungandr.utils.model_fetcher import fetch_detr_model


class Encoder(Protocol):
    def forward(
        self,
        flattened_feature_maps: Tensor,
        position_embedding: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> torch.Tensor: ...


MergeStrategy = Literal["add", "concat"]


class BidirectionalMambaLayer(nn.Module):
    """
    One layer of bidirectional Mamba.

    Runs two independent SSMs:
      - forward_mamba : processes the sequence left-to-right
      - backward_mamba: processes the sequence right-to-left (sequence is
                        flipped before the scan and flipped back after)

    The two outputs are then merged via addition (default) or concatenation
    followed by a learned projection back to `d_model`.

    Why two *separate* Mamba blocks instead of one shared block?
    Mamba's selective-scan parameters (A, B, C, Δ) are position-aware; the
    forward and backward passes capture structurally different patterns, so
    sharing weights would force both directions into the same learned
    dynamics — which hurts quality in practice (see Vim, MambaND).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        merge: MergeStrategy = "add",
    ) -> None:
        super().__init__()
        self.merge = merge
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state)

        # Only needed when merging by concatenation.
        self.merge_proj = (
            nn.Linear(d_model * 2, d_model, bias=False)
            if merge == "concat"
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        fwd = self.forward_mamba(x)

        # Flip along the sequence dimension, scan, flip back so that
        # backward_out[i] contains information from positions i..seq_len-1.
        bwd = self.backward_mamba(x.flip(1)).flip(1)

        if self.merge == "add":
            return fwd + bwd

        # concat → project
        return self.merge_proj(torch.cat([fwd, bwd], dim=-1))


class MambaEncoder(nn.Module, Encoder):
    """
    Drop-in replacement for the unidirectional MambaEncoder with full
    bidirectional context via paired forward/backward Mamba SSMs.

    Architecture (per layer, pre-norm residual):

        x  ──┬──► RMSNorm ──► (+pos_emb) ──► BidirectionalMambaLayer ──► (* mask) ──┐
             │                                                                        │
             └────────────────────────────────────────────────────────────────────(+)─► x'

    Args:
        model_dimension:  token embedding width (d_model).
        hidden_state_dim: SSM state size (d_state in Mamba notation).
        num_layers:       number of stacked BidirectionalMambaLayers.
        merge:            how to combine the two scan directions —
                          "add"    → element-wise sum (no extra parameters,
                                     works well when d_model is large enough)
                          "concat" → concatenate then linear-project back to
                                     d_model (richer but doubles inner width)
    """

    def __init__(
        self,
        model_dimension: int = 256,
        hidden_state_dim: int = 16,
        num_layers: int = 6,
        merge: MergeStrategy = "add",
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
                    merge=merge,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.RMSNorm(model_dimension) for _ in range(num_layers)]
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
            pixel_mask:         (batch, seq_len) bool/float mask; 0 = padding
        Returns:
            (batch, seq_len, model_dimension)
        """
        for layer, norm in zip(self.layers, self.norms):
            residual = x

            normed = norm(x)

            # Inject positional signal into the scan input only — not the
            # residual — mirroring how DETR adds pos to Q/K but not V.
            # Both scan directions receive the same position embedding so
            # each can gate against absolute position from its own causal
            # vantage point.
            layer_input = (
                normed + position_embedding
                if position_embedding is not None
                else normed
            )

            layer_output = layer(layer_input)

            # Zero-out padding positions before the residual add.
            if pixel_mask is not None:
                layer_output = layer_output * pixel_mask.unsqueeze(-1)

            x = residual + layer_output

        return self.final_norm(x)


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
