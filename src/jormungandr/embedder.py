from typing import Protocol

import torch
from torch import nn
import math

# @compile_compatible_method_lru_cache(maxsize=1) https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L242


class Embedder(Protocol):
    def forward(
        self,
        shape: torch.Size,
        device: torch.device | str,
        dtype: torch.dtype,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class DetrSinePositionEmbedding(nn.Module, Embedder):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_position_features: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float | None = None,
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_position_features = num_position_features
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    # @compile_compatible_method_lru_cache(maxsize=1)
    def forward(
        self,
        shape: torch.Size,
        device: torch.device | str,
        dtype: torch.dtype,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            shape: The shape of the feature maps for which to compute the position embedding, expected to be (batch_size, channels, height, width)
            device: The device on which to create the position embedding
            dtype: The dtype of the position embedding
            mask: An optional mask tensor of shape (batch_size, height, width) where True values indicate masked positions. If None, no positions are masked.
        Returns:
            A position embedding tensor of shape (batch_size, sequence_length, hidden_size) where sequence_length is height * width and hidden_size is num_position_features * 2 (for sine and cosine components)
        """
        if mask is None:
            mask = torch.zeros(
                (shape[0], shape[2], shape[3]), device=device, dtype=torch.bool
            )
        y_embed = mask.cumsum(1, dtype=dtype)
        x_embed = mask.cumsum(2, dtype=dtype)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_position_features, dtype=torch.int64, device=device
        ).to(dtype)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_position_features
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # Flatten spatial dimensions and permute to (batch_size, sequence_length, hidden_size) format
        # expected by the encoder
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos


class DetrLearnedPositionEmbedding(nn.Module, Embedder):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    # @compile_compatible_method_lru_cache(maxsize=1)
    def forward(
        self,
        shape: torch.Size,
        device: torch.device | str,
        dtype: torch.dtype,
        mask: torch.Tensor | None = None,
    ):
        height, width = shape[-2:]
        width_values = torch.arange(width, device=device)
        height_values = torch.arange(height, device=device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).repeat(height, 1, 1),
                y_emb.unsqueeze(1).repeat(1, width, 1),
            ],
            dim=-1,
        )
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(shape[0], 1, 1, 1)
        # Flatten spatial dimensions and permute to (batch_size, sequence_length, hidden_size) format
        # expected by the encoder
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos


class TemporalSinePositionEmbedding(nn.Module, Embedder):
    def __init__(
        self,
        num_position_features: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float | None = None,
    ):
        super().__init__()
        self.num_position_features = num_position_features
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(
            self,
            shape: torch.Size,
            device: torch.device | str,
            dtype: torch.dtype,
            delta_t: float = 1.0,
        ) -> torch.Tensor:
            """
            Generate temporal sine position embeddings.
            Args:
                shape: The shape of the input tensor for which to compute the position embedding, expected to be (n_frames, sequence_length, model_dimension)
                device: The device on which to create the position embedding
                dtype: The dtype of the position embedding
                delta_t: The time interval between frames, used to compute the sine and cosine values.
                n_frames: The number of frames in the temporal sequence for which to compute the position embeddings.
            Returns:
            A position embedding tensor of shape (sequence_length * n_frames, num_position_features * 2) where num_position_features is the number of sine and cosine features for each temporal position. The first half of the features correspond to sine values and the second half correspond to cosine values.

            PE(n_f, 2i) = sin(n_f * delta_t / (10000^(2i/d_model)))
            PE(n_f, 2i+1) = cos(n_f * delta_t / (10000^(2i/d_model)))
            """

            n_frames, sequence_length, model_dimension = shape

            # Create frame indices tensor of shape (n_frames,)
            frame_indices = torch.arange(n_frames, device=device, dtype=dtype)
            dim_t = torch.arange(
                self.num_position_features, dtype=torch.int64, device=device
            ).to(dtype)

            dim_t = self.temperature ** (
                2 * dim_t / (self.num_position_features * 2) # torch.div(dim_t, 2, rounding_mode="floor")
            )

            frame_indices = frame_indices[:, None]  # Shape (n_frames, 1)
            dim_t = dim_t[None, :]  # Shape (1, num_position_features

            angles= frame_indices * delta_t / dim_t  # Shape (n_frames, num_position_features)



            pos = torch.stack((angles.sin(), angles.cos()), dim=-1) # Shape (n_frames, num_position_features, 2)
            pos = pos.flatten(-2) # Shape (n_frames, num_position_features * 2)


            pos = pos.unsqueeze(1)  # Add sequence length dimension
            pos = pos.repeat(1, sequence_length, 1)  # Repeat for each position in the sequence
            pos = pos.flatten(0, 1)  # Flatten to (n_frames * sequence_length, num_position_features * 2)

            return pos 