from torch import nn, Tensor
import torch

from jormungandr.encoder import MambaEncoder
from jormungandr.detr_decoder import DETRDecoder
from jormungandr.output_head import FCNNPredictionHead
from jormungandr.backbone import Backbone
from jormungandr.embedder import Embedder, DetrSinePositionEmbedding
from jormungandr.config.configuration import (
    JormungandrConfig,
    DecoderConfig,
    EncoderConfig,
)


class Jormungandr(nn.Module):
    """
    Jormungandr is an novel end-to-end video object detection system that leverages the Spatial-Temporal Mamba architecture to detect objects across video frames.
    """

    def __init__(
        self,
        backbone: Backbone = Backbone(),
        embedder: Embedder | None = None,
        model_dimension: int = 256,
        variant="Jormungandr-b",
        device: torch.device | str = "cuda",
        config: JormungandrConfig = JormungandrConfig(),
    ):
        super(Jormungandr, self).__init__()
        self.device = device

        # Backbone
        self.backbone = backbone.to(device)
        self.embedder = (
            embedder
            if embedder is not None
            else DetrSinePositionEmbedding(num_position_features=model_dimension // 2)
        )
        assert self.embedder is not None, (
            "Embedder should not be None after initialization"
        )

        self.embedder = self.embedder.to(device)

        # Encoders
        self.spatial_encoder = MambaEncoder(
            model_dimension=model_dimension,
            hidden_state_dim=config.spatial_encoder.hidden_state_dim,
            num_layers=config.spatial_encoder.num_layers,
        ).to(device)
        self.temporal_encoder = MambaEncoder(
            model_dimension=model_dimension,
            hidden_state_dim=config.temporal_encoder.hidden_state_dim,
            num_layers=config.temporal_encoder.num_layers,
        ).to(device)

        # Decoder + Output Head
        self.decoder = DETRDecoder(
            decoder_config=config.decoder,
        ).to(device)

        self.output_head = FCNNPredictionHead().to(device)

    def forward(
        self,
        frames_pixel_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Sequence-to-sequence forward pass for the Jormungandr model.
        This method processes a batch of video frames and produces class labels and bounding box coordinates for object detection.
        Args:
            frames_pixel_values (Tensor): A tensor of shape (batch_size, num_frames, channels, height, width) containing the pixel values of the input video frames.
        Returns:
            class_labels (Tensor): A tensor of shape (batch_size, num_queries, num_classes) containing the predicted class probabilities for each query.
            bbox_coordinates (Tensor): A tensor of shape (batch_size, num_queries, 4) containing the predicted bounding box coordinates for each query, where the last dimension represents (x_center, y_center, width, height) normalized to [0, 1].
        """
        frames_pixel_values = frames_pixel_values.to(self.device)
        # Backbone

        # Extract Spatial features from each frame using the Spatial encoders with weight sharing

        # Generate position embeddings for each frame and flatten the spatial features of all frames

        # Extract Temporal features across frames using the Temporal encoders

        # Sequence to sequence prediction using the decoders with weight sharing
        decoder_output = self.decoder.forward(
            encoder_output=encoder_outputs,
            position_embedding=position_embedding,
            encoder_mask_flattened=flattened_mask,
        )

        # Detection Heads with weight sharing
        class_labels, bbox_coordinates = self.output_head.forward(decoder_output)
        return class_labels, bbox_coordinates
