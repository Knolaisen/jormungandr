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
        backbone: Backbone | None = None,
        embedder: Embedder | None = None,
        model_dimension: int = 256,
        variant="Jormungandr-b",
        device: torch.device | str = "cuda",
        config: JormungandrConfig = JormungandrConfig(),
    ):
        super(Jormungandr, self).__init__()
        self.device = device

        # Backbone
        if backbone is None:
            self.backbone = Backbone(
                model_name=config.detr_name,
                freeze_backbone=config.backbone.freeze_backbone,
            ).to(device)
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
            frames_pixel_values (Tensor): A tensor of shape (num_frames, channels, height, width) containing the pixel values of the input video frames.
        Returns:
            class_labels (Tensor): A tensor of shape (num_frames, num_queries, num_classes) containing the predicted class probabilities for each query.
            bbox_coordinates (Tensor): A tensor of shape (num_frames, num_queries, 4) containing the predicted bounding box coordinates for each query, where the last dimension represents (x_center, y_center, width, height) normalized to [0, 1].
        """
        frames_pixel_values = frames_pixel_values.to(self.device)
        # Backbone
        feature_maps, mask = self.backbone.forward(frames_pixel_values)
        projected_feature_maps = self.backbone.project_feature_maps(feature_maps)
        
        # Flatten H and W into sequence length, and permute to (num_frames, sequence_length, model_dimension)
        flattened_feature_maps = projected_feature_maps.flatten(2).permute(0, 2, 1)
        flattened_mask = mask.flatten(1)


        # Generate position embeddings for each frame
        feature_map_shape = feature_maps.shape
        position_embedding = self.embedder.forward(
            shape=feature_map_shape,
            device=self.device,
            dtype=feature_maps.dtype,
            mask=mask,
        )

        # Extract Spatial features from each frame using the Spatial encoders with weight sharing
        spatial_features = self.spatial_encoder.forward(
            flattened_feature_maps,
            position_embedding=position_embedding,
            pixel_mask=flattened_mask,
        )

        # Flatten spacial features across frames to create a long sequence for the temporal encoder. We have now done temporal_sequence = [sequence_frame_1, sequence_frame_2, ..., sequence_frame_n]. Might want to experiment with other ways of flattening, e.g. interleaving pixels from different frames, or adding special tokens to indicate frame boundaries, etc.
        num_frames, sequence_length, model_dimension = spatial_features.shape
        temporal_input = spatial_features.reshape(num_frames * sequence_length, model_dimension)

        # add time positional embeddings to the temporal input
        # TODO


        # Extract Temporal features across frames using the Temporal encoders
        temporal_features = self.temporal_encoder.forward(temporal_input)

        # Reshape temporal features back to (num_frames, sequence_length, model_dimension) for the decoder
        encoder_outputs = temporal_features.reshape(num_frames, sequence_length, model_dimension)

        # Sequence to sequence prediction using the decoders with weight sharing
        decoder_output = self.decoder.forward(
            encoder_output=encoder_outputs,
            position_embedding=position_embedding,
            encoder_mask_flattened=flattened_mask,
        )

        # Detection Heads with weight sharing
        class_labels, bbox_coordinates = self.output_head.forward(decoder_output)
        return class_labels, bbox_coordinates
