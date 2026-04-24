from torch import nn, Tensor
import torch

from jormungandr.encoder import MambaEncoder
from jormungandr.detr_decoder import DETRDecoder
from jormungandr.output_head import FCNNPredictionHead
from jormungandr.backbone import Backbone
from jormungandr.embedder import (
    Embedder,
    DetrSinePositionEmbedding,
    TemporalSinePositionEmbedding,
)
from jormungandr.config.configuration import (
    JormungandrConfig,
)


class Jormungandr(nn.Module):
    """
    Jormungandr is an novel end-to-end video object detection system that leverages the Spatial-Temporal Mamba architecture to detect objects across video frames.
    """

    def __init__(
        self,
        device: torch.device | str = "cuda",
        config: JormungandrConfig = JormungandrConfig(),
    ):
        super(Jormungandr, self).__init__()
        self.device = device

        # Backbone
        self.backbone = Backbone(
            model_name=config.detr_name,
            freeze_backbone=config.backbone.freeze_backbone,
        ).to(device)
        self.spatial_embedder: Embedder = DetrSinePositionEmbedding(
            num_position_features=config.model_dimension // 2,
        ).to(device)
        self.temporal_embedder: Embedder = TemporalSinePositionEmbedding(
            num_position_features=config.model_dimension // 2,
        ).to(device)

        # Encoders
        self.spatial_encoder = MambaEncoder(
            model_dimension=config.model_dimension,
            hidden_state_dim=config.spatial_encoder.hidden_state_dim,
            num_layers=config.spatial_encoder.num_layers,
        ).to(device)
        self.temporal_encoder = MambaEncoder(
            model_dimension=config.model_dimension,
            hidden_state_dim=config.temporal_encoder.hidden_state_dim,
            num_layers=config.temporal_encoder.num_layers,
        ).to(device)

        # Decoder + Output Head
        self.decoder = DETRDecoder(
            model_name=config.detr_name,
            decoder_config=config.decoder,
        ).to(device)

        self.output_head = FCNNPredictionHead(
            model_name=config.detr_name,
            config=config.output_head,
        ).to(device)

    def forward(
        self,
        frames_pixel_values: Tensor,
        pixel_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """
        Sequence-to-sequence forward pass for the Jormungandr model.
        This method processes a batch of video frames and produces class labels and bounding box coordinates for object detection.
        Args:
            frames_pixel_values (Tensor): A tensor of shape (num_frames, channels, height, width) containing the pixel values of the input video frames.
            pixel_mask (Tensor | None): An optional tensor of shape (num_frames, height, width) indicating valid pixels (1 for valid, 0 for padding). If None, all pixels are considered valid.
        Returns:
            class_labels (Tensor): A tensor of shape (num_frames, num_queries, num_classes) containing the predicted class probabilities for each query.
            bbox_coordinates (Tensor): A tensor of shape (num_frames, num_queries, 4) containing the predicted bounding box coordinates for each query, where the last dimension represents (x_center, y_center, width, height) normalized to [0, 1].
            intermediate_outputs (Tensor | None): A tensor containing intermediate outputs from the model.
        """
        if frames_pixel_values.ndim != 4:
            raise ValueError(
                f"Expected frames_pixel_values to have 4 dimensions (num_frames, channels, height, width), but got {frames_pixel_values.ndim} dimensions."
            )

        batch_size = 1

        frames_pixel_values = frames_pixel_values.to(self.device)
        # Backbone
        feature_maps, mask = self.backbone.forward(frames_pixel_values, pixel_mask)
        projected_feature_maps = self.backbone.project_feature_maps(feature_maps)

        # Flatten H and W into sequence length, and permute to (num_frames, sequence_length, model_dimension)
        flattened_feature_maps = projected_feature_maps.flatten(2).permute(0, 2, 1)
        flattened_mask = mask.flatten(1)

        # Generate position embeddings for each frame
        feature_map_shape = feature_maps.shape
        position_embedding = self.spatial_embedder.forward(
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
        temporal_input = spatial_features.reshape(
            batch_size, num_frames * sequence_length, model_dimension
        )

        # add time positional embeddings to the temporal input
        temporal_position_embedding = self.temporal_embedder.forward(
            shape=temporal_input.shape,
            device=self.device,
            dtype=temporal_input.dtype,
        )
        temporal_input = temporal_input + temporal_position_embedding

        # Extract Temporal features across frames using the Temporal encoders
        temporal_features = self.temporal_encoder.forward(
            temporal_input,
        )

        # Reshape temporal features back to (num_frames, sequence_length, model_dimension) for the decoder
        encoder_outputs = temporal_features.reshape(
            num_frames, sequence_length, model_dimension
        )

        # Sequence to sequence prediction using the decoders with weight sharing
        decoder_output, intermediate = self.decoder.forward(
            encoder_output=encoder_outputs,
            position_embedding=position_embedding,
            encoder_mask_flattened=flattened_mask,
        )

        # Detection Heads with weight sharing
        class_labels, bbox_coordinates = self.output_head.forward(decoder_output)
        return class_labels, bbox_coordinates, intermediate
