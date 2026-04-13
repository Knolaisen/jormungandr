from torch import nn, Tensor
import torch

from jormungandr.encoder import MambaEncoder, DETREncoder
from jormungandr.detr_decoder import DETRDecoder
from jormungandr.output_head import FCNNPredictionHead
from jormungandr.backbone import Backbone
from jormungandr.embedder import Embedder, DetrSinePositionEmbedding
from jormungandr.config.configuration import FafnirConfig, DecoderConfig, EncoderConfig


class Fafnir(nn.Module):
    def __init__(
        self,
        backbone: Backbone | None = None,
        embedder: Embedder | None = None,
        model_dimension: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_classes: int = 10,
        variant="fafnir-b",
        device: torch.device | str = "cuda",
        config: FafnirConfig = FafnirConfig(),
    ):
        super(Fafnir, self).__init__()
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

        # Encoder
        self.encoder = None
        match config.encoder.encoder_type.lower():
            case "mamba":
                self.encoder = MambaEncoder(
                    model_dimension=model_dimension,
                    hidden_state_dim=config.encoder.hidden_state_dim,
                    num_layers=config.encoder.num_layers,
                ).to(device)
            case "detr":
                self.encoder = DETREncoder(
                    model_name=config.detr_name,
                    use_pre_trained=config.encoder.use_pre_trained,
                    num_layers=config.encoder.num_layers,
                ).to(device)
            case _:
                raise ValueError(f"Unsupported encoder type: {config.encoder.type}")

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
        pixel_values: Tensor,
        pixel_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pixel_values = pixel_values.to(self.device)
        # Backbone
        feature_maps, mask = self.backbone.forward(pixel_values, pixel_mask)
        # Encoder
        feature_map_shape = feature_maps.shape
        position_embedding = self.embedder.forward(
            shape=feature_map_shape,
            device=self.device,
            dtype=feature_maps.dtype,
            mask=mask,
        )

        projected_feature_maps = self.backbone.project_feature_maps(feature_maps)

        # Flatten H and W into sequence length, and permute to (batch_size, sequence_length, model_dimension)
        flattened_feature_maps = projected_feature_maps.flatten(2).permute(0, 2, 1)
        flattened_mask = mask.flatten(1)

        encoder_outputs = self.encoder.forward(
            flattened_feature_maps,
            position_embedding=position_embedding,
            pixel_mask=flattened_mask,
        )

        # Decoder
        decoder_output, intermediate = self.decoder.forward(
            encoder_output=encoder_outputs,
            position_embedding=position_embedding,
            encoder_mask_flattened=flattened_mask,
        )

        # Detection Head
        class_labels, bbox_coordinates = self.output_head.forward(decoder_output)
        return class_labels, bbox_coordinates, intermediate
