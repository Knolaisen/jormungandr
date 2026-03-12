from torch import nn, Tensor
import torch

from jormungandr.encoder import MambaEncoder, DETREncoder
from jormungandr.detr_decoder import DETRDecoder
from jormungandr.output_head import FCNNPredictionHead
from jormungandr.backbone import Backbone
from jormungandr.embedder import Embedder, DetrSinePositionEmbedding


class Fafnir(nn.Module):
    def __init__(
        self,
        backbone: Backbone | None = None,
        embedder: Embedder | None = None,
        encoder_type: str = "mamba",
        model_dimension: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_classes: int = 10,
        num_queries: int | None = None,
        variant="fafnir-b",
        device: torch.device | str | None = None,
    ):
        super(Fafnir, self).__init__()

        # Backbone
        self.backbone = backbone if backbone is not None else Backbone()
        self.embedder = (
            embedder
            if embedder is not None
            else DetrSinePositionEmbedding(num_position_features=model_dimension // 2)
        )
        assert self.embedder is not None, (
            "Embedder should not be None after initialization"
        )

        # Encoder
        self.encoder = None
        match encoder_type.lower():
            case "mamba":
                self.encoder = MambaEncoder(
                    model_dimension=model_dimension, num_layers=num_encoder_layers
                )
            case "detr":
                self.encoder = DETREncoder()
            case _:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")

        # TODO: Find if hidden dim is model dimension or something else
        self.decoder = DETRDecoder(num_queries=num_queries, hidden_dim=model_dimension)

        self.output_head = FCNNPredictionHead(model_name="facebook/detr-resnet-50")

        if device is not None:
            self.to(device)

    def forward(
        self,
        pixel_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        device = pixel_values.device
        # Backbone
        feature_maps, mask = self.backbone(pixel_values)
        # Encoder
        feature_map_shape = feature_maps.shape
        position_embedding = self.embedder(
            shape=feature_map_shape,
            device=device,
            dtype=feature_maps.dtype,
            mask=mask,
        )

        projected_feature_maps = self.backbone.project_feature_maps(feature_maps)

        # Flatten H and W into sequence length, and permute to (batch_size, sequence_length, model_dimension)
        flattened_feature_maps = projected_feature_maps.flatten(2).permute(0, 2, 1)
        flattened_mask = mask.flatten(1)

        encoder_outputs = self.encoder(
            flattened_feature_maps,
            position_embedding=position_embedding,
        )

        # Decoder
        decoder_output = self.decoder(
            encoder_output=encoder_outputs,
            position_embedding=position_embedding,
            encoder_mask_flattened=flattened_mask,
        )

        # Detection Head
        class_labels, bbox_coordinates = self.output_head(decoder_output)
        return class_labels, bbox_coordinates
