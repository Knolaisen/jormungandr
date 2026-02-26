
from jormungandr.mamba_encoder import MambaEncoder
from jormungandr.backbone import Backbone
from jormungandr.detr_decoder import DETRDecoder
from jormungandr.output_head import MLPPredictionHead
from jormungandr.embedder import DetrSinePositionEmbedding
from torch import nn, Tensor
import torch



class Fafnir(nn.Module):

    def __init__(self,
                 backbone: Backbone,
                 embedder: DetrSinePositionEmbedding = None,
                 model_dimension: int = 256,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 num_classes: int = 10,
                 num_queries: int = 16,
                 variant="fafnir-b", 
                 device: torch.device | str = "cuda"  
                 ):
        super(Fafnir, self).__init__()
        self.device = device

        # Backbone
        self.backbone = backbone.to(device)
        self.embedder = embedder.to(device) if embedder is not None else DetrSinePositionEmbedding(num_position_features=model_dimension // 2).to(device)
        # Encoder
        self.mamba_encoder = MambaEncoder(model_dimension=model_dimension, num_layers=num_encoder_layers).to(device)

      
    def forward(self, pixel_values: Tensor) -> Tensor:
        pixel_values = pixel_values.to(self.device)
        # Backbone
        feature_maps, mask = self.backbone.forward(pixel_values)
        # Encoder
       
        
        feature_map_shape = feature_maps.shape
        position_embedding = self.embedder.forward(shape=feature_map_shape, device=self.device, dtype=feature_maps.dtype, mask=mask)
        
        projected_feature_maps = self.backbone.project_feature_maps(feature_maps)
        flattened_feature_maps = projected_feature_maps.flatten(2).permute(0, 2, 1) #Flatten H and W into sequence length, and permute to (batch_size, sequence_length, model_dimension)
        flattened_mask = mask.flatten(1)

        features = self.mamba_encoder.forward(flattened_feature_maps, position_embedding=position_embedding)
        return features
