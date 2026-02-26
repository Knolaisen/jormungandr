
from jormungandr.mamba_encoder import MambaEncoder
from torch import nn

class DETRDecoder(nn.Module):
    def __init__(self, num_queries: int = 16, hidden_dim: int = 16):
        super(DETRDecoder, self).__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # Additional layers can be added here


class Fafnir(nn.Module):

    def __init__(self,
                 backbone: nn.Module,
                 num_encoders: int = 6,
                 num_decoders: int = 6,
                 num_classes: int = 10,
                 num_queries: int = 16,
                 variant="fafnir-b",
                 pretrained=True,
                 aux_loss=False
                 config=None
                 ):
        super(Fafnir, self).__init__()
        self.pretrained = pretrained
        self.variant = variant
        # Backbone
        self.backbone = backbone
        # Encoder
        self.mamba_encoder = MambaEncoder(config.encoder)
        self.aux_loss = aux_loss

        # Decoder
        self.decoder = nn.ModuleList([DETRDecoder(num_queries=num_queries) for _ in range(num_decoders)])
        # Classification head
        self.obb_head = nn.Linear(16, num_classes)


    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        # Encoder
        for encoder in self.mamba_encoders:
            features = encoder(features)
        # Decoder
        
        # Additional decoder processing can be added here
        # Classification head
        outputs = self.obb_head(queries)
        return outputs