
from jormungandr.mamba_encoder import MambaEncoder
from torch import nn, Tensor



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
        self.embedder = nn.Linear(2048, 16)  # Example embedding layer, adjust input/output dimensions as needed 
        # Encoder
        self.mamba_encoder = MambaEncoder(config.encoder, position_embedder=self.embedder)
        self.aux_loss = aux_loss

        # Decoder
        self.decoder = nn.ModuleList([DETRDecoder(num_queries=num_queries) for _ in range(num_decoders)])
        # Classification head
        self.obb_head = nn.Linear(16, num_classes)


    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        # Encoder
        features = features.flatten()
        position_embedding = self.embedder(features)
        features = self.mamba_encoder(features, position_embedding)
        # Decoder
        
        # Additional decoder processing can be added here
        # Classification head
        outputs = self.obb_head(queries)
        return outputs