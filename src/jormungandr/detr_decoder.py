import torch
from torch import nn, Tensor
from transformers import DetrForObjectDetection

from jormungandr.utils.model_fetcher import fetch_detr_model
from jormungandr.config.configuration import DecoderConfig


class DETRDecoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        decoder_config: DecoderConfig = DecoderConfig(),
    ):
        super(DETRDecoder, self).__init__()
        self.decoder_config = decoder_config

        if self.decoder_config.num_queries is not None:
            self.query_position_embeddings = nn.Embedding(
                self.decoder_config.num_queries, self.decoder_config.hidden_dim
            )
        else:
            self.query_position_embeddings = fetch_detr_model(
                model_name
            ).model.query_position_embeddings

        # Additional layers can be added here

        self.decoder = fetch_detr_model(model_name).model.decoder

        if self.decoder_config.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        encoder_output: Tensor,
        position_embedding: Tensor,
        encoder_mask_flattened: Tensor | None = None,
        decoder_inputs_embeds: Tensor | None = None,
        decoder_attention_mask: torch.FloatTensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            encoder_output: Tensor of shape (batch_size, sequence_length, hidden_dim)
            position_embedding: Tensor of shape (batch_size, sequence_length, hidden_dim)
            decoder_inputs_embeds: Optional tensor of shape (batch_size, num_queries, hidden_dim)
        Returns:
            Tensor of shape (batch_size, num_queries, hidden_dim)
        """

        batch_size = encoder_output.shape[0]
        object_queries_position_embeddings = (
            self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        )
        # Use decoder_inputs_embeds as queries if provided, otherwise initialize with zeros
        if decoder_inputs_embeds is not None:
            queries = decoder_inputs_embeds
        else:
            queries = torch.zeros_like(object_queries_position_embeddings)

        decoder_output = self.decoder.forward(
            inputs_embeds=queries,
            encoder_hidden_states=encoder_output,
            spatial_position_embeddings=position_embedding,
            object_queries_position_embeddings=object_queries_position_embeddings,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_mask_flattened,
        )
        decoder_final_output = decoder_output[0]
        decoder_intermediate_outputs = decoder_output[1]

        return decoder_final_output, decoder_intermediate_outputs
