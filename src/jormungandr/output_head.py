from torch import nn, Tensor
from transformers import DetrForObjectDetection
from jormungandr.config.configuration import OutputHeadConfig
from jormungandr.utils.model_fetcher import fetch_detr_model


class FCNNPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    """

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        config: OutputHeadConfig = OutputHeadConfig(),
    ):
        super().__init__()
        detr = fetch_detr_model(
            model_name=model_name,
            is_pre_trained=config.use_pre_trained,
        )
        self.class_labels_classifier = detr.class_labels_classifier
        self.bbox_predictor = detr.bbox_predictor

        if config.freeze_prediction_head:
            for param in self.class_labels_classifier.parameters():
                param.requires_grad = False
            for param in self.bbox_predictor.parameters():
                param.requires_grad = False

    def forward(
        self,
        decoder_output: Tensor,
        labels: list[dict] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            decoder_output: Tensor of shape (batch_size, num_queries, hidden_dim)
            labels (`list[Dict]` of len `(batch_size,)`, *optional*):
                Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
                following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
                respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
                in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
        Returns:
            class_labels: Tensor of shape (batch_size, num_queries, num_classes)
            bbox_coordinates: Tensor of shape (batch_size, num_queries, 4) with normalized center coordinates, height and width
        """
        class_labels = self.class_labels_classifier(decoder_output)
        bbox_coordinates = self.bbox_predictor(decoder_output).sigmoid()

        return class_labels, bbox_coordinates
